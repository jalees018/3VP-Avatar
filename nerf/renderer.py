import math
import numpy as np

import torch
import torch.nn as nn
import random

import raymarching as raymarching
from .utils import custom_meshgrid
from .NN import Network
from .projection.map3 import project_position, project_direction
import bvh_distance_queries

# """ from .projection.map import SurfaceAlignedConverter

# from pytorch3d.structures import Meshes, Pointclouds """

M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
dd1 = np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)


def perspective_projection(points, rotation, translation, focal_lengthx, focal_lengthy, camera_centerx, camera_centery):
    """ """ """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_lengthx
    K[:, 1, 1] = focal_lengthy
    K[:, 2, 2] = 1.0
    K[:, 0, -1] = camera_centerx
    K[:, 1, -1] = camera_centery

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)
    return projected_points[:, :, :-1]


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeRFRenderer(nn.Module):
    def __init__(
        self,
        bound=1,
        cuda_ray=False,
        density_scale=1,  # scale up deltas (or sigmas), to make the density grid more sharp. larger value than 1 usually improves performance.
        min_near=0.2,
        density_thresh=0.01,
        bg_radius=-1,
    ):
        super().__init__()

        self.bound = bound
        self.cascade = 1 + math.ceil(math.log2(bound))
        self.grid_size = 128
        self.density_scale = density_scale
        self.min_near = min_near
        self.density_thresh = density_thresh
        self.bg_radius = bg_radius  # radius of the background sphere.
        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.
        aabb_train = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        aabb_infer = aabb_train.clone()
        self.register_buffer("aabb_train", aabb_train)
        self.register_buffer("aabb_infer", aabb_infer)

        # extra state for cuda raymarching
        self.cuda_ray = cuda_ray
        if cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size**3])  # [CAS, H * H * H]
            density_bitfield = torch.zeros(
                self.cascade * self.grid_size**3 // 8, dtype=torch.uint8
            )  # [CAS * H * H * H // 8]
            self.register_buffer("density_grid", density_grid)
            self.register_buffer("density_bitfield", density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32)  # 16 is hardcoded for averaging...
            self.register_buffer("step_counter", step_counter)
            self.mean_count = 0
            self.local_step = 0

    def forward(self, x, d):
        raise NotImplementedError()

    # separated density and color query (can accelerate non-cuda-ray mode.)
    def density(self, x):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def warp_position(self, triangles, trisc, xyzs, tryii=None):
        m = bvh_distance_queries.BVH(queue_size=1024)
        distances, closest_points, closest_faces, closest_bcs = m(triangles, xyzs.unsqueeze(dim=0))
        if tryii == None:
            deformed_triangles = torch.index_select(triangles, 1, closest_faces[0])
            canonical_triangles = torch.index_select(trisc, 1, closest_faces[0])
            canon_xyzs = project_position(xyzs, deformed_triangles, canonical_triangles, tryii, closest_faces[0])
        else:
            canon_xyzs = project_position(xyzs, triangles, trisc, tryii, closest_faces[0])

        return canon_xyzs, closest_faces, distances[0]

    def warp_direction(self, triangles, trisc, closest_faces, dirs):
        deformed_triangles = torch.index_select(triangles, 1, closest_faces[0])
        canonical_triangles = torch.index_select(trisc, 1, closest_faces[0])

        #  """        deformed_neighbours = torch.index_select(triangles, 1, neighbours.reshape(-1))

        canon_xyzs = project_direction(dirs, deformed_triangles, canonical_triangles)

        return canon_xyzs

    def conditioning2(self, closest_faces, exp, lips_ids, faces_num, ones_only=False):
        B, N = closest_faces.shape[:2]
        E = exp.shape[1]
        cond = torch.ones((N, E)).cuda()
        a = torch.zeros((faces_num, 1)).float().cuda()
        a[lips_ids] = 1.0
        cond = a[closest_faces]
        return cond[0]

    def run(
        self,
        rays_o,
        rays_d,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        tryi=100,
        iddd=None,
        num_steps=128,
        upsample_steps=128,
        bg_color=None,
        perturb=False,
        **kwargs
    ):

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)
        z_val = z_val.contiguous().view(-1, num_steps)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        sample_dist = (fars - nears) / num_steps
        # if perturb:
        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_val.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]

        # query SDF and RGB
        if self.training == True or ver == None:
            animate = True
        else:
            animate = True
            scale = 1.0
        a, b, c = xyzs.shape
        debug_input_data = False
        if debug_input_data:
            print(xyzs.shape)
            inp_pose = xyzs.reshape(-1, 3)
            pp2 = inp_pose.clone().cpu().float().detach().numpy()
            print(pp2.shape)
            np.savetxt(
                "sampled_points_2_01_Ray_Sampled_Canonical_222_111.off",
                pp2,
                header="OFF\n" + str(pp2.shape[0]) + " 0 0",
                comments="",
            )
            exit()
        if animate:

            sm = Network(converter, face_id, ver)
            rays_d1 = rays_d.unsqueeze(-2).repeat(1, b, 1).reshape(-1, 3)
            inputt_111_2, id_x_111_2, local_coord, tri, tri_def, vert_uv, face_t, triangles_meshes_111_2 = (
                sm.calculate_density_color(xyzs.reshape(-1, 3).unsqueeze(0), None, None)
            )

            ###########################################################################################################################################################################
            v0, v1, v2 = triangles_meshes_111_2.unbind(-2)
            v0 = (v0).unsqueeze(-1).cuda() / scale
            v00, _, _ = triangles_meshes.unbind(-2)
            v00 = (v00).unsqueeze(-1).cuda() / scale

            local_coord = local_coord.permute((0, 2, 1))
            local_coordinates = (local_coordinates_meshes).permute((0, 2, 1))

            oncc = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(int(local_coordinates.shape[0]), 1, 1).cuda()
            can_tra = torch.cat((torch.cat((local_coordinates, v00), 2), oncc), 1)
            def_tra = torch.cat((torch.cat((local_coord, v0), 2), oncc), 1)
            tryiii = tryi
            msk_tryi = tryiii != 0
            tryiii[tryiii == 0] = 1

            oo_111 = torch.tensor([[1.0]]).repeat(int(xyzs.reshape(-1, 3).shape[0]), 1).cuda()
            pts_111 = torch.cat((xyzs.reshape(-1, 3), oo_111), 1)

            mmt = torch.bmm((can_tra), torch.inverse(def_tra))
            tt = (tryiii[id_x_111_2])[0]  ########################
            tt1 = tt.reshape(-1, 1)[:, 0]  ########################

            msk_tryi_111 = ((msk_tryi[id_x_111_2])[0].unsqueeze(-1)).unsqueeze(-1)

            cent = (v0[:, :, 0] + v1.cuda() + v2.cuda()) / 3
            cent_111 = (v0[:, :, 0])[(tt1 - 1).long()].reshape(tt.shape[0], 19, 3)
            dist = torch.norm(cent_111 - xyzs.reshape(-1, 3).unsqueeze(-2).repeat((1, 19, 1)), dim=2, keepdim=True)
            wts = torch.exp(-16.05000 * dist).unsqueeze(-1)
            mmt2 = (mmt[(tt1 - 1).long()].reshape(tt.shape[0], 19, 4, 4) * wts * msk_tryi_111).sum(dim=1) / (
                (wts * msk_tryi_111).sum(dim=1)
            )

            transform_2 = torch.bmm(mmt2, pts_111.unsqueeze(1).permute((0, 2, 1)))

            xyzs = transform_2[:, 0:3, 0]
            xyzs = xyzs.reshape(a, b, -1)

            rays_d = rays_d1.reshape(a, b, c)
        debug_input_data = False
        if debug_input_data:
            print(xyzs.shape)
            inp_pose = xyzs.reshape(-1, 3)
            pp2 = inp_pose.clone().cpu().float().detach().numpy()
            pp9 = pts_111[:, 0:3].reshape(-1, 3).clone().cpu().float().detach().numpy()
            print(pp2.shape)
            np.savetxt("sampled_points_111.off", pp2, header="OFF\n" + str(pp2.shape[0]) + " 0 0", comments="")
            np.savetxt("sampled_points_111_can.off", pp9, header="OFF\n" + str(pp9.shape[0]) + " 0 0", comments="")
            exit()
        density_outputs = self.density(xyzs.reshape(-1, 3))
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)
        z_vals = z_val
        # if upsample_steps > 0:

        #     for k, v in new_density_outputs.items():

        #     for k in density_outputs:
        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        Tr = torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
        weights = alphas * Tr  # [N, T+t]
        dirs = rays_d
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            polar = raymarching.polar_from_ray(
                rays_o, rays_d[:, 0, :].reshape(-1, 3), self.bg_radius
            )  # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_d[:, 0, :])  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360

        return {
            "depth": depth,
            "image": image,
        }

    def run_cond_test(
        self,
        rays_o,
        rays_ddd,
        model2,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        tryi=100,
        iddd=None,
        abcdefghhhh=None,
        trisc=None,
        trisd=None,
        nears=None,
        fars=None,
        msk_depth=None,
        face_msk=None,
        epoch=10,
        num_steps=128,
        upsample_steps=128,
        bg_color=None,
        perturb=False,
        **kwargs
    ):
        iddd = iddd
        prefix = rays_o.shape[:-1]
        lenn = rays_o.shape[1]
        if msk_depth.sum() != 0 or msk_depth == None:
            rays_o = (rays_o.contiguous().view(-1, 3))[msk_depth.reshape(-1, 1)[:, 0]]
            rays_ddd = (rays_ddd.contiguous().view(-1, 3))[msk_depth.reshape(-1, 1)[:, 0]]
            if face_msk is not None:
                face_msk = face_msk[msk_depth.reshape(-1, 1)[:, 0]]

            N = rays_o.shape[0]  # N = B * N, in fact
            device = rays_o.device

            # choose aabb
            aabb = self.aabb_train if self.training else self.aabb_infer

            # """ # sample steps
            nears = nears[msk_depth.reshape(-1, 1)[:, 0]]
            fars = fars[msk_depth.reshape(-1, 1)[:, 0]]

            z_val = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
            z_val = z_val.expand((N, num_steps))  # [N, T]
            z_val = nears + (fars - nears) * z_val  # [N, T], in [nears, fars]

            sample_dist = (fars - nears) / num_steps
            if perturb:
                z_val = z_val + (torch.rand(z_val.shape, device=device) - 0.5) * sample_dist
            # generate xyzs
            xyzss = rays_o.unsqueeze(-2) + rays_ddd.unsqueeze(-2) * z_val.unsqueeze(
                -1
            )  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
            xyzss = torch.min(torch.max(xyzss, aabb[:3]), aabb[3:])  # a manual clip.

            scale = 1.0
            xyzs, closest_faces, _ = self.warp_position(
                trisd.unsqueeze(0), trisc.unsqueeze(0), xyzss.reshape(-1, 3), tryi
            )
            rays_d = self.warp_direction(
                trisd.unsqueeze(0),
                trisc.unsqueeze(0),
                closest_faces,
                rays_ddd.view(-1, 1, 3).expand_as(xyzss).reshape(-1, 3),
            )
            z_vals = z_val

            jpp3 = torch.tensor(abcdefghhhh["jaw_pose"]).unsqueeze(0).cuda()
            exprs3 = torch.tensor(abcdefghhhh["expressions"]).unsqueeze(0).cuda()
            de = torch.cat((exprs3, jpp3), dim=2).unsqueeze(-1).unsqueeze(-1).cuda()

            jpp3 = torch.tensor(abcdefghhhh["jaw_posei"]).unsqueeze(0).cuda()
            exprs3 = torch.tensor(abcdefghhhh["expressionsi"]).unsqueeze(0).cuda()
            de_i = torch.cat((exprs3, jpp3), dim=2).unsqueeze(-1).unsqueeze(-1).cuda()

            jpp3 = torch.tensor(abcdefghhhh["jaw_posef"]).unsqueeze(0).cuda()
            exprs3 = torch.tensor(abcdefghhhh["expressionsf"]).unsqueeze(0).cuda()
            de_f = torch.cat((exprs3, jpp3), dim=2).unsqueeze(-1).unsqueeze(-1).cuda()

            cond = (
                self.conditioning2(
                    closest_faces, de[0, :, :, 0, 0], local_coordinates_meshes, trisd.shape[0], ones_only=False
                )
            ).unsqueeze(0)

            if model2 is not None:  # self is the mapping network, model2 the stage-1 model
                cond2 = (self.forward(de) + self.forward(de_i) + self.forward(de_f)) / 3
            else:  # stage-1 model only: use its learned per-frame latents
                model2 = self
                cond2 = None

            density_outputs_c, _, _, color_code = model2.density(
                xyzs.reshape(-1, 3), iddd, epoch, cond, abcdefghhhh["json_file"], 1, cond2
            )
            for k, v in density_outputs_c.items():
                density_outputs_c[k] = v.view(N, num_steps, -1)
            # if upsample_steps > 0:

            #     for k, v in new_density_outputs.items():

            #     for k in density_outputs:
            deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
            deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
            ssgma = density_outputs_c["sigma"].squeeze(-1)
            alphas = 1 - torch.exp(-deltas * self.density_scale * ssgma)  # [N, T+t]
            alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
            Tr = torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
            weights = alphas * Tr  # [N, T+t]
            dirs = rays_d
            for k, v in density_outputs_c.items():
                density_outputs_c[k] = v.view(-1, v.shape[-1])

            mask = weights > 1e-4  # hard coded
            rgbs = model2.color(
                xyzs.reshape(-1, 3),
                dirs.reshape(-1, 3),
                mask=mask.reshape(-1),
                color_code=color_code,
                **density_outputs_c
            )
            rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]

            # calculate weight_sum (mask)
            weights_sum = weights.sum(dim=-1)  # [N]

            # calculate depth
            depth2 = torch.sum(weights * z_vals, dim=-1)

            # calculate color
            corrective2 = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

            # mix background color
            if self.bg_radius > 0:
                # use the bg model to calculate bg_color
                polar = raymarching.polar_from_ray(rays_o, rays_ddd, self.bg_radius)  # [N, 2] in [-1, 1]
                bg_color = model2.background(polar, rays_ddd.reshape(-1, 3))  # [N, 3]
            elif bg_color is None:
                bg_color = 1

            corrective2 = corrective2 + (1 - weights_sum).unsqueeze(-1) * bg_color

            corrective = torch.ones((lenn, 3)).cuda()

            corrective[msk_depth.reshape(-1, 1)[:, 0], :] = corrective2
            image2 = corrective.view(*prefix, 3)
            depth = torch.zeros((lenn,)).cuda()
            depth[msk_depth.reshape(-1, 1)[:, 0]] = depth2
            depth = depth.view(*prefix)

            # tmp: reg loss in mip-nerf 360
        else:
            corrective = torch.ones((lenn, 3)).cuda()

            image2 = corrective.view(*prefix, 3)
            depth = torch.zeros((lenn,)).cuda()
            depth = depth.view(*prefix)

        return {
            "depth": depth,
            "image": image2,
        }

    def run_cond(
        self,
        rays_o,
        rays_ddd,
        model2,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        tryi=100,
        data=None,
        iddd=None,
        trisc=None,
        trisd=None,
        nears=None,
        fars=None,
        msk_depth=None,
        face_msk=None,
        epoch=10,
        num_steps=128,
        upsample_steps=128,
        bg_color=None,
        perturb=False,
        **kwargs
    ):
        indx = data["index"]
        length = data["length"]
        fr = data["json_file"]
        threshold = data["threshold"]
        if indx < length:
            iddd1 = 0
        else:
            iddd1 = indx - length
        epochh = epoch  # iddd1
        lenn = rays_o.shape[1]
        prefix = rays_o.shape[:-1]
        rays_o = (rays_o.contiguous().view(-1, 3))[msk_depth.reshape(-1, 1)[:, 0]]
        rays_ddd = (rays_ddd.contiguous().view(-1, 3))[msk_depth.reshape(-1, 1)[:, 0]]

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears = nears[msk_depth.reshape(-1, 1)[:, 0]]
        fars = fars[msk_depth.reshape(-1, 1)[:, 0]]

        z_val = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_val = z_val.expand((N, num_steps))  # [N, T]
        z_val = nears + (fars - nears) * z_val  # [N, T], in [nears, fars]

        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_val = z_val + (torch.rand(z_val.shape, device=device) - 0.5) * sample_dist
        # generate xyzs
        xyzss = rays_o.unsqueeze(-2) + rays_ddd.unsqueeze(-2) * z_val.unsqueeze(
            -1
        )  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzss = torch.min(torch.max(xyzss, aabb[:3]), aabb[3:])  # a manual clip.

        indxs = random.sample(range(0, 700), 1)

        scale = 1.0
        xyzs, closest_faces, dist = self.warp_position(
            trisd.unsqueeze(0), trisc.unsqueeze(0), xyzss.reshape(-1, 3), tryi
        )
        msk_dist = abs(dist) >= 0.04
        z_vals = z_val

        rays_d = self.warp_direction(
            trisd.unsqueeze(0),
            trisc.unsqueeze(0),
            closest_faces,
            rays_ddd.view(-1, 1, 3).expand_as(xyzss).reshape(-1, 3),
        )

        de = torch.cat((data["expressions"], data["jaw_pose"]), dim=2)
        cond = self.conditioning2(
            closest_faces, de[0, :, :, 0, 0], local_coordinates_meshes, trisd.shape[0], ones_only=False
        )

        density_outputs_c, deform, gt, color_code = self.density(
            xyzs.reshape(-1, 3), iddd1, epochh, cond.unsqueeze(0), fr, threshold, None
        )
        for k, v in density_outputs_c.items():
            density_outputs_c[k] = v.view(N, num_steps, -1)

        #     for k in density_outputs:
        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        ssgma = (density_outputs_c["sigma"]).squeeze(-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * ssgma)  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        Tr = torch.cumprod(alphas_shifted, dim=-1)[..., :-1]
        weights = alphas * Tr  # [N, T+t]
        dirs = rays_d
        for k, v in density_outputs_c.items():
            density_outputs_c[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        rgbs0 = self.color(
            xyzs.reshape(-1, 3).detach(),
            dirs.reshape(-1, 3),
            mask=mask.reshape(-1),
            color_code=color_code,
            **density_outputs_c
        )
        rgbs = rgbs0.view(N, -1, 3)  # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth
        depth2 = torch.sum(weights * z_vals, dim=-1)

        # """ # calculate color
        corrective2 = torch.sum(
            weights.unsqueeze(-1) * rgbs, dim=-2
        )  # [N, 3], in [0, 1] #torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2) # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            polar = raymarching.polar_from_ray(rays_o, rays_ddd, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_ddd.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        corrective2 = corrective2 + (1 - weights_sum).unsqueeze(-1) * bg_color

        corrective = torch.ones((lenn, 3)).cuda()

        corrective[msk_depth.reshape(-1, 1)[:, 0], :] = corrective2
        image2 = corrective.view(*prefix, 3)

        # tmp: reg loss in mip-nerf 360
        msk = z_val > (nears + 0.20 + 0.02)
        return {
            "image": image2,
            "depth2": depth2,
            "depth3": weights_sum,
            "deform": deform,
            "GT": gt,
        }

    def run_real(self, rays_o, rays_d, num_steps=128, upsample_steps=128, bg_color=None, perturb=False, **kwargs):

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]

                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T]

                # sample new z_vals
                z_vals_mid = z_vals[..., :-1] + 0.5 * deltas[..., :-1]  # [N, T-1]
                new_z_vals = sample_pdf(
                    z_vals_mid, weights[:, 1:-1], upsample_steps, det=not self.training
                ).detach()  # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
                    -1
                )  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])  # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * self.density_scale * density_outputs["sigma"].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        mask = weights > 1e-4  # hard coded
        rgbs = self.color(xyzs.reshape(-1, 3), dirs.reshape(-1, 3), mask=mask.reshape(-1), **density_outputs)
        rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]

        # calculate depth
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            sph = raymarching.sph_from_ray(rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(sph, rays_d.reshape(-1, 3))  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        # tmp: reg loss in mip-nerf 360

        return {
            "depth": depth,
            "image": image,
            "weights_sum": weights_sum,
        }

    def run_cuda(
        self,
        rays_o,
        rays_d,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        dt_gamma=0,
        bg_color=None,
        perturb=False,
        force_all_rays=False,
        max_steps=1024,
        **kwargs
    ):

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(
            rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near
        )

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            polar = raymarching.polar_from_ray(rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_d)  # [N, 3]
        elif bg_color is None:
            bg_color = 1

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(
                rays_o,
                rays_d,
                self.bound,
                self.density_bitfield,
                self.cascade,
                self.grid_size,
                nears,
                fars,
                counter,
                self.mean_count,
                perturb,
                128,
                force_all_rays,
                dt_gamma,
                max_steps,
            )

            scale = 1.0
            xyzs, closest_faces = self.warp_position(trisd.unsqueeze(0), trisc.unsqueeze(0), xyzs.reshape(-1, 3), tryi)

            de = torch.cat((data["expressions"], data["jaw_pose"]), dim=2)
            cond = None

            sigmas, rgbs = self(xyzs, dirs * 0.0 + (1 / np.sqrt(3)), xyzs.reshape(-1, 3), cond, iddd1, de)
            sigmas = self.density_scale * sigmas

            # special case for CCNeRF's residual learning
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))

                depth = torch.stack(depths, axis=0)  # [K, B, N]
                image = torch.stack(images, axis=0)  # [K, B, N, 3]

            else:

                weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)

        else:

            # allocate outputs
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            # output should always be float32! only network inference uses half.
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            weights_sum1 = weights_sum
            depth1 = depth
            image1 = image

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0
            if self.training == True or ver == None:
                animate = False
            else:
                animate = True
            if animate:
                scale = 1.0
                N_samples = prefix[1]

            while step < max_steps:

                # count alive rays
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)
                xyzs, dirs, deltas = raymarching.march_rays(
                    n_alive,
                    n_step,
                    rays_alive,
                    rays_t,
                    rays_o,
                    rays_d,
                    self.bound,
                    self.density_bitfield,
                    self.cascade,
                    self.grid_size,
                    nears,
                    fars,
                    128,
                    perturb,
                    dt_gamma,
                    max_steps,
                )

                if animate:

                    sm = Network(converter, face_id, ver)
                    rays_d1 = dirs
                    inputt_111_2, id_x_111_2, local_coord, tri, tri_def, vert_uv, face_t, triangles_meshes_111_2 = (
                        sm.calculate_density_color(xyzs.unsqueeze(0), None, None)
                    )

                    ###########################################################################################################################################################################

                    v0, _, _ = triangles_meshes_111_2.unbind(-2)
                    v0 = (v0[id_x_111_2])[0].unsqueeze(-1).cuda() / scale
                    v00, _, _ = triangles_meshes.unbind(-2)
                    v00 = (v00[id_x_111_2])[0].unsqueeze(-1).cuda() / scale

                    local_coord = local_coord.permute((0, 2, 1))
                    local_coordinates = (local_coordinates_meshes[id_x_111_2])[0].permute((0, 2, 1))

                    oncc = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(int(local_coordinates.shape[0]), 1, 1).cuda()
                    oo_111 = torch.tensor([[1.0]]).repeat(int(local_coordinates.shape[0]), 1).cuda()
                    can_tra = torch.cat((torch.cat((local_coordinates, v00), 2), oncc), 1)
                    def_tra = torch.cat((torch.cat((local_coord, v0), 2), oncc), 1)

                    pts_111 = torch.cat((xyzs.reshape(-1, 3), oo_111), 1)

                    mmt = torch.bmm((can_tra), torch.inverse(def_tra))

                    transform_1 = torch.bmm(mmt, pts_111.unsqueeze(1).permute((0, 2, 1)))

                    transform_2 = transform_1
                    xyzs = transform_2[:, 0:3, 0].float()

                    raysd1_111 = torch.cat((rays_d1, oo_111), 1)
                    transform_11 = torch.bmm(mmt[:, 0:3, 0:3], raysd1_111.unsqueeze(1).permute((0, 2, 1))[:, 0:3, 0:3])
                    transform_22 = transform_11
                    dirs = transform_22[:, 0:3, 0].float()

                sigmas, rgbs = self(xyzs, dirs * 0.0 + (1 / np.sqrt(3)))
                debug_input_data = False
                if debug_input_data == True:
                    print(xyzs.shape)
                    inp_pose = xyzs.reshape(-1, 3)
                    pp2 = inp_pose.clone().cpu().float().detach().numpy()
                    print(pp2.shape)
                    np.savetxt(
                        "sampled_points_111_Run_Cuda_Marching_Cubes_.off",
                        pp2,
                        header="OFF\n" + str(pp2.shape[0]) + " 0 0",
                        comments="",
                    )
                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(
                    n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image
                )

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            depth = torch.clamp(depth - nears, min=0) / (fars - nears)
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)

        return {
            "depth": depth,
            "image": image,
        }

    def run_cuda2(
        self,
        rays_o,
        rays_d,
        model2,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        tryi=100,
        data=None,
        iddd=None,
        trisc=None,
        trisd=None,
        nears=None,
        fars=None,
        msk_depth=None,
        face_msk=None,
        epoch=10,
        dt_gamma=0,
        bg_color=None,
        perturb=False,
        force_all_rays=False,
        max_steps=1024,
        **kwargs
    ):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(
            rays_o, rays_d, self.aabb_train if self.training else self.aabb_infer, self.min_near
        )

        # mix background color
        if self.bg_radius > 0:
            # use the bg model to calculate bg_color
            polar = raymarching.polar_from_ray(rays_o, rays_d, self.bg_radius)  # [N, 2] in [-1, 1]
            bg_color = self.background(polar, rays_d, None, None)  # [N, 3]
        elif bg_color is None:
            bg_color = 1
        scale = 1.0
        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(
                rays_o,
                rays_d,
                self.bound,
                self.density_bitfield,
                self.cascade,
                self.grid_size,
                nears,
                fars,
                counter,
                self.mean_count,
                perturb,
                128,
                force_all_rays,
                dt_gamma,
                max_steps,
            )

            xyzs, closest_faces = self.warp_position(trisd.unsqueeze(0), trisc.unsqueeze(0), xyzs, tryi)

            de = torch.cat((data["expressions"], data["jaw_pose"]), dim=2)
            cond = None

            sigmas, rgbs = self(xyzs.detach(), dirs * 0.0 + (1 / np.sqrt(3)), cond, None, de)
            sigmas = self.density_scale * sigmas

            # special case for CCNeRF's residual learning
            if len(sigmas.shape) == 2:
                K = sigmas.shape[0]
                depths = []
                images = []
                for k in range(K):
                    weights_sum, depth, image = raymarching.composite_rays_train(sigmas[k], rgbs[k], deltas, rays)
                    image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                    images.append(image.view(*prefix, 3))
                    depths.append(depth.view(*prefix))

                depth = torch.stack(depths, axis=0)  # [K, B, N]
                image = torch.stack(images, axis=0)  # [K, B, N, 3]

            else:

                weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs, deltas, rays)
                image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
                image = image.view(*prefix, 3)
                depth = depth.view(*prefix)

        else:

            # allocate outputs
            # if use autocast, must init as half so it won't be autocasted and lose reference.
            # output should always be float32! only network inference uses half.
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0
            scale = 1.0

            while step < max_steps:

                # count alive rays
                n_alive = rays_alive.shape[0]

                # exit loop
                if n_alive <= 0:
                    break

                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)
                xyzs, dirs, deltas = raymarching.march_rays(
                    n_alive,
                    n_step,
                    rays_alive,
                    rays_t,
                    rays_o,
                    rays_d,
                    self.bound,
                    self.density_bitfield,
                    self.cascade,
                    self.grid_size,
                    nears,
                    fars,
                    128,
                    perturb,
                    dt_gamma,
                    max_steps,
                )
                xyzs, closest_faces = self.warp_position(trisd.unsqueeze(0), trisc.unsqueeze(0), xyzs, tryi)

                sigmas, rgbs = self(xyzs, dirs * 0.0 + (1 / np.sqrt(3)), cond, None, de)
                debug_input_data = False
                if debug_input_data == True:
                    print(xyzs.shape)
                    inp_pose = xyzs.reshape(-1, 3)
                    pp2 = inp_pose.clone().cpu().float().detach().numpy()
                    print(pp2.shape)
                    np.savetxt(
                        "sampled_points_111_Run_Cuda_Marching_Cubes_.off",
                        pp2,
                        header="OFF\n" + str(pp2.shape[0]) + " 0 0",
                        comments="",
                    )
                sigmas = self.density_scale * sigmas

                raymarching.composite_rays(
                    n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image
                )

                rays_alive = rays_alive[rays_alive >= 0]

                step += n_step

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
            image = image.view(*prefix, 3)
            depth = depth.view(*prefix)

        return {
            "depth": depth,
            "image": image,
            "depth3": weights_sum,
        }

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):

        if not self.cuda_ray:
            return

        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]

        fx, fy, cx, cy = intrinsic

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1
                    )  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0)  # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2**cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3]  # [S, N, 3]

                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0  # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1)  # [N]

                            # update count
                            count[cas, indices] += mask
                            head += S

        # mark untrained grid as -1
        self.density_grid[count == 0] = -1

    @torch.no_grad()
    def update_extra_state(self, trisc=None, trisd=None, data=None, tryi=None, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return

        de = torch.cat((data["expressions"], data["jaw_pose"]), dim=2)

        ### update density grid

        tmp_grid = -torch.ones_like(self.density_grid)

        # full update.
        if self.iter_density < 16:
            # if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:

                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat(
                            [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1
                        )  # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long()  # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2**cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size

                            cas_xyzs, closest_faces = self.warp_position(
                                trisd.unsqueeze(0), trisc.unsqueeze(0), cas_xyzs, None
                            )

                            sigmas = self.density(cas_xyzs, None, None, de)["sigma"].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign
                            tmp_grid[cas, indices] = sigmas

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size**3 // 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(
                    0, self.grid_size, (N, 3), device=self.density_bitfield.device
                )  # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long()  # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(self.density_grid[cas] > 0).squeeze(-1)  # [Nz]
                rand_mask = torch.randint(
                    0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device
                )
                occ_indices = occ_indices[rand_mask]  # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices)  # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]
                bound = min(2**cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                cas_xyzs, closest_faces = self.warp_position(trisd.unsqueeze(0), trisc.unsqueeze(0), cas_xyzs, None)

                # query density
                sigmas = self.density(cas_xyzs, None, None, de)["sigma"].reshape(-1).detach()
                sigmas *= self.density_scale
                # assign
                tmp_grid[cas, indices] = sigmas

        ## max-pool on tmp_grid for less aggressive culling [No significant improvement...]

        # ema update
        valid_mask = (self.density_grid >= 0) & (tmp_grid >= 0)
        self.density_grid[valid_mask] = torch.max(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(
            self.density_grid.clamp(min=0)
        ).item()  # -1 non-training regions are viewed as 0 density.
        self.iter_density += 1

        # convert to bitfield
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

    def render(
        self,
        rays_o,
        rays_d,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        adj=None,
        iddd=None,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):

        if self.cuda_ray:
            _run = self.run_cuda
        else:
            _run = self.run_real

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:

            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            aaa = 0
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        converter,
                        ver,
                        face_id,
                        local_coordinates_meshes,
                        triangles_meshes,
                        z_val.unsqueeze(0)[b : b + 1, head:tail],
                        adj,
                        iddd,
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_["depth"]
                    image[b : b + 1, head:tail] = results_["image"]
                    head += max_ray_batch
            results = {}
            results["depth"] = depth
            results["image"] = image

        else:
            results = _run(
                rays_o, rays_d, **kwargs
            )  # , converter, ver, face_id, local_coordinates_meshes, triangles_meshes, **kwargs)

        return results

    def render_cond_test(
        self,
        rays_o,
        rays_d,
        model2,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        adj=None,
        iddd=None,
        abcdefghhhh=None,
        trisc=None,
        trisd=None,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):

        if self.cuda_ray:
            _run = self.run_cuda2
        else:
            _run = self.run_cond_test

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:

            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            aaa = 0
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        model2,
                        converter,
                        ver,
                        face_id,
                        local_coordinates_meshes,
                        triangles_meshes,
                        None,
                        adj,
                        iddd,
                        abcdefghhhh,
                        trisc,
                        trisd,
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_["depth"]
                    image[b : b + 1, head:tail] = results_["image"]
                    head += max_ray_batch
            results = {}
            results["depth"] = depth
            results["image"] = image

        else:
            results = _run(
                rays_o,
                rays_d,
                model2,
                converter,
                ver,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_val,
                adj,
                None,
                iddd,
                trisc,
                trisd,
                **kwargs
            )

        return results

    def render_cond(
        self,
        rays_o,
        rays_d,
        model2,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        adj=None,
        data=None,
        iddd=None,
        trisc=None,
        trisd=None,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):

        if self.cuda_ray:
            _run = self.run_cuda2
        else:
            _run = self.run_cond

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:

            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            aaa = 0
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        model2,
                        converter,
                        ver,
                        face_id,
                        local_coordinates_meshes,
                        triangles_meshes,
                        z_val,
                        adj,
                        data,
                        iddd,
                        trisc,
                        trisd,
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_["depth"]
                    image[b : b + 1, head:tail] = results_["image"]
                    head += max_ray_batch
            results = {}
            results["depth"] = depth
            results["image"] = image

        else:
            results = _run(
                rays_o,
                rays_d,
                model2,
                converter,
                ver,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_val,
                adj,
                data,
                iddd,
                trisc,
                trisd,
                **kwargs
            )

        return results

    def render_cond_sample_depth(
        self,
        rays_o,
        rays_d,
        model2,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        adj=None,
        data=None,
        iddd=None,
        trisc=None,
        trisd=None,
        nears=None,
        fars=None,
        msk_depth=None,
        face_msk=None,
        epoch=10,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):

        if self.cuda_ray:
            _run = self.run_cuda2
        else:
            _run = self.run_cond

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:

            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            aaa = 0
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        model2,
                        converter,
                        ver,
                        face_id,
                        local_coordinates_meshes,
                        triangles_meshes,
                        z_val,
                        adj,
                        data,
                        iddd,
                        trisc,
                        trisd,
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_["depth"]
                    image[b : b + 1, head:tail] = results_["image"]
                    head += max_ray_batch
            results = {}
            results["depth"] = depth
            results["image"] = image

        else:
            results = _run(
                rays_o,
                rays_d,
                model2,
                converter,
                ver,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_val,
                adj,
                data,
                iddd,
                trisc,
                trisd,
                nears,
                fars,
                msk_depth,
                face_msk,
                epoch,
                **kwargs
            )

        return results

    def render_cond_test_depth_sample(
        self,
        rays_o,
        rays_d,
        model2,
        converter=None,
        ver=None,
        face_id=None,
        local_coordinates_meshes=None,
        triangles_meshes=None,
        z_val=None,
        adj=None,
        iddd=None,
        abcdefghhhh=None,
        trisc=None,
        trisd=None,
        nears=None,
        fars=None,
        msk_depth=None,
        face_msk=None,
        epoch=10,
        staged=False,
        max_ray_batch=4096,
        **kwargs
    ):

        if self.cuda_ray:
            _run = self.run_cuda2
        else:
            _run = self.run_cond_test

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # never stage when cuda_ray
        if staged and not self.cuda_ray:

            depth = torch.empty((B, N), device=device)
            image = torch.empty((B, N, 3), device=device)
            aaa = 0
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(
                        rays_o[b : b + 1, head:tail],
                        rays_d[b : b + 1, head:tail],
                        model2,
                        converter,
                        ver,
                        face_id,
                        local_coordinates_meshes,
                        triangles_meshes,
                        None,
                        adj,
                        iddd,
                        abcdefghhhh,
                        trisc,
                        trisd,
                        nears[head:tail].cuda(),
                        fars[head:tail].cuda(),
                        msk_depth[head:tail].cuda(),
                        None,
                        epoch,
                        **kwargs
                    )
                    depth[b : b + 1, head:tail] = results_["depth"]
                    image[b : b + 1, head:tail] = results_["image"]
                    head += max_ray_batch
            results = {}
            results["depth"] = depth
            results["image"] = image

        else:
            results = _run(
                rays_o,
                rays_d,
                model2,
                converter,
                ver,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_val,
                adj,
                None,
                iddd,
                trisc,
                trisd,
                **kwargs
            )
        return results
