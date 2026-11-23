import torch

from pytorch3d.structures import Meshes, Pointclouds

from .map_utils import point_mesh_face_distance


class SurfaceAlignedConverter:

    def __init__(self, verts1, faces, faces_idx, verts_uvs, faces_t, device):
        self.device = device
        self.verts = verts1
        self.faces_idx = faces_idx
        self.verts_uvs = None  # verts_uvs
        self.faces_t = None  # faces_t
        self.xyzc = verts1

    def xyz_to_xyzch(self, points, verts, f_idx):
        h, barycentric, idx, local_coordinates, _, _, triangles_meshes = self.projection(points, verts, f_idx)
        xyzc_triangles = None

        return h, idx, local_coordinates, xyzc_triangles, self.verts_uvs, self.faces_t, triangles_meshes

    def projection(self, points, verts, f_idx, points_inside_mesh_approx=True, scaling_factor=1.0):  # 50

        # STEP 0: preparation
        faces = (self.faces_idx[f_idx, :])[None, ...].repeat(len(verts), 1, 1)
        meshes = Meshes(verts=verts * scaling_factor, faces=faces)
        pcls = Pointclouds(points=points * scaling_factor)
        # compute nearest faces
        dist, idx = point_mesh_face_distance(meshes, pcls)
        triangles_meshes = (verts[0])[meshes.faces_packed()]

        # if points_inside_mesh_approx:

        h = dist  # * sign_tensor

        # revert scaling
        barycentric = None
        h = h / scaling_factor
        nearest_new = None  # nearest_new / scaling_factor
        nearest = None  # nearest / scaling_factor

        # local_coordinates
        local_coordinates_meshes = self._calculate_local_coordinates_meshes(
            meshes.faces_normals_packed(), triangles_meshes
        )
        local_coordinates = local_coordinates_meshes

        idx = idx.view(len(verts), -1)

        return h, barycentric, idx, local_coordinates, nearest_new, nearest, triangles_meshes

    def _calculate_local_coordinates_meshes(self, faces_normals_meshes, triangles_meshes):
        tangents, bitangents, normals = self.compute_tangent_bitangent_111(triangles_meshes, faces_normals_meshes)
        local_coordinates_meshes = torch.stack(
            (tangents, normals, bitangents), dim=1
        )  # [N, 3, 3] #local_coordinates_meshes = torch.stack((faces_normals_meshes, tangents, bitangents), dim=1)  # [N, 3, 3]
        return local_coordinates_meshes

    def compute_tangent_bitangent_111(self, triangles_meshes, faces_normals_meshes) -> torch.Tensor:
        # Compute tangets and bitangents following:
        # https://learnopengl.com/Advanced-Lighting/Normal-Mapping
        face_xyz = triangles_meshes
        assert face_xyz.shape[-2:] == (3, 3)
        v0, v1, v2 = face_xyz.unbind(-2)
        dv10 = v1 - v0
        dv20 = v2 - v0
        tangents = v1 - v0
        normals = torch.cross(v2 - v0, v1 - v0)
        bitangents = torch.cross(normals, tangents)
        return tangents, bitangents, normals
