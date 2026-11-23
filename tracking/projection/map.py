import torch
import torch.nn.functional as F

from pytorch3d.structures import Meshes, Pointclouds

from .map_utils import load_model, point_mesh_face_distance


class SurfaceAlignedConverter:

    def __init__(self, obj_path1):
        verts1, faces, faces_idx, verts_uvs, faces_t, device = load_model(obj_path1)
        obj_path = obj_path1
        verts, faces, faces_idx, verts_uvs, faces_t, device = load_model(obj_path)
        self.device = device
        self.verts = verts1
        self.faces_idx = faces_idx
        self.verts_uvs = verts_uvs
        self.faces_t = faces_t
        self.xyzc = verts1

    def xyz_to_xyzch(self, points, verts):
        h, barycentric, idx, local_coordinates, _, _, triangles_meshes = self.projection(points, verts)
        xyzc_triangles = None

        return h, idx, local_coordinates, xyzc_triangles, self.verts_uvs, self.faces_t, triangles_meshes

    def projection(self, points, verts, points_inside_mesh_approx=True, scaling_factor=1):  # 50

        # STEP 0: preparation
        faces = (self.faces_idx)[None, ...].repeat(len(verts), 1, 1)
        meshes = Meshes(verts=verts * scaling_factor, faces=faces)
        pcls = Pointclouds(points=points * scaling_factor)
        # compute nearest faces
        _, idx = point_mesh_face_distance(meshes, pcls)

        triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]

        # if points_inside_mesh_approx:

        # revert scaling
        barycentric = None
        h = None  # h / scaling_factor
        nearest_new = None  # nearest_new / scaling_factor
        nearest = None  # nearest / scaling_factor

        # local_coordinates
        local_coordinates_meshes = self._calculate_local_coordinates_meshes(
            meshes.faces_normals_packed(), triangles_meshes
        )
        local_coordinates = local_coordinates_meshes[idx]

        idx = idx.view(len(verts), -1)

        return h, barycentric, idx, local_coordinates, nearest_new, nearest, triangles_meshes

    def _calculate_local_coordinates_meshes(self, faces_normals_meshes, triangles_meshes):
        tangents, bitangents = self.compute_tangent_bitangent_111(triangles_meshes, faces_normals_meshes)
        local_coordinates_meshes = torch.stack(
            (tangents, bitangents, faces_normals_meshes), dim=1
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
        tangents = F.normalize(tangents, p=2, dim=-1, eps=1e-6)
        bitangents = torch.cross(faces_normals_meshes, tangents)
        return tangents, bitangents
