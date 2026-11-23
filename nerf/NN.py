import torch
import torch.nn as nn

import pytorch3d
import pytorch3d.io


class Network(nn.Module):
    def __init__(self, converter, face_idx, ver):
        super(Network, self).__init__()

        self.face_idx = face_idx

        self.converter = converter
        self.vver = ver

    def prepare_input(self, wpts, sp_input):

        frame_index = 0
        # if self.training:
        # """ if cfg.optimize_smpl and (frame_index >= self.train_frame_start) and (frame_index < self.train_frame_end):
        verts = torch.tensor(self.vver).unsqueeze(0).float().cuda()
        # transform sample points to surface-aligned representation (inp)
        inp, id_x_111_2, local_coordinates, tri, v, t, triangles_meshes = self.converter.xyz_to_xyzch(
            wpts, verts, self.face_idx
        )
        tri_def = None

        return inp, id_x_111_2, local_coordinates, tri, tri_def, v, t, triangles_meshes

    def calculate_density_color(self, wpts, viewdir, sp_input):

        inp, id_x_111_2, local_coordinates, tri, tri_def, v, t, triangles_meshes = self.prepare_input(wpts, sp_input)

        return inp, id_x_111_2, local_coordinates, tri, tri_def, v, t, triangles_meshes


def load_model(mesh_path):
    verts, faces, aux = pytorch3d.io.load_obj(mesh_path)
    faces_idx = faces.verts_idx.cuda()
    return verts, faces, faces_idx  # , verts_uvs, faces_t, device


def compute_tangent_bitangent_111(triangles_meshes, faces_normals_meshes) -> torch.Tensor:
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
    return torch.stack((tangents, normals, bitangents), dim=1)  # tangents, bitangents
