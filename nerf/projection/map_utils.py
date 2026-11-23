import torch

import bvh_distance_queries
import pytorch3d
from pytorch3d.structures import Meshes, Pointclouds
import pytorch3d.io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(mesh_path, device=device):
    verts, faces, aux = pytorch3d.io.load_obj(mesh_path, device=device)
    faces_idx = faces.verts_idx.to(device)
    faces_t = faces.textures_idx.to(device)
    if aux[1] is not None:
        verts_uvs = aux[1].to(device)
    else:
        verts_uvs = None
    return verts, faces, faces_idx, aux[1], faces_t, device


def point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()
    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]

    m = bvh_distance_queries.BVH()
    distances, closest_points, closest_faces, closest_bcs = m(tris.unsqueeze(0), points.unsqueeze(0))
    #         points, points_first_idx, tris, tris_first_idx, max_points
    return distances[0], closest_faces[0]  # point_to_face, idxs
