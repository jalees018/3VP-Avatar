"""Nearest-triangle correspondences between the canonical (scanned) mesh and the registered SMPL-X mesh."""

import numpy as np
import torch
import trimesh
from projection.map import SurfaceAlignedConverter


def main(canonical_mesh_path, smplx_meshes_path, subject):
    """Saves, for every vertex of the canonical mesh, the index of the closest SMPL-X triangle
    to ``<smplx_meshes_path>/reg_results/<subject>_NN.npy``."""
    mesh = trimesh.load(canonical_mesh_path, indd=3, process=False)
    smplx_mesh = smplx_meshes_path + "reg_results/" + subject + ".obj"
    mesh0 = trimesh.load(smplx_mesh, process=False)
    wpts = torch.tensor((np.array(mesh.vertices))[:, 0:3]).float().cuda()
    verts = torch.tensor(np.array(mesh0.vertices)).float().cuda()
    converter = SurfaceAlignedConverter(smplx_mesh)
    _, closest_face_ids, _, _, _, _, _ = converter.xyz_to_xyzch(wpts.unsqueeze(0), verts.unsqueeze(0))
    np.save(smplx_meshes_path + "reg_results/" + subject + "_NN.npy", closest_face_ids.detach().cpu().numpy())
