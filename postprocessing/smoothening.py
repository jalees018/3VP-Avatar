import numpy as np
import trimesh

import os
from tqdm import tqdm
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (paths.py)
from paths import IMAGES_DIR, TRACKER_DIR, WORK_DIR
from smplx_masks import load_mask_mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoothening Script")

    parser.add_argument("--subject", type=str, default="", help="Name of the subject")
    parser.add_argument("--exp", type=str, default="", help="Name of the experiment")
    parser.add_argument("--exp_track", type=str, default="Results_", help="Expression Tracking name")

    args = parser.parse_args()

    subject = args.subject
    exp = args.exp
    exp_track = args.exp_track

    smplx_meshes_path = WORK_DIR
    num_imgs = (
        len(
            [
                name
                for name in os.listdir(IMAGES_DIR + subject + "/Images/")
                if os.path.isfile(os.path.join(IMAGES_DIR + subject + "/Images/", name))
            ]
        )
        - 100
    )

    abcccdef = load_mask_mesh("Smooth_Mask.off")
    iddxxx = abcccdef.vertices[:, 5] == 255
    mask = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
    mask[iddxxx] = True

    obj_path1 = smplx_meshes_path + exp + "/" "reg_results/" + subject + ".obj"
    mesh0 = trimesh.load(obj_path1, process=False)
    verts_mask = mesh0.vertices
    trimesh.smoothing.filter_laplacian(mesh0, iterations=5)
    vert00 = mesh0.vertices
    vert00[~mask] = verts_mask[~mask]
    mesh00 = trimesh.Trimesh(vertices=vert00, faces=mesh0.faces)
    mesh00.export(smplx_meshes_path + exp + "/" "reg_results/" + subject + "_s.obj")

    for i in tqdm(range(0, num_imgs)):

        obj_path1 = TRACKER_DIR + exp_track + subject + "/result_final_Real_tt_" + str(i) + ".obj"
        mesh0 = trimesh.load(obj_path1, process=False)
        verts_mask = mesh0.vertices
        trimesh.smoothing.filter_laplacian(mesh0, iterations=5)
        vert00 = mesh0.vertices
        vert00[~mask] = verts_mask[~mask]
        mesh00 = trimesh.Trimesh(vertices=vert00, faces=mesh0.faces)
        mesh00.export(TRACKER_DIR + exp_track + subject + "/result_final_Real_tt_" + str(i) + "_s.obj")
