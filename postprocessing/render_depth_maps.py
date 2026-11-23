import os
import tqdm

import numpy as np
import torch

import trimesh
from pytorch3d.structures import Meshes  # , Pointclouds
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (paths.py)
from paths import DATA_DIR, IMAGES_DIR, TRACKER_DIR, WORK_DIR

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sampling Maps")

    parser.add_argument("--subject", type=str, default="", help="Name of the subject")
    parser.add_argument("--workspace", type=str, default="", help="Name of the Workspace")
    parser.add_argument("--exp_track", type=str, default="Results_", help="Expression Tracking name")
    parser.add_argument("--train_imgs", type=int, default=3000, help="Number of Tracking image Frames")

    opt = parser.parse_args()

    subject = opt.subject
    workspace = opt.workspace
    exp_track = opt.exp_track
    train_imgs = opt.train_imgs

    M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    W = 512  # 1000
    H = 512  # 800

    path1 = TRACKER_DIR + opt.exp_track + opt.subject + "/"
    obj_path2 = WORK_DIR + opt.workspace + "/meshes/" + opt.subject + ".obj"
    root_path = DATA_DIR + opt.subject
    path3 = IMAGES_DIR + opt.subject + "/Images/Depth_Maps"
    if not os.path.exists(path3):
        os.makedirs(path3)
    lstt_111 = []
    with open(os.path.join(root_path, f"transforms_train.json"), "r") as f:
        transform = json.load(f)
    frames = transform["frames"]
    # load intrinsics
    focal_scale = H / int(transform["h"])
    if "fl_x" in transform or "fl_y" in transform:
        fl_x = (transform["fl_x"] if "fl_x" in transform else transform["fl_y"]) * focal_scale  # / downscale
        fl_y = (transform["fl_y"] if "fl_y" in transform else transform["fl_x"]) * focal_scale  # / downscale
    elif "camera_angle_x" in transform or "camera_angle_y" in transform:
        # blender, assert in radians. already downscaled since we use H/W
        fl_x = W / (2 * np.tan(transform["camera_angle_x"] / 2)) if "camera_angle_x" in transform else None
        fl_y = H / (2 * np.tan(transform["camera_angle_y"] / 2)) if "camera_angle_y" in transform else None
        if fl_x is None:
            fl_x = fl_y
        if fl_y is None:
            fl_y = fl_x
    else:
        raise RuntimeError("Failed to load focal length, please check the transforms.json!")

    cx = (transform["cx"] * focal_scale) if "cx" in transform else (W / 2)
    cy = (transform["cy"] * focal_scale) if "cy" in transform else (H / 2)

    intrinsics = np.array([fl_x, fl_y, cx, cy])
    intr = intrinsics
    with open(os.path.join(root_path, f"trajectory.json"), "r") as f:
        meta111 = json.load(f)
    for f in tqdm.tqdm(frames, desc=f"Loading data"):
        f_path = os.path.join(root_path, f["file_path"])
        lstt_111.append(int(f_path[-8:-4]))
    lstt_222 = list(lstt_111)
    lstt_222.sort()
    id_111 = np.where(np.array(lstt_111) == np.array(lstt_111).max())
    j = lstt_111[int(id_111[0])]

    arr = np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    for iit in tqdm.tqdm(range(0, len(lstt_111) + opt.train_imgs), desc=f"Loading Depth Maps"):
        if iit < len(lstt_111):
            mesh_111 = trimesh.load(obj_path2, process=False)
            trimesh.repair.fix_inversion(mesh_111)
            mesh_111.merge_vertices(merge_tex=True, merge_norm=True)
            mesh_1111 = trimesh.Trimesh(
                vertices=mesh_111.vertices + 0.02 * mesh_111.vertex_normals, faces=mesh_111.faces
            )
            i = lstt_111[iit]
            frame = meta111[lstt_222.index(i)]
            camera_pose111 = np.array(frame["transform_matrix"], dtype=np.float32)

        else:
            mesh_111 = trimesh.load(path1 + "Results_Deformed_tt_" + str(iit - len(lstt_111)) + ".obj", process=False)
            trimesh.repair.fix_inversion(mesh_111)
            mesh_111.merge_vertices(merge_tex=True, merge_norm=True)
            mesh_1111 = trimesh.Trimesh(
                vertices=mesh_111.vertices + 0.02 * mesh_111.vertex_normals, faces=mesh_111.faces
            )
            frame = meta111[lstt_222.index(j)]
            camera_pose111 = np.array(frame["transform_matrix"], dtype=np.float32)

        camera = pyrender.camera.IntrinsicsCamera(intr[0], intr[1], intr[2], intr[3])

        camera_pose111 = np.concatenate((camera_pose111, arr), axis=0)
        xf = np.linalg.inv(M) @ camera_pose111

        ###if indx > 838:

        scene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]), bg_color=np.array([0.0, 0.0, 0.0]))
        scene.add(pyrender.Mesh.from_trimesh(mesh_1111, smooth=False))
        nc = pyrender.Node(camera=camera, matrix=xf)
        scene.add_node(nc)
        r = pyrender.OffscreenRenderer(W, H)
        col, depth = r.render(scene, flags=flags)
        if iit == 0:
            col2 = col.copy()
            col2[col2 <= 128] = 0
        depth = torch.tensor(depth)
        msk_depth = depth != 0.0

        r.delete()
        nears = depth.clone()
        fars = depth.clone()
        nears[msk_depth] -= 0.07
        fars[msk_depth] += 0.80

        np.savez(path3 + "/" + "Depth_" + str(iit) + ".npz", near=nears, far=fars, msk_depthh=msk_depth)
