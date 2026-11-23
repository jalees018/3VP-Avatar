# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import numpy as np
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (paths.py)
from paths import TRACKER_DIR

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pickle as pkl
import trimesh

import json
import pytorch3d

import pymeshlab
import pickle
import pytorch3d.transforms

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

BODY_POSE_111_2 = [
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
]  # [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
OP2SMPL_X = [
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
]
pose_111 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
pose_1111 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50


def example_save_mesh(base_path, output_path, subject):
    # lines needed to run this specific example

    # create a new MeshSet
    ms = pymeshlab.MeshSet()

    # load a new mesh
    ms.load_new_mesh(base_path)

    # save the current mesh with default parameters
    ms.save_current_mesh(output_path + subject + ".off")

    # save the current mesh without face color


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmark Demo")

    parser.add_argument("--base_file", type=str, default="", help="The Base model directory")
    parser.add_argument("--keypoints_file", type=str, default="", help="The Keypoints directory")
    parser.add_argument("--output", type=str, default="", help="Name of the Output Path")
    parser.add_argument("--Mesh_Path", type=str, default="", help="Name of the Mesh Path")
    parser.add_argument("--subject", type=str, default="", help="Name of the Mesh Path")
    parser.add_argument("--export_mesh", action="store_true", help="Whether to export the mesh")
    parser.add_argument("--num_imgs", type=int, default=3000, help="Number of Tracking image Frames")
    parser.add_argument("--exp_track", type=str, default="Results_", help="Expression Tracking name")

    args = parser.parse_args()

    base_file = args.base_file
    keypoints_file = args.keypoints_file
    output = args.output
    mesh_path = args.Mesh_Path
    subject = args.subject
    export_mesh = args.export_mesh
    num_imgs = args.num_imgs
    exp_track = args.exp_track

    with open(os.path.join(base_file, "transforms_train.json"), "r") as f:
        transform = json.load(f)

    with open(os.path.join(base_file, "trajectory.json"), "r") as f:
        pose = json.load(f)  # [-1]
    camera_pose111 = []
    for iiikjk in pose:
        camera_pose222 = np.concatenate(
            (iiikjk["transform_matrix"], np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)), axis=0
        )
        M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        xf = np.linalg.inv(M) @ camera_pose222

        camera_pose111.append(xf)
    # load intrinsics
    downscale = 1.0
    H = 512
    W = 512

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
    fov_y = transform["camera_angle_y"]

    eul_angles = pytorch3d.transforms.matrix_to_euler_angles(
        torch.inverse(torch.tensor(camera_pose111))[:, 0:3, 0:3], "ZXY"
    )
    binssss1 = np.linspace(-np.pi, np.pi, 24)
    globb = []
    if export_mesh == False:
        for iterrr in range(0, num_imgs):
            path1 = TRACKER_DIR + exp_track + subject + "/"
            abcdefghhhh = pkl.load(open(path1 + "result_final_Real_tt_" + str(iterrr) + ".pkl", "rb"))
            jpp = abcdefghhhh["jaw_pose"]
            exprs = abcdefghhhh["expressions"]
            gbb = abcdefghhhh["global_orient"]
            globb.append(gbb)
        datttaaa = np.array(globb)[:, 1].tolist()
        hist, bin_edges = np.histogram(datttaaa, bins=binssss1)
        probabi = 1.0 - hist / hist.sum()
        bins_per_point = np.fmin(np.digitize(datttaaa, bin_edges), len(bin_edges) - 1)
        probability_perpoint = [(hist / hist.sum())[bins_per_point[i] - 1] for i in range(len(datttaaa))]
        aa = 1 - np.array(probability_perpoint)
        np.save(mesh_path + "/Probs222.npy", aa)
    if export_mesh == True:
        print("Rendering Depth Maps")
        device = torch.device("cuda:0")
        mesh0 = trimesh.load(mesh_path + "meshes/" + subject + "_111.obj", process=False)
        print(mesh0.vertices.shape)

        offs1 = np.expand_dims(np.array(transform["Trans"]), axis=0)
        offs = offs1.copy()
        offs[:, 0] = offs1[:, 1]
        offs[:, 1] = offs1[:, 2]
        offs[:, 2] = offs1[:, 0]
        vv = (np.array(mesh0.vertices[:, 0:3]) - offs) * transform["Scalee"]
        vertex_colors = False
        if vertex_colors == False:

            color_visuals = trimesh.visual.TextureVisuals(uv=mesh0.visual.uv, material=mesh0.visual.material)
            Mesh_Final = trimesh.Trimesh(
                vertices=vv, faces=mesh0.faces, face_normals=mesh0.face_normals, visual=color_visuals
            )
        else:
            Mesh_Final = trimesh.Trimesh(vertices=vv, faces=mesh0.faces, vertex_colors=mesh0.visual.vertex_colors)
        Mesh_Final.export(mesh_path + "/meshes/" + subject + ".obj")
        example_save_mesh(mesh_path + "/meshes/" + subject + ".obj", mesh_path + "/meshes/", subject)
