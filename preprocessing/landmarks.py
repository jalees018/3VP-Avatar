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
from paths import IMAGES_DIR

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import trimesh
import cv2

import time
import json
from pytorch3d.structures import Meshes

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


def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def backproject(depth, focalLengthx, focalLengthy, center_x, center_y, pose, scaling_factor=1.0):
    color_points = []
    a = np.zeros((depth.shape[0], depth.shape[1], 3))
    for v in range(depth.shape[1]):
        for u in range(depth.shape[0]):
            z = depth[u, v] / scaling_factor
            # if z == 0:   # here np.where can be used prior to the loop
            x = (u - center_x) * z / focalLengthx  # u and v are needed, no clue how to do this
            y = (v - center_y) * z / focalLengthy

            a[u, v, 0] = y
            a[u, v, 1] = x
            a[u, v, 2] = z

            a[u, v, :] = np.matmul(pose_111[0:3, 0:3], a[u, v, :])
            a[u, v, :] = np.matmul(pose[0:3, 0:3], a[u, v, :]) + pose[0:3, 3]

    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landmark Demo")

    parser.add_argument("--base_file", type=str, default="", help="The Base model directory")
    parser.add_argument("--keypoints_file", type=str, default="", help="The Keypoints directory")
    parser.add_argument("--output", type=str, default="", help="Name of the Output Path")
    parser.add_argument("--Mesh_Path", type=str, default="", help="Name of the Mesh Path")
    parser.add_argument("--subject", type=str, default="", help="Name of the Mesh Path")

    args = parser.parse_args()

    base_file = args.base_file
    keypoints_file = args.keypoints_file
    output = args.output
    mesh_path = args.Mesh_Path
    subject = args.subject

    with open(os.path.join(base_file, "transforms_train.json"), "r") as f:
        transform = json.load(f)

    with open(os.path.join(base_file, "trajectory.json"), "r") as f:
        pose = json.load(f)[0]
    camera_pose111 = (((transform["frames"])[0])["transform_matrix"])[0:3][0:4]
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

    Time_Begin = time.time()

    lsttt = os.listdir(IMAGES_DIR + subject + "/Images/_results/")
    offset = int(min([int(((iii).split("/")[-1])[:-4]) for iii in lsttt]))  # 1560
    key_point_loss = True
    with open(keypoints_file + "abc1_keypoints.json") as keypoint_file:
        data = json.load(keypoint_file)
    for idx2, person_data in enumerate(data["people"]):
        face_keypoints = np.array(person_data["face_keypoints_2d"], dtype=np.float32)
        face_keypoints = face_keypoints.reshape([-1, 2])
    xxxx_ = np.array(face_keypoints[0:-2, :])
    time_222 = time.time()

    camera_pose111 = np.concatenate((camera_pose111, np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)), axis=0)
    print(camera_pose111)

    # [-1,  0,  0,  0],
    # [ 0,  0,  1,  0],
    # [ 0,  0,  0,  1]]) """
    M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    xf = np.linalg.inv(M) @ camera_pose111
    print(xf)

    mesh_111 = trimesh.load(mesh_path + "meshes/" + subject + ".obj", process=False)
    trimesh.repair.fix_inversion(mesh_111)
    from pytorch3d.io import load_objs_as_meshes, load_obj

    device = torch.device("cuda:0")
    verts22, faces22, aux22 = load_obj(mesh_path + "meshes/" + subject + ".obj", device=device)
    meshes222 = Meshes(verts=[verts22], faces=[faces22.verts_idx])
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh_111, smooth=False))

    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
    camera = pyrender.camera.IntrinsicsCamera(fl_x, fl_y, cx, cy)

    light = pyrender.PointLight(color=np.ones(3), intensity=8.0)
    nc = pyrender.Node(light=light, camera=camera, matrix=xf)
    scene.add_node(nc)

    # Render the scene
    r = pyrender.OffscreenRenderer(W, H)
    col, depth = r.render(scene)
    print(depth.max())
    cv2.imwrite(mesh_path + "col.png", cv2.cvtColor(np.array(col).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(mesh_path + "depth.png", np.array(depth).astype(np.uint8))
    r.delete()
    render_pts = True
    if render_pts == True:
        # Util function for loading meshes

        # Data structures and functions for rendering
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import (
            PerspectiveCameras,
            PointLights,
            RasterizationSettings,
            MeshRenderer,
            MeshRasterizer,
            SoftPhongShader,
        )

        mesh1 = load_objs_as_meshes([mesh_path + "meshes/" + subject + ".obj"], device=device)

        # [-1,  0,  0,  0],
        # [ 0,  0,  1,  0],
        # [ 0,  0,  0,  1]]) """
        M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        # Initialize a camera.
        #         """ # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
        #         [1,  0,  0,  0],
        #         [ 0,  1,  0,  0],
        #         [ 0,  0,  0,  1]]) """
        camera_pose = np.linalg.inv(M) @ camera_pose111
        camera_pose111 = nerf_matrix_to_ngp(camera_pose111, scale=1.0, offset=[0, 0, 0])
        camera_pose111 = np.linalg.inv(camera_pose111)
        R = torch.tensor(camera_pose111[0:3, 0:3]).unsqueeze(0).cuda()
        t = torch.tensor(camera_pose111[0:3, 3]).t().unsqueeze(0).cuda()
        R_pytorch3d = R.clone().permute(0, 2, 1)
        T_pytorch3d = t.clone()
        R_pytorch3d[:, :, :2] *= -1
        T_pytorch3d[:, :2] *= -1
        cameras = PerspectiveCameras(
            focal_length=((fl_x, fl_y),),
            principal_point=(((cx), (cy)),),
            device=device,
            R=R_pytorch3d.cuda(),
            T=T_pytorch3d.cuda(),
            image_size=torch.tensor([[H, W]]),
            in_ndc=False,
        )

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
        # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
        # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
        # the difference between naive and coarse-to-fine rasterization.
        raster_settings = RasterizationSettings(
            image_size=(int(H), int(W)),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
        # -z direction.
        lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

        # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        depth1 = rasterizer(mesh1).zbuf
        fragments = (rasterizer(meshes222).pix_to_face)[0, :, :, :].cpu().numpy()
        lnd_verts = meshes222.verts_packed()[
            (
                meshes222.faces_packed()[
                    (fragments[xxxx_[:, 1].astype(np.int32), xxxx_[:, 0].astype(np.int32), :])[:, 0]
                ]
            )[:, 0]
        ]
        cv2.imwrite(mesh_path + "depth_lndmks.png", np.array(depth1[0, :, :, 0].cpu().numpy()) * 255.0)
        np.save(
            mesh_path + "Landmarks_vertices.npy",
            (
                meshes222.faces_packed()
                .detach()
                .cpu()
                .numpy()[(fragments[xxxx_[:, 1].astype(np.int32), xxxx_[:, 0].astype(np.int32), :])[:, 0]]
            )[:, 0],
        )
        save2 = True
    pts = backproject(depth, fl_x, fl_y, cy, cx, xf, scaling_factor=1.0)
    print(pts.shape)

    land_marks = pts[xxxx_[:, 1].astype(np.int32), xxxx_[:, 0].astype(np.int32), :]
    save = True
    if save:
        vert_000 = land_marks.reshape(-1, 3)
        np.savetxt(
            mesh_path + "Landmarks_OpenPose.off",
            vert_000,
            header="OFF\n" + str(vert_000.shape[0]) + " 0 0",
            comments="",
        )
        np.save(mesh_path + "Landmarks_OpenPose_Body.npy", pts)
        np.save(mesh_path + "Landmarks_OpenPose.npy", vert_000)
        print("Done!")
        exit()
