import os
import argparse
import numpy as np

import tqdm

import trimesh
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (paths.py)
from paths import ASSETS_DIR, DATA_DIR, IMAGES_DIR, TRACKER_DIR, WORK_DIR
from smplx_masks import load_mask_mesh

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

import json

pose_111 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
pose_1111 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

W = 512
H = 512

arr = np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sampling Maps")

    parser.add_argument("--subject", type=str, default="", help="Name of the subject")
    parser.add_argument("--workspace", type=str, default="", help="Name of the Workspace")
    parser.add_argument("--render_tracker", action="store_true", help="Render Tracker")
    parser.add_argument("--exp_track", type=str, default="Results_", help="Expression Tracking name")
    parser.add_argument("--num_imgs", type=int, default=3000, help="Number of Tracking image Frames")

    args = parser.parse_args()

    subject = args.subject
    workspace = args.workspace
    render_tracker = args.render_tracker
    exp_track = args.exp_track
    num_imgs = args.num_imgs

    root_path = DATA_DIR + subject
    obj_path2 = WORK_DIR + workspace + "/reg_results/" + subject + ".obj"
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

    path1 = TRACKER_DIR + exp_track + subject + "/"
    path2 = IMAGES_DIR + subject + "/Images/error_maps"

    if not os.path.exists(path2):
        os.makedirs(path2)

    face_msk = []

    facess = np.load(ASSETS_DIR + "canonical_template.npy")
    mesh_msks2 = load_mask_mesh("Face_msks.obj")
    m_vv2 = (mesh_msks2.visual.vertex_colors[:, 1] == 0) & (mesh_msks2.visual.vertex_colors[:, 0] == 255)
    boxf_face_colors2 = np.ones((m_vv2.shape[0], 3)) * 255
    cc3 = boxf_face_colors2 * np.expand_dims(m_vv2, axis=1)
    with open(WORK_DIR + subject + ".json") as ff:
        fr = json.load(ff)
    num_imgs2 = fr[11]
    mesh_msks2 = load_mask_mesh("Face_msks2.obj")
    m_vv2 = (mesh_msks2.visual.vertex_colors[:, 1] == 0) & (mesh_msks2.visual.vertex_colors[:, 0] == 255)
    boxf_face_colors2 = np.ones((m_vv2.shape[0], 3)) * 255
    cc4 = boxf_face_colors2 * np.expand_dims(m_vv2, axis=1)

    flags = pyrender.RenderFlags.SKIP_CULL_FACES

    print("Length:", len(lstt_111))
    for iit in tqdm.tqdm(range(0, len(lstt_111) + num_imgs), desc=f"Loading Depth Maps"):

        if iit < len(lstt_111):
            mesh_111 = trimesh.load(obj_path2, process=False)
            mesh_1111 = trimesh.Trimesh(vertices=mesh_111.vertices, faces=facess)
            i = lstt_111[iit]
            frame = meta111[lstt_222.index(i)]
            camera_pose111 = np.array(frame["transform_matrix"], dtype=np.float32)
            mesh_1111.visual.vertex_colors = cc3

        else:
            mesh_111 = trimesh.load(path1 + "result_final_Real_tt_" + str(iit - len(lstt_111)) + ".obj", process=False)
            mesh_1111 = trimesh.Trimesh(vertices=mesh_111.vertices, faces=facess)
            flags = pyrender.RenderFlags.SKIP_CULL_FACES
            frame = meta111[lstt_222.index(j)]
            camera_pose111 = np.array(frame["transform_matrix"], dtype=np.float32)
            if (iit - len(lstt_111)) < num_imgs2:
                mesh_1111.visual.vertex_colors = cc3  # cc
            else:
                mesh_1111.visual.vertex_colors = cc4  # cc

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
        col2 = col.copy()
        r.delete()
        if render_tracker:
            col2[col2 <= 160] = 0.0
            col2[col2 > 160] = 1.0
            np.save(path2 + "/" + str(iit).zfill(4) + "_tracker.npy", (col2.copy()))
        else:
            msk = col2 <= 160
            col2[msk] = 1.0
            col2[~msk] = 4.0
            if iit >= len(lstt_111):
                if col2.sum() == 0:
                    raise Exception("Zero: 0 Depth!")
            np.save(path2 + "/" + str(iit).zfill(4) + ".npy", (col2.copy()))
