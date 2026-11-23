"""SMPL-X fitting to the canonical (360°) mesh of a subject.

The same fit is used twice in the pipeline (run_pipeline.sh):

  --stage normalise   (step 4) fit SMPL-X, then centre the canonical mesh and rescale it so that the fitted
                      SMPL-X scale equals --ref_scale (all subjects share one metric scale). Writes
                      <mesh_dir>/<subject>.obj/.off and applies the same transform to the cameras in
                      <DATA_DIR>/<subject>/{transforms_train,trajectory}.json. The original files are kept
                      as *_unnormalised.obj / *.json.orig and are always used as input, so re-running is safe.
  --stage register    (step 6, after the landmarks were recomputed on the normalised mesh) fit SMPL-X again
                      and save the registration used by tracking/tracker.py to <output>:
                      <subject>.pkl, <subject>.obj, <subject>_verts.npy

Fitting (2599 Adam steps):
   1. steps    0-765   scale, global orientation, translation   <- 8 body keypoints (3D, on the mesh)
   2. steps  766-1200  scale, global orientation, translation   <- Chamfer distance to the mesh
   3. steps 1201-2598  + shape, body / jaw pose, expression     <- Chamfer + 51 facial landmarks + pose priors
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import trimesh
from pytorch3d.ops import knn_points
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from paths import ASSETS_DIR, DATA_DIR
from smplx_masks import load_mask_mesh
import smplx_deformer

# OpenPose body keypoints (BODY_25 indices) and the matching SMPL-X joints
OPENPOSE_BODY = [0, 1, 2, 5, 15, 16, 17, 18]
SMPLX_BODY_JOINTS = [55, 12, 17, 16, 57, 56, 59, 58]
# 51 facial landmarks (68-point layout without the jaw contour) and the matching SMPL-X joints
FACE_LANDMARKS = list(range(17, 68))
SMPLX_FACE_JOINTS = slice(76, 127)

STEPS = 2599
STAGE1_END, STAGE2_END = 765, 1200


def initial_body_pose(device="cuda"):
    """Arms down (rest pose of the shoulders), all other joints zero."""
    pose = torch.zeros((1, 63), device=device)
    pose[0, 47] = -np.pi / 3  # left shoulder
    pose[0, 50] = np.pi / 3  # right shoulder
    return pose


def chamfer_distance(p1, p2):
    """Symmetric squared Chamfer distance, mean over the points of each cloud (as kaolin's chamfer_distance)."""
    d12 = knn_points(p1, p2, K=1).dists[..., 0].mean(dim=1)
    d21 = knn_points(p2, p1, K=1).dists[..., 0].mean(dim=1)
    return d12 + d21


def load_targets(landmarks, key_points_file, device="cuda"):
    """3D body keypoints and facial landmarks on the canonical mesh (from preprocessing/landmarks.py)."""
    face_lmks = np.load(landmarks).astype(np.float32)
    body_map = np.load(landmarks[:-4] + "_Body.npy").astype(np.float32)  # per-pixel 3D position map
    with open(os.path.join(key_points_file, "abc1_keypoints.json")) as f:
        person = json.load(f)["people"][-1]
    body_kp = np.array(person["pose_keypoints_3d"], dtype=np.float32).reshape(-1, 3)[OPENPOSE_BODY]
    body = body_map[body_kp[:, 1].astype(int), body_kp[:, 0].astype(int), :]
    return torch.tensor(body, device=device), torch.tensor(face_lmks[FACE_LANDMARKS], device=device)


def fit(smpl, scan_vertices, body_targets, face_targets, body_mask):
    """Optimise SMPL-X (and a global scale) to the canonical mesh. Returns the scale (1, 1)."""
    scan = torch.tensor(scan_vertices, dtype=torch.float32, device="cuda").unsqueeze(0)
    scale = torch.ones((1, 1), device="cuda", requires_grad=True)
    pose_init = initial_body_pose()
    shape_params = [smpl.betas, smpl.body_pose, smpl.jaw_pose, smpl.expression]
    optimizer = torch.optim.Adam(
        [smpl.transl, smpl.betas, scale, smpl.body_pose, smpl.global_orient, smpl.jaw_pose, smpl.expression,
         smpl.verts_d],
        0.2,
        betas=(0.9, 0.999),
    )
    smpl.verts_d.requires_grad = False  # per-vertex offsets are not optimised
    mse = torch.nn.MSELoss()

    loop = tqdm(range(STEPS), desc="Fitting SMPL-X")
    for i in loop:
        lr = 0.2 if i <= 765 else 0.02 if i <= 1600 else 0.002 if i <= 1900 else 0.005
        for g in optimizer.param_groups:
            g["lr"] = lr
        for p in shape_params:
            p.requires_grad = i > STAGE2_END
        optimizer.zero_grad()

        out = smpl()
        if i <= STAGE1_END:
            loss = mse(out.joints[0, SMPLX_BODY_JOINTS, :] * scale, body_targets)
        elif i <= STAGE2_END:
            loss = chamfer_distance(scan, (out.vertices * scale)[:, body_mask, :])
        else:
            bp = smpl.body_pose
            loss = (
                0.01 * chamfer_distance(scan, (out.vertices * scale)[:, body_mask, :])
                + 10 * mse(out.joints[0, SMPLX_FACE_JOINTS, :] * scale, face_targets)
                + 1e3 * torch.sum(bp[:, 0:33] ** 2)  # lower body / spine
                + 1e3 * torch.sum(bp[:, 51:] ** 2)  # elbows / wrists
                + 1e6 * torch.sum((bp[:, 45:51] - pose_init[:, 45:51]) ** 2)  # shoulders stay in the rest pose
                + 1e6 * torch.sum(bp[:, 36:42] ** 2)  # collars
            )
        loss = (100.0 * loss).sum()
        loss.backward()
        optimizer.step()
        loop.set_description(f"Fitting SMPL-X, step {i}, loss {loss.item():.4f}")
    return scale


def backup(path, suffix):
    """Keep the original of a file that is overwritten; return the path of the original."""
    root, ext = os.path.splitext(path)
    orig = root + suffix + ext if suffix != ".orig" else path + ".orig"
    if not os.path.exists(orig):
        shutil.copy(path, orig)
    return orig


def normalise(scan_path, subject, scale, ref_scale):
    """Centre and rescale the canonical mesh and the cameras (the fitted SMPL-X scale becomes ref_scale)."""
    import pymeshlab

    scan = trimesh.load(scan_path, process=False)
    factor = np.float32(ref_scale) / scale.detach().cpu().numpy().reshape(1)
    offset = scan.vertices.mean(axis=0)

    mesh_dir = os.path.dirname(scan_path)
    out_obj = os.path.join(mesh_dir, subject + ".obj")
    visual = trimesh.visual.TextureVisuals(uv=scan.visual.uv, material=scan.visual.material)
    trimesh.Trimesh(
        vertices=(np.array(scan.vertices[:, 0:3]) - offset) * factor,
        faces=scan.faces,
        face_normals=scan.face_normals,
        visual=visual,
    ).export(out_obj)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(out_obj)
    ms.save_current_mesh(os.path.join(mesh_dir, subject + ".off"))

    # cameras: the same transform in the NeRF coordinate frame (axes permuted)
    offset_cam = np.expand_dims(offset[[2, 0, 1]], axis=1)
    for name in ("trajectory.json", "transforms_train.json"):
        path = os.path.join(DATA_DIR + subject, name)
        with open(backup(path, ".orig")) as f:
            meta = json.load(f)
        frames = meta if name == "trajectory.json" else meta["frames"]
        for frame in frames:
            pose = np.array(frame["transform_matrix"], dtype=np.float32)
            pose[0:3, 3:4] = (pose[0:3, 3:4] - offset_cam) * factor
            frame["transform_matrix"] = pose.tolist()
        with open(path, "w") as f:
            json.dump(meta, f)
    print(f"Normalised {subject}: scale factor {factor[0]:.6f}, offset {offset}")


def save_registration(smpl, scale, output, subject):
    os.makedirs(output, exist_ok=True)
    s = scale.detach().cpu()
    verts = smpl().vertices.detach().cpu().numpy().squeeze()
    trimesh.Trimesh(vertices=verts * s.numpy(), faces=smpl.faces, process=False).export(
        os.path.join(output, subject + ".obj")
    )
    verts_d = smpl.verts_d.detach().cpu().numpy().squeeze()
    params = {
        "pose": smpl.body_pose.detach().cpu().numpy()[0],
        "global_oruient": smpl.global_orient.detach().cpu()[0],  # (sic) key name read by tracking/tracker.py
        "shape_betas": smpl.betas.detach().cpu()[0],
        "Scale": s[0],
        "trans": smpl.transl.detach().cpu().numpy()[0],
        "jaw_pose": smpl.jaw_pose.detach().cpu().numpy()[0],
        "expressions": smpl.expression.detach().cpu().numpy()[0],
        "verts_ddx": verts_d[0],
    }
    with open(os.path.join(output, subject + ".pkl"), "wb") as f:
        import pickle

        pickle.dump(params, f)
    np.save(os.path.join(output, subject + "_verts.npy"), verts_d)
    print(f"Registration saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="SMPL-X fitting to the canonical mesh")
    parser.add_argument("--stage", required=True, choices=["normalise", "register"])
    parser.add_argument("--model-folder", required=True, type=str, help="folder containing smplx/SMPLX_*.npz")
    parser.add_argument("--subject", required=True, type=str)
    parser.add_argument("--Mesh_Path", required=True, type=str, help="folder of the canonical mesh <subject>.obj")
    parser.add_argument("--Landmarks", required=True, type=str, help="Landmarks_OpenPose.npy (preprocessing/landmarks.py)")
    parser.add_argument("--key_points_file", required=True, type=str, help="Keypoints folder (abc1_keypoints.json)")
    parser.add_argument("--output", type=str, default="", help="registration output folder (--stage register)")
    parser.add_argument("--mask_folder", type=str, default=ASSETS_DIR)
    parser.add_argument("--gender", type=str, default="neutral")
    parser.add_argument("--ref_scale", type=float, default=3.9808340072631836,
                        help="SMPL-X scale all subjects are normalised to (--stage normalise)")
    args = parser.parse_args()

    scan_path = os.path.join(args.Mesh_Path, args.subject + ".obj")
    if args.stage == "normalise":
        scan_path = backup(scan_path, "_unnormalised")  # always start from the original mesh
    scan = trimesh.load(scan_path, process=False)

    smpl = smplx_deformer.create(
        os.path.expanduser(args.model_folder),
        model_type="smplx",
        gender=args.gender,
        use_face_contour=True,
        num_betas=10,
        num_expression_coeffs=50,
        ext="npz",
        create_global_orient=True,
        create_transl=True,
        global_orient=torch.zeros((1, 3), device="cuda"),
        body_pose=initial_body_pose(),
    ).cuda()
    body_targets, face_targets = load_targets(args.Landmarks, args.key_points_file)
    body_mask = load_mask_mesh("SMPL-X22.obj", args.mask_folder).visual.vertex_colors[:, 0] == 255

    scale = fit(smpl, scan.vertices, body_targets, face_targets, body_mask)
    print("SMPL-X scale:", scale.item())
    if args.stage == "normalise":
        normalise(scan_path, args.subject, scale, args.ref_scale)
    else:
        save_registration(smpl, scale, args.output, args.subject)


if __name__ == "__main__":
    main()
