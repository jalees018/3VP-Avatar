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
from paths import DECA_RESULTS, SMPLX_MODEL_DIR, TRACKER_DIR

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import warnings

warnings.filterwarnings("ignore")
from tqdm import tqdm
import trimesh
import cv2
import util

import pickle
import time
import json
import pytorch3d.transforms as transforms1
import smplx_deformer

import nn_meshes
import nvdiffrast.torch as dr
from math import factorial
from blur import gaussian_blur

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10.0 * torch.log(x) / torch.log(torch.Tensor([10.0]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

BODY_POSE_111_2 = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]
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
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
]
OP2SMPL_X_No_EYES = [
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
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
OP2SMPL_X_No_EYES_JOINTS = [x + 59 for x in OP2SMPL_X_No_EYES]
MOUTH = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
SHOULDERSMPL_X = [12, 16, 17]  # Init SMPL, pose with mean smpl pose, as in ch.registration
OP2SHOULDERSMPL_X = [1, 5, 2]  # Init SMPL, pose with mean smpl pose, as in ch.registration
bpp_111 = torch.zeros((1, 63)).cuda()  # Roll, Yaw, Pitch
bpp_111[0, 45] = 0.0
bpp_111[0, 46] = 0.0
bpp_111[0, 47] = -np.pi / 3
bpp_111[0, 48] = 0.0
bpp_111[0, 49] = 0.0
bpp_111[0, 50] = np.pi / 3


# https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay


def savitzky_smooth(lmks, window, order, dim):
    smoothed_lmks = []
    for i in range(len(lmks[1])):
        coords = []
        for axis in range(dim):
            y = lmks[:, i, axis]
            yhat = savitzky_golay(y, window, order)
            coords.append(yhat)
        smoothed_lmks.append(np.array(coords))
    smoothed_lmks = np.array(smoothed_lmks).transpose(2, 0, 1)
    return smoothed_lmks


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode="valid")


def laplace_regularizer_const(mesh_verts, mesh_faces):
    term = torch.zeros_like(mesh_verts)
    norm = torch.zeros_like(mesh_verts[..., 0:1])

    v0 = mesh_verts[mesh_faces[:, 0], :]
    v1 = mesh_verts[mesh_faces[:, 1], :]
    v2 = mesh_verts[mesh_faces[:, 2], :]

    term.scatter_add_(0, mesh_faces[:, 0:1].repeat(1, 3), (v1 - v0) + (v2 - v0))
    term.scatter_add_(0, mesh_faces[:, 1:2].repeat(1, 3), (v0 - v1) + (v2 - v1))
    term.scatter_add_(0, mesh_faces[:, 2:3].repeat(1, 3), (v0 - v2) + (v1 - v2))

    two = torch.ones_like(v0) * 2.0
    norm.scatter_add_(0, mesh_faces[:, 0:1], two)
    norm.scatter_add_(0, mesh_faces[:, 1:2], two)
    norm.scatter_add_(0, mesh_faces[:, 2:3], two)

    term = term / torch.clamp(norm, min=1.0)

    return torch.mean(term**2)


def transform_pos111(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = pos
    return torch.matmul(posw, t_mtx.t())[None, ...]


def render_uv(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolutionw: int, resolutionh: int):
    # Setup TF graph for reference.
    pos_clip = transform_pos111(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolutionh, resolutionw])
    # color       = dr.antialias(color, rast_out, pos_clip, pos_idx) """
    enable_mip = True
    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs="all")
        color_111 = dr.texture(tex[None, ...], texc, texd, filter_mode="linear-mipmap-linear", max_mip_level=5)
        color_111 = dr.antialias(color_111, rast_out, pos_clip, pos_idx)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color_111 = dr.texture(tex[None, ...], texc, filter_mode="linear")
        color_111 = dr.antialias(color_111, rast_out, pos_clip, pos_idx)
    return rast_out, color_111, None


def render(glctx, mtx, pos, pos_idx, col, col_idx, resolutionw: int, resolutionh: int):
    # Setup TF graph for reference.
    pos_clip = transform_pos111(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolutionh, resolutionw])
    color, _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)
    color_111 = color
    return rast_out, color_111


def render_bkg_uv(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolutionw: int, resolutionh: int):
    # Setup TF graph for reference.
    pos_clip = transform_pos111(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolutionh, resolutionw])
    # color       = dr.antialias(color, rast_out, pos_clip, pos_idx) """
    enable_mip = False
    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs="all")
        color_111 = dr.texture(tex[None, ...], texc, texd, filter_mode="linear-mipmap-linear", max_mip_level=9)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color_111 = dr.texture(tex[None, ...], texc, filter_mode="linear")
    color_111 = torch.where(rast_out[..., 3:] > 0, color_111, 1.0 * torch.tensor(1.0).cuda())
    return rast_out, color_111


# Cube pose fitter.
def perspective_projection(points, rotation, translation, focal_lengthx, focal_lengthy, camera_center, H):
    """ """ """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_lengthx
    K[:, 1, 1] = focal_lengthy
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum("bij,bkj->bki", rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum("bij,bkj->bki", K, projected_points)
    return projected_points[:, :, :-1]


def fit_pose(
    canonical_mesh_path="",
    smplx_meshes_path="",
    subject="",
    exp="Results_",
    input_images_path="",
    base_file="",
    max_iter=10000,
    repeats=1,
    log_interval=10,
    display_interval=None,
    display_res=256,
    lr_base=0.0008,
    lr_falloff=1.0,
    nr_base=1.0,
    nr_falloff=1e-4,
    grad_phase_start=0.5,
    resolution_w=512,
    resolution_h=512,
    out_dir=None,
    log_fn=None,
    mp4save_interval=None,
    mp4save_fn=None,
    use_opengl=False,
):

    log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    else:
        mp4save_interval = None
    model_folder = SMPLX_MODEL_DIR
    # Some input geometry contains vertex positions in (N, 4) (with v[:,3]==1).  Drop
    # the last column in that case.
    nn_meshes.main(canonical_mesh_path + subject + ".off", smplx_meshes_path, subject)
    mesh = trimesh.load(canonical_mesh_path + subject + ".off", indd=3, process=False)
    # Create position/triangle index tensors
    pos_idx = torch.from_numpy(mesh.faces.astype(np.int32)).cuda()
    vtx_pos = torch.from_numpy(mesh.vertices[:, 0:3].astype(np.float32)).cuda()
    vtx_pos = torch.cat([vtx_pos, torch.ones([vtx_pos.shape[0], 1]).cuda()], axis=1)
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtx_pos.shape[0]))

    meshhh = trimesh.load(canonical_mesh_path + subject + ".obj", process=False)
    uv = torch.from_numpy(meshhh.visual.uv.astype(np.float32)).cuda()
    uv_idx = torch.from_numpy(meshhh.faces.astype(np.int32)).cuda()
    aaaaa111 = cv2.cvtColor(cv2.flip(cv2.imread(canonical_mesh_path + subject + "_111.jpg"), 0), cv2.COLOR_BGR2RGB)
    tex = torch.from_numpy(aaaaa111.astype(np.float32) / 255.0).cuda()
    meshhh_Decoy = trimesh.load(canonical_mesh_path + subject + ".off", indd=3, process=False)
    v0 = meshhh_Decoy.vertices
    trimesh.grouping.merge_vertices(meshhh_Decoy, merge_tex=True, merge_norm=True)
    v1 = meshhh_Decoy.vertices
    # for i in v0:
    #     for j in v1:
    aa, ids = np.unique(v0, return_index=True, axis=0)
    ids = np.sort(ids)

    import pickle as pkl

    abcdefgh = pkl.load(open(smplx_meshes_path + "reg_results/" + subject + ".pkl", "rb"))
    p = abcdefgh["pose"]
    gb = abcdefgh["global_oruient"]
    b = abcdefgh["shape_betas"]
    s = abcdefgh["Scale"]
    t = abcdefgh["trans"]
    jaw = abcdefgh["jaw_pose"]
    exp0 = abcdefgh["expressions"]
    verts_d = 0.0 * torch.tensor(np.load(smplx_meshes_path + "reg_results/" + subject + "_verts.npy")).squeeze(0).cuda()

    with open(os.path.join(base_file, "transforms_train.json"), "r") as f:
        transform = json.load(f)

    with open(os.path.join(base_file, "trajectory.json"), "r") as f:
        pose = json.load(f)[-1]
    camera_pose111 = pose["transform_matrix"]
    # load intrinsics
    downscale = 1.0
    H = resolution_h
    W = resolution_w

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
    focal_lengthx = fl_x
    focal_lengthy = fl_y
    results_dir = TRACKER_DIR + exp + subject
    continue_load = False
    if continue_load == True:
        img_idxx_111 = 372  # 428 #432 #351 #331 ##256 #248 #140 #102
        print("Reloading from the ", img_idxx_111)
        abcdefgh = pkl.load(open(results_dir + "/result_final_Real_tt_" + str(img_idxx_111) + ".pkl", "rb"))
        p2 = abcdefgh["pose"]
        gb2 = abcdefgh["global_orient"]
        t2 = abcdefgh["transl"]
        j2 = abcdefgh["jaw_pose"]
        exp2 = abcdefgh["expressions"]
    else:
        p2 = p
        gb2 = gb
        t2 = t
        j2 = jaw
        exp2 = exp0
    model1 = smplx_deformer.create(
        model_folder,
        model_type="smplx",
        gender="neutral",
        use_face_contour=True,  # False, #False,
        num_betas=10,
        global_orient=torch.tensor(gb2).unsqueeze(0).cuda(),
        body_pose=torch.tensor(p2).unsqueeze(0).cuda(),
        jaw_pose=torch.tensor(j2).unsqueeze(0).cuda(),
        leye_pose=torch.zeros((1, 3)).cuda(),
        reye_pose=torch.zeros((1, 3)).cuda(),
        betas=b.detach().unsqueeze(0).cuda(),
        transl=torch.tensor(t2).unsqueeze(0).cuda(),
        verts_d=verts_d.unsqueeze(0),
        num_expression_coeffs=50,
        ext="npz",
    ).cuda()

    mesh0 = trimesh.load(smplx_meshes_path + "reg_results/" + subject + ".obj", process=False)
    m_mesh = trimesh.load(canonical_mesh_path + subject + ".off", indd=3, process=False)
    fac_c_idx_111 = np.load(smplx_meshes_path + "reg_results/" + subject + "_NN.npy")
    verts_c_idx_111 = (mesh0.faces[fac_c_idx_111[0]])[:, 0]
    ptsss111 = torch.tensor(m_mesh.vertices[:, 0:3]).float().cuda() / s.float().cuda()
    homogen_coord = torch.ones([ptsss111.shape[0], 1]).unsqueeze(-1).cuda()
    gbb = transforms1.axis_angle_to_matrix(gb.detach().unsqueeze(0))
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    bp2 = torch.tensor(p2).unsqueeze(0).cuda()
    g22 = torch.tensor(gb2).cuda()
    ttt = torch.tensor(t2).cuda()
    jpp = torch.tensor(j2).unsqueeze(0).cuda()
    exp = torch.zeros((1, 50)).cuda()
    exp[0, 0:50] = torch.tensor(exp2[:]).cuda()
    l_eye = torch.zeros_like(jpp).cuda()
    r_eye = torch.zeros_like(jpp).cuda()
    g22.requires_grad = True
    ttt.requires_grad = True
    jpp.requires_grad = True
    exp.requires_grad = True
    l_eye.requires_grad = True
    r_eye.requires_grad = True
    exp_eeo_111 = exp.clone().detach()
    zzz = torch.tensor([[0.0, 0.0, 0.0]]).cuda()
    zzz = zzz.detach()
    gb_init = torch.inverse(gbb)

    lbsw_1112, b_ss0, _ = model1.forward2(
        body_pose=bp2,
        jaw_pose=torch.tensor(j2).unsqueeze(0).cuda(),
        leye_pose=torch.zeros((1, 3)).cuda(),
        reye_pose=torch.zeros((1, 3)).cuda(),
        transl=zzz,
        global_orient=g22.detach().unsqueeze(0),
        expression=exp,
        betas=b.detach().unsqueeze(0).cuda(),
        verts_d=verts_d.unsqueeze(0),
    )

    lbsw_11122 = torch.inverse(lbsw_1112[0])
    ddd_222 = ptsss111 - torch.tensor(t).unsqueeze(0).cuda()
    ddd_222 = torch.cat([ddd_222, homogen_coord[:, :, 0]], dim=1)
    defor_111 = torch.bmm((lbsw_11122)[verts_c_idx_111, :, :], ddd_222.unsqueeze(-1))
    v_posed_homo = defor_111[:, 0:3, :] - (b_ss0[0])[verts_c_idx_111, :].unsqueeze(-1)
    v_posed_homo = torch.cat([v_posed_homo, homogen_coord], dim=1)

    camera_pose111 = np.concatenate((camera_pose111, np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)), axis=0)
    M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    xf = np.linalg.inv(M) @ camera_pose111

    mtx_extr = torch.tensor(xf[0:3, 0:4]).type(torch.float32).cuda()
    RR = mtx_extr[0:3, 0:3]
    tt = mtx_extr[0:3, 3:]

    R_111 = RR.T
    t_111 = -torch.matmul(RR.T, tt)
    mtx_extr_111 = torch.cat((R_111, t_111), dim=1)
    mtx_extr_111 = torch.cat((mtx_extr_111, torch.tensor([[0.0, 0.0, 0.0, 1.0]]).type(torch.float32).cuda()), dim=0)

    # simple version for working with CWD
    num_imgs = (
        len(
            [
                name
                for name in os.listdir(input_images_path + subject + "/Images/")
                if os.path.isfile(os.path.join(input_images_path + subject + "/Images/", name))
            ]
        )
        - 100
    )
    print("Number of Images:", num_imgs)
    start = 0  # start + 1
    restart = 0  # 429
    if continue_load == True:
        restart = img_idxx_111 + 1  # 103
    else:
        restart = 0  # 23

    v_posed_homo = v_posed_homo.detach()
    on111 = torch.ones([v_posed_homo.shape[0], 1, 1]).cuda()
    on111.detach()
    onee = torch.ones((1, 1, 1)).cuda()
    onee = onee.detach()

    camera_pose = xf
    Time_Begin = time.time()

    lsttt = os.listdir(input_images_path + subject + "/Images/_results/")
    offset = int(min([int(((iii).split("/")[-1])[:-4]) for iii in lsttt]))  # 1560
    print("Offset:", offset)

    key_point_loss = True
    lmks = []
    shld_lmks = []
    with open(input_images_path + subject + "/Keypoints/keypoints.pkl", "rb") as input_file:
        preds = pickle.load(input_file)
    # keys are image paths as written by preprocessing/detect_keypoints.py; match on the file name only
    preds = {os.path.basename(k): v for k, v in preds.items()}
    for rep in tqdm(range(0, num_imgs)):
        lmks.append(((preds[str(rep + offset).zfill(4) + ".png"])[0])[:, 0:2])
    lmkss = np.array(lmks)

    smoothed_lmks = savitzky_smooth(lmkss, 7, 3, 2)
    xxxx_ = torch.tensor(smoothed_lmks).cuda() * (H / 512)

    eye_gaze = np.zeros((num_imgs, 2, 2))
    IMAGE_FILES = []
    # For static images:
    # with mp_face_mesh.FaceMesh(
    #     min_detection_confidence=0.5) as face_mesh:
    #     for file in tqdm(range(restart,num_imgs)):

    #         for face_landmarks in results.multi_face_landmarks:

    bp21 = bp2[:, 33:36]
    bp23 = bp2[:, 42:45]
    bp21.requires_grad = True
    bp23.requires_grad = True

    jpp_reg = []
    exp_reg = []
    for iit in range(0, num_imgs):

        jppr = np.load(DECA_RESULTS + subject + "/" + str(iit + offset).zfill(4) + "_pose.npy")
        expr = np.load(DECA_RESULTS + subject + "/" + str(iit + offset).zfill(4) + "_expressions.npy")
        jpp_reg.append(jppr)
        exp_reg.append(expr)
    jpp_reg = torch.tensor(savitzky_smooth(np.stack(jpp_reg, axis=0), 5, 3, 3)).float()
    exp_reg = torch.tensor(savitzky_smooth(np.stack(exp_reg, axis=0), 5, 3, 50)).float()

    for rep in tqdm(range(restart, num_imgs)):
        print("Frame:", rep)
        # Modelview + projection matrix.
        color_all = (
            torch.tensor(
                cv2.cvtColor(
                    cv2.flip(
                        cv2.resize(
                            cv2.imread(
                                input_images_path + subject + "/Images/_results/" + str(rep + offset).zfill(4) + ".png"
                            ),
                            (resolution_w, resolution_h),
                            interpolation=cv2.INTER_AREA,
                        ),
                        0,
                    ),
                    cv2.COLOR_BGR2RGBA,
                ).astype(np.float32)
            )
            .unsqueeze(0)
            .cuda()
        )
        color = color_all[:, :, :, 0:3]
        msk_111 = color_all[0, :, :, 3] != 0
        msk_111 = (msk_111).unsqueeze(-1).repeat((1, 1, 3)).unsqueeze(0)
        color = (gaussian_blur(color.permute((0, 3, 1, 2)), [3, 3], sigma=[4, 4])).permute((0, 2, 3, 1))
        color = color / 255.0

        s = s.cuda()
        s = s.detach()
        #     for idx2, person_data in enumerate(data['people']):
        dx = torch.zeros((v_posed_homo.shape[0], 3)).cuda()
        dx.requires_grad = True
        bp2 = bp2.detach()
        bp20 = bp2[:, 0:33]
        bp22 = bp2[:, 36:42]
        bp24 = bp2[:, 45:]
        optimizer = torch.optim.Adam(
            [
                {"params": bp21},
                {"params": bp23},
                {"params": g22},
                {"params": ttt},
                {"params": jpp},
                {"params": exp},
                {"params": dx},
            ],
            betas=(0.9, 0.999),
            lr=lr_base,
        )
        torch.cuda.synchronize()
        lr = lr_base
        if rep == restart:
            vtx_pos111_p = 0.0
            dx_p = 0.0
            lr = 0.001
        else:
            vtx_pos111_p = defor_11111.clone().detach()
            dx_p = dx.clone().detach()
        bp2_p = bp2.clone().detach()
        g22_p = g22.clone().detach()
        ttt_p = ttt.clone().detach()
        exp_p = exp.clone().detach()
        jpp_p = jpp.clone().detach()
        mvp = torch.tensor(
            torch.matmul(
                util.projection2(focal_lengthx, focal_lengthy, cx, cy, 0.01, 50.0, H).cuda(),
                util.translate2(0.0, 0.0, 0.0).type(torch.float32).cuda(),
            ),
            device="cuda",
        )

        jaw_p = jpp_reg[rep].cuda()
        exprs_p = exp_reg[rep].cuda()
        for it in range(max_iter + 1):
            if it <= 600:  # 200
                dx.requires_grad = False
                wts_lap = 0.0
            if it <= 250:
                wts_smoothing = 0.0
            else:
                wts_smoothing = 4.0
            mtx_total_opt = torch.matmul(mvp, mtx_extr_111)
            bp2 = torch.cat((bp20, bp21, bp22, bp23, bp24), 1)
            # Set learning rate.
            itf = 1.0 * it / max_iter
            nr = nr_base * nr_falloff**itf
            if it != 0 and log_interval and (it % log_interval == 0):
                lr = lr / 2.0
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            lbsw_111, b_ss, joints = model1.forward2(
                body_pose=bp2,
                jaw_pose=jpp,
                leye_pose=l_eye,
                reye_pose=r_eye,
                transl=zzz,
                global_orient=g22.unsqueeze(0),
                expression=exp,
                betas=b.unsqueeze(0),
                verts_d=verts_d.unsqueeze(0),
            )

            b_ss_111 = torch.cat([(b_ss[0])[verts_c_idx_111, :].unsqueeze(-1), 0.0 * on111], axis=1)
            defor_111 = torch.bmm((lbsw_111[0])[verts_c_idx_111, :, :], v_posed_homo + b_ss_111)
            defor_1111 = defor_111[:, :3, :]
            defor_11111 = s * defor_1111 + s * (ttt).unsqueeze(-1)
            joints2 = s * joints[0] + s * (ttt)

            vtx_pos111 = torch.cat([defor_11111.squeeze(-1) + dx, torch.ones([defor_11111.shape[0], 1]).cuda()], axis=1)
            _, color_opt, _ = render_uv(
                glctx, mtx_total_opt, vtx_pos111, pos_idx, uv, uv_idx, tex, resolution_w, resolution_h
            )
            if rep == restart and it == 0 and key_point_loss == True:
                cv2.imwrite(
                    TRACKER_DIR + subject + "_first_frame_render.png",
                    cv2.flip(
                        cv2.resize(
                            color_opt[0].detach().cpu().numpy() * 255.0,
                            (resolution_w, resolution_h),
                            interpolation=cv2.INTER_AREA,
                        ),
                        0,
                    ),
                )
                lnd_verts_ids = np.load(smplx_meshes_path + "Landmarks_vertices.npy")
            diff = (
                torch.sum(
                    torch.abs(
                        (gaussian_blur(color_opt.permute((0, 3, 1, 2)), [3, 3], sigma=[4 / 255, 4 / 255])).permute(
                            (0, 2, 3, 1)
                        )[msk_111]
                        - color[msk_111]
                    )
                )
                / msk_111.float().sum()
            )

            if rep == restart:
                wts_smoothing = 0.00
                wts_smoothing_dx = 0.00
            else:
                wts_smoothing_dx = 16
            jaw_wght = 0.1
            exp_weight = 0.1
            loss = (
                diff
                + jaw_wght * torch.mean((jpp - jaw_p) ** 2)
                + 0.0 * 2000 * torch.mean((ttt - ttt_p) ** 2)
                + 0.0 * 2000 * torch.mean((g22_p - g22) ** 2)
                + 10000 * torch.mean((bp2[:, 0:33]) ** 2)
                + 10000 * torch.mean((bp2[:, 51:]) ** 2)
                + 30 * torch.sum((bp2[:, 45:51] - bpp_111[:, 45:51]) ** 2)
                + 0.001 * torch.mean((bp2_p - bp2) ** 2)
                + exp_weight * torch.mean((exp - exprs_p) ** 2)
                + 0.0 * 0.01 * torch.mean((exp - exp_p) ** 2)
                + 0.1 * torch.mean((jpp - jpp_p) ** 2)
                + 0.1 * wts_smoothing_dx * torch.mean((dx - dx_p) ** 2)
                + wts_smoothing * torch.mean((defor_11111 - vtx_pos111_p) ** 2)
                + wts_lap
                * laplace_regularizer_const(
                    defor_11111.squeeze(-1) + dx, torch.from_numpy(mesh.faces.astype(np.int32)).long().cuda()
                )
            )
            if key_point_loss == True:
                ppr = perspective_projection(
                    (joints2).unsqueeze(0),
                    torch.inverse(torch.tensor(camera_pose[0:3, 0:3])).unsqueeze(0).float().cuda(),
                    torch.bmm(
                        torch.inverse(torch.tensor(camera_pose[0:3, 0:3])).unsqueeze(0),
                        torch.tensor(-camera_pose[0:3, 3:]).unsqueeze(0),
                    )[0]
                    .t()
                    .float()
                    .cuda(),
                    focal_lengthx,
                    focal_lengthy,
                    torch.tensor([[cx, cy]]).float().cuda(),
                    resolution_h,
                )
                ppr[0, :, 0] = 2 * cx - ppr[0, :, 0]

                if it == 0 or rep == restart:
                    key_wght = 0.0000
                elif it == 200 and rep != restart:
                    key_wght = 10 * 0.1 * 0.1 * 0.1 * 0.1
                key_point_losss = key_wght * torch.mean(
                    (ppr[0, OP2SMPL_X_No_EYES_JOINTS, :] - xxxx_[rep, OP2SMPL_X_No_EYES, :]) ** 2
                )
                loss = loss + key_point_losss
            # Measure image-space loss and update best found pose.

            if log_interval and (it % log_interval == 0):
                print("Loss:", loss.item())

            # Run gradient training step.
            # if itf >= grad_phase_start:

            ###############################################################################################################################################################################################
            ###############################################################################################################################################################################################
            # """ import matplotlib.pyplot as plt

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it == max_iter:
                # Show/save image.
                display_image = display_interval and (it % display_interval == 0)
                save_mp4 = mp4save_interval and (it % mp4save_interval == 0)
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                vertices222 = (
                    model1.forward3(
                        body_pose=bp2.detach(),
                        jaw_pose=jpp.detach(),
                        leye_pose=l_eye.detach(),
                        reye_pose=r_eye.detach(),
                        global_orient=g22.detach().unsqueeze(0),
                        transl=ttt.detach().unsqueeze(0),
                        expression=exp.detach(),
                        betas=b.detach().unsqueeze(0).cuda(),
                        verts_d=verts_d.unsqueeze(0),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                m11 = trimesh.Trimesh(vertices=(vertices222) * s.cpu().numpy(), faces=model1.faces, process=False)
                m11.export(results_dir + "/result_final_Real_tt_" + str(rep) + ".obj")
                mesh_222 = trimesh.Trimesh(
                    vertices=vtx_pos111[:, 0:3].detach().cpu().numpy()[ids], faces=meshhh_Decoy.faces, process=False
                )
                mesh_222.export(results_dir + "/Results_Deformed_tt_" + str(rep) + ".obj")
                for p, gbbb, jp, ttrans, expp in zip(
                    bp2.detach().cpu().numpy(),
                    g22.detach().unsqueeze(0).cpu().numpy(),
                    jpp.detach().cpu().numpy(),
                    ttt.detach().unsqueeze(0).cpu().numpy(),
                    exp.detach().cpu().numpy(),
                ):
                    smpl_dict = {
                        "pose": p,
                        "global_orient": gbbb,
                        "jaw_pose": jp,
                        "transl": ttrans,
                        "expressions": expp,
                    }
                pkl.dump(smpl_dict, open(results_dir + "/result_final_Real_tt_" + str(rep) + ".pkl", "wb"))
                scene = pyrender.Scene()
                scene.add(pyrender.Mesh.from_trimesh(m11, smooth=False))

                # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
                camera = pyrender.camera.IntrinsicsCamera(fl_x, fl_y, cx, cy)

                # Create ray batch

                nc = pyrender.Node(camera=camera, matrix=camera_pose)
                scene.add_node(nc)

                # Set up the light -- a point light in the same spot as the camera
                light = pyrender.PointLight(color=np.ones(3), intensity=8.0)
                nl = pyrender.Node(light=light, matrix=camera_pose)
                scene.add_node(nl)

                # Render the scene
                r = pyrender.OffscreenRenderer(resolution_w, resolution_h)
                color_111, depth = r.render(scene)
                # """ pts = backproject(depth, 449.29763794, 256, 256, pose)
                r.delete()
                if display_image or save_mp4:
                    img_ref = color[0].detach().cpu().numpy()
                    _, color_opt_111 = render_bkg_uv(
                        glctx, mtx_total_opt, vtx_pos111, pos_idx, uv, uv_idx, tex, resolution_w, resolution_h
                    )
                    img_opt = color_opt_111[0].detach().cpu().numpy()
                    diff1 = abs(torch.sqrt(((color_opt_111 - color) ** 2)[0])).detach().cpu().numpy()
                    img_best = np.flip(color_111 / 255.0, axis=0)
                    result_image = np.concatenate([img_ref, img_best, img_opt, diff1], axis=1)[::-1]
                    path_pose = out_dir + "/Images/" + subject + "_tracking"
                    if not os.path.exists(path_pose):
                        os.makedirs(path_pose)
                    cv2.imwrite(
                        path_pose + "/" + str(rep).zfill(4) + ".png",
                        cv2.cvtColor(
                            np.clip(np.rint(result_image * 255.0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
                        ),
                    )
                    # if display_image:

    # """ if writer is not None:


def main():
    parser = argparse.ArgumentParser(description="Cube pose fitting example")
    parser.add_argument("--opengl", help="enable OpenGL rendering", action="store_true", default=False)
    parser.add_argument("--outdir", help="specify output directory", default="")
    parser.add_argument("--display-interval", type=int, default=0)
    parser.add_argument("--mp4save-interval", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument("--canonical_mesh_path", type=str, help="specify caninical mesh directory", default="")
    parser.add_argument("--smplx_meshes_path", type=str, help="specify meshes directory", default="")
    parser.add_argument("--subject", type=str, help="specify subject name", default="")
    parser.add_argument("--input_images_path", type=str, help="specify images directory", default="")
    parser.add_argument("--base_file", type=str, help="specify base data directory", default="")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--exp", type=str, default="Results_", help="experiment name; results go to <TRACKER_DIR>/<exp><subject>"
    )

    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        out_dir = TRACKER_DIR + "pose"
        print(f"Saving results under {out_dir}")
    else:
        out_dir = None
        print("No output directory specified, not saving log or images")

    fit_pose(
        canonical_mesh_path=args.canonical_mesh_path,
        smplx_meshes_path=args.smplx_meshes_path,
        subject=args.subject,
        exp=args.exp,
        input_images_path=args.input_images_path,
        base_file=args.base_file,
        max_iter=args.max_iter,
        repeats=args.repeats,
        log_interval=100,
        display_interval=args.display_interval,
        out_dir=out_dir,
        log_fn=args.subject + "_log.txt",
        mp4save_interval=args.mp4save_interval,
        mp4save_fn=args.subject + "_tracking.mp4",
        use_opengl=args.opengl,
        resolution_w=args.width,
        resolution_h=args.height,
    )

    print("Done.")


if __name__ == "__main__":
    main()
