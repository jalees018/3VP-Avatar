"""Command line options shared by ``train.py`` and ``render.py``.

All options are forwarded to the renderer via ``**vars(opt)``, so option names
must stay in sync with the keyword arguments used in ``nerf/renderer.py``.
"""

import argparse

# Defaults that differ between training and rendering (values used for the paper).
TRAIN_DEFAULTS = dict(
    modes="Double_train", iters=140000 * 6, num_rays=1024, num_steps=600, max_ray_batch=4096, dt_gamma=1 / 256
)
RENDER_DEFAULTS = dict(modes="single", iters=60000, num_rays=4096, num_steps=800, max_ray_batch=256, dt_gamma=1 / 128)


def get_parser(stage):
    assert stage in ("train", "render")
    parser = argparse.ArgumentParser(description=f"3VP-Avatar: {stage}")
    parser.add_argument("path", type=str, help="dataset directory (contains transforms_*.json, trajectory.json)")
    parser.add_argument("-O", action="store_true", help="equals --fp16 --cuda_ray --preload")
    parser.add_argument("--workspace", type=str, default="workspace", help="output / checkpoint directory")
    parser.add_argument("--subject", type=str, default="", help="name of the subject")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modes", type=str, help="rendering mode ('single' renders with the corrective network only)")
    parser.add_argument("--notrainposes", action="store_true", help="do not use the training poses")
    parser.add_argument("--trainMapping", action="store_true", help="stage 2: train the mapping network only")
    parser.add_argument("--freeze", action="store_true", help="freeze the network")

    # tracking / data
    parser.add_argument("--train_imgs", type=int, default=4000, help="number of training images")
    parser.add_argument(
        "--expr",
        type=str,
        default="Results_",
        help="tracking experiment name (results in <TRACKER_DIR>/<expr><subject>, see paths.py)",
    )

    # re-enactment (render only)
    parser.add_argument("--drive", action="store_true", help="drive the avatar with another subject's tracking")
    parser.add_argument(
        "--expression_source",
        type=str,
        default="deca",
        choices=["deca", "tracker"],
        help="expression / jaw input of the mapping network when rendering the subject's own sequence: raw DECA "
        "(as in the paper) or the tracked values (<TRACKER_DIR>/<expr><subject>/result_final_Real_tt_<n>.pkl)",
    )
    parser.add_argument("--driver", type=str, default="", help="subject whose tracking drives the avatar")
    parser.add_argument("--driver_incrementor", type=int, default=0, help="first frame index of the driver sequence")
    parser.add_argument("--driver_neutral", type=int, default=0, help="neutral frame index of the driver")
    parser.add_argument("--my_neutral", type=int, default=0, help="neutral frame index of the avatar")
    parser.add_argument("--driving_restart", type=int, default=0, help="restart index of the driving sequence")
    parser.add_argument("--driving_folder", type=str, default="driving", help="output folder for re-enactment results")

    # training
    parser.add_argument("--iters", type=int, help="training iterations")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--ckpt", type=str, default="latest")
    parser.add_argument("--num_rays", type=int, help="number of rays sampled per image for each training step")
    parser.add_argument("--cuda_ray", action="store_true", help="use CUDA raymarching instead of pytorch")
    parser.add_argument(
        "--max_steps", type=int, default=1024, help="max num steps sampled per ray (only valid with --cuda_ray)"
    )
    parser.add_argument("--num_steps", type=int, help="num steps sampled per ray (only valid without --cuda_ray)")
    parser.add_argument(
        "--upsample_steps", type=int, default=0, help="num steps up-sampled per ray (only valid without --cuda_ray)"
    )
    parser.add_argument(
        "--update_extra_interval",
        type=int,
        default=16,
        help="iter interval to update extra status (only valid with --cuda_ray)",
    )
    parser.add_argument(
        "--max_ray_batch", type=int, help="batch size of rays at inference to avoid OOM (only valid without --cuda_ray)"
    )
    parser.add_argument(
        "--gpu_mem_fraction",
        type=float,
        default=0.8,
        help="max fraction of the GPU memory used by PyTorch; the BVH library allocates outside of PyTorch",
    )
    parser.add_argument("--fp16", action="store_true", help="use amp mixed precision training")

    # scene
    parser.add_argument("--color_space", type=str, default="srgb", help="color space, supports (linear, srgb)")
    parser.add_argument("--preload", action="store_true", help="preload all data into GPU (faster, more GPU memory)")
    parser.add_argument("--bound", type=float, default=2, help="scene is bounded in box[-bound, bound]^3")
    parser.add_argument("--scale", type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument("--offset", type=float, nargs="*", default=[0, 0, 0], help="offset of camera location")
    parser.add_argument("--dt_gamma", type=float, help="dt_gamma (>=0) for adaptive ray marching, 0 disables it")
    parser.add_argument("--min_near", type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument("--density_thresh", type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument(
        "--bg_radius", type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)"
    )
    parser.add_argument("--view", type=str, default="results", help="name of the output folder for rendered views")

    # experimental (kept for compatibility with the torch-ngp trainer)
    parser.add_argument("--error_map", action="store_true", help="use error map to sample rays")
    parser.add_argument("--clip_text", type=str, default="", help="text input for CLIP guidance")
    parser.add_argument(
        "--rand_pose",
        type=int,
        default=-1,
        help="<0: no random pose, =0: only random poses, >0: one random pose every N known poses",
    )

    parser.set_defaults(**(TRAIN_DEFAULTS if stage == "train" else RENDER_DEFAULTS))
    return parser


def parse_args(stage):
    opt = get_parser(stage).parse_args()
    if opt.O:
        opt.fp16 = opt.cuda_ray = opt.preload = True
    return opt
