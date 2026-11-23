"""Per-frame FLAME expression and jaw pose with DECA (https://github.com/yfeng95/DECA).

Writes, for every frame <name>.png of <IMAGES_DIR>/<subject>/Images/:
    <DECA_RESULTS><subject>/<name>_pose.npy         jaw pose, shape (1, 3)   (DECA pose[:, 3:])
    <DECA_RESULTS><subject>/<name>_expressions.npy  expression, shape (1, 50)
They initialise and regularise the expression / jaw tracking in tracking/tracker.py and are the
expression input of render.py (incl. re-enactment with --drive).

Run it in the DECA environment (see the DECA README); DECA itself is located through paths.DECA_DIR:
    python preprocessing/run_deca.py --subject <subject>
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (paths.py)
from paths import DECA_DIR, DECA_RESULTS, IMAGES_DIR


def _face_alignment_compat():
    """face-alignment >= 1.3.5 renamed LandmarksType._2D/_3D to TWO_D/THREE_D; accept both."""
    import face_alignment

    lt = face_alignment.LandmarksType
    for old, new in (("_2D", "TWO_D"), ("_2halfD", "TWO_HALF_D"), ("_3D", "THREE_D")):
        if not hasattr(lt, old) and hasattr(lt, new):
            setattr(lt, old, getattr(lt, new))


def main():
    parser = argparse.ArgumentParser(description="DECA expression / jaw pose estimation")
    parser.add_argument("--subject", type=str, required=True, help="name of the subject")
    parser.add_argument("--input", type=str, default=None, help="image folder (default: <IMAGES_DIR>/<subject>/Images/)")
    parser.add_argument("--rasterizer", type=str, default="pytorch3d", choices=["pytorch3d", "standard"])
    parser.add_argument("--detector", type=str, default="fan", help="face detector used by DECA for cropping")
    parser.add_argument("--overwrite", action="store_true", help="recompute frames that already have results")
    args = parser.parse_args()

    _face_alignment_compat()
    sys.path.insert(0, DECA_DIR)
    from decalib.datasets import datasets
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    input_dir = args.input or IMAGES_DIR + args.subject + "/Images/"
    out_dir = DECA_RESULTS + args.subject
    os.makedirs(out_dir, exist_ok=True)

    data = datasets.TestData(input_dir, iscrop=True, face_detector=args.detector)
    deca_cfg.rasterizer_type = args.rasterizer
    deca_cfg.model.use_tex = False
    deca = DECA(config=deca_cfg, device="cuda")

    for i in tqdm(range(len(data))):
        name = os.path.splitext(os.path.basename(data.imagepath_list[i]))[0]
        pose_file = os.path.join(out_dir, name + "_pose.npy")
        exp_file = os.path.join(out_dir, name + "_expressions.npy")
        if not args.overwrite and os.path.exists(pose_file) and os.path.exists(exp_file):
            continue
        image = data[i]["image"].to("cuda")[None, ...]
        with torch.no_grad():
            code = deca.encode(image)
        np.save(pose_file, code["pose"][:, 3:].cpu().numpy().astype(np.float32))  # jaw pose
        np.save(exp_file, code["exp"].cpu().numpy().astype(np.float32))
    print(f"DECA results written to {out_dir}")


if __name__ == "__main__":
    main()
