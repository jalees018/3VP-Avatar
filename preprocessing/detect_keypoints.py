"""Detect 3D facial landmarks (face-alignment, https://github.com/1adrianb/face-alignment).

Input : <IMAGES_DIR>/<subject>/Images/*.png
Output: <IMAGES_DIR>/<subject>/Images2/*.png       (512x512 copies, used as keys of the keypoint dict)
        <IMAGES_DIR>/<subject>/Keypoints/keypoints.pkl
"""
import argparse
import os
import pickle
import shutil
import sys

import face_alignment
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root (paths.py)
from paths import IMAGES_DIR

RESOLUTION = 512


def _face_alignment_compat():
    """face-alignment >= 1.3.5 renamed LandmarksType._2D/_3D to TWO_D/THREE_D; accept both."""
    import face_alignment

    lt = face_alignment.LandmarksType
    for old, new in (("_2D", "TWO_D"), ("_2halfD", "TWO_HALF_D"), ("_3D", "THREE_D")):
        if not hasattr(lt, old) and hasattr(lt, new):
            setattr(lt, old, getattr(lt, new))


def resize_images(folder, size=RESOLUTION):
    for item in os.listdir(folder):
        path = os.path.join(folder, item)
        if os.path.isfile(path):
            Image.open(path).resize((size, size), Image.LANCZOS).save(path, "PNG", quality=90)


def main():
    parser = argparse.ArgumentParser(description="Facial keypoint detection")
    parser.add_argument("--subject", type=str, required=True, help="name of the subject")
    args = parser.parse_args()

    src_dir = IMAGES_DIR + args.subject + "/Images/"
    dest_dir = IMAGES_DIR + args.subject + "/Images2/"
    out_dir = IMAGES_DIR + args.subject + "/Keypoints"

    if not os.path.exists(dest_dir):
        shutil.copytree(src_dir, dest_dir)
    resize_images(dest_dir)

    _face_alignment_compat()
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, face_detector="sfd", flip_input=False, device="cuda:0"
    )
    preds = fa.get_landmarks_from_directory(dest_dir)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + "/keypoints.pkl", "wb") as f:
        pickle.dump(preds, f)
    print("Finished!")


if __name__ == "__main__":
    main()
