#!/bin/bash
# Full 3VP-Avatar pipeline for one subject (mirrors the configuration used for the paper).
#
#   bash run_pipeline.sh <subject> <num_train_images> [workspace]
#
# Directory locations are configured in paths.py (override with VP_* environment variables).
# Steps can be skipped with e.g. SKIP="1 2 3" bash run_pipeline.sh ...
# Steps 1 and 7 need the DECA environment: DECA_PYTHON=/path/to/envs/3vp-deca/bin/python bash run_pipeline.sh ...
set -euo pipefail

subject=${1:?usage: run_pipeline.sh <subject> <num_train_images> [workspace]}
num_images=${2:?number of training images}
workspace=${3:-${subject}_ws}
exp=Results_                                   # tracking experiment name

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"
eval "$(python - <<'EOF'
import paths
for k in ("WORK_DIR", "IMAGES_DIR", "DATA_DIR", "SMPLX_MODEL_DIR", "ASSETS_DIR", "DECA_DIR"):
    print(f'{k}="{getattr(paths, k)}"')
EOF
)"
datadir=${DATA_DIR}${subject}
keypoints=${IMAGES_DIR}${subject}/Keypoints/
ws=${WORK_DIR}${workspace}/
SKIP=${SKIP:-}
DECA_PYTHON=${DECA_PYTHON:-python}              # python of the DECA environment (steps 1 and 7)
run() { local n=$1; shift; if [[ " $SKIP " == *" $n "* ]]; then echo "-- skipping step $n"; else echo "== [$n] $*"; "$@"; fi; }

# Inputs (see README, "Data"): cameras + textured canonical mesh of the static 360° turn from Agisoft Metashape
#   ${DATA_DIR}${subject}/{transforms_train.json, trajectory.json, images/},  ${ws}meshes/${subject}_111.obj (+ .jpg)

# 0. (external) portrait matting of the frames -> ${IMAGES_DIR}${subject}/Images/_results
#    we used U^2-Net portrait matting (https://github.com/axinc-ai/ailia-models/tree/master/background_removal/u2net-portrait-matting)

# 1. facial keypoints (DECA environment)
run 1 "$DECA_PYTHON" preprocessing/detect_keypoints.py --subject "$subject"

# 2. normalise the canonical (textured) mesh  ${ws}meshes/${subject}_111.obj -> ${ws}meshes/${subject}.obj
run 2 python preprocessing/process.py --base_file "$datadir" --keypoints_file "$keypoints" --Mesh_Path "$ws" \
    --subject "$subject" --export_mesh --num_imgs "$num_images"

# 3. 3D landmarks on the canonical mesh, metric scale normalisation, landmarks again
run 3 python preprocessing/landmarks.py --base_file "$datadir" --keypoints_file "$keypoints" --Mesh_Path "$ws" --subject "$subject"
#    (step 4 rescales ${ws}meshes/${subject}.obj and the cameras in place; the originals are kept as
#     *_unnormalised.obj / *.json.orig and re-used as input, so the step can be re-run safely)
run 4 python preprocessing/fit_smplx.py --stage normalise --model-folder "$SMPLX_MODEL_DIR" --subject "$subject" \
    --mask_folder "$ASSETS_DIR" --Mesh_Path "${ws}meshes/" --Landmarks "${ws}Landmarks_OpenPose.npy" \
    --key_points_file "$keypoints"
run 5 python preprocessing/landmarks.py --base_file "$datadir" --keypoints_file "$keypoints" --Mesh_Path "$ws" --subject "$subject"

# 6. SMPL-X registration of the canonical mesh
run 6 python preprocessing/fit_smplx.py --stage register --model-folder "$SMPLX_MODEL_DIR" --subject "$subject" \
    --output "${ws}reg_results/" --mask_folder "$ASSETS_DIR" --Mesh_Path "${ws}meshes/" \
    --Landmarks "${ws}Landmarks_OpenPose.npy" --key_points_file "$keypoints"

# 7. DECA expressions / jaw pose per frame (run in the DECA environment, see README)
run 7 "$DECA_PYTHON" preprocessing/run_deca.py --subject "$subject"

# 8. template-based tracking of torso, head and expressions
run 8 python tracking/tracker.py --outdir . --canonical_mesh_path "${ws}meshes/" --smplx_meshes_path "$ws" \
    --subject "$subject" --input_images_path "$IMAGES_DIR" --base_file "$datadir" --width 512 --height 512 --exp "$exp"

# 9. post-tracking: ray-sampling probabilities, error maps, smoothing, depth maps
run 9 python preprocessing/process.py --base_file "$datadir" --keypoints_file "$keypoints" --Mesh_Path "$ws" \
    --subject "$subject" --num_imgs "$num_images" --exp_track "$exp"
run 10 python postprocessing/error_maps.py --subject "$subject" --workspace "$workspace" --exp_track "$exp" --num_imgs "$num_images"
run 11 python postprocessing/smoothening.py --subject "$subject" --exp "$workspace" --exp_track "$exp"
run 12 python postprocessing/render_depth_maps.py --subject "$subject" --workspace "$workspace" --exp_track "$exp" --train_imgs "$num_images"

# 13. training: stage 1 (NeRF + deformation correctives), stage 2 (mapping network)
TRAIN_ARGS=(--workspace "$workspace" --subject "$subject" --scale 1.0 --bound 1.20 --dt_gamma 0.0 --bg_radius 2.0
            --num_steps 700 --lr 0.0025 --train_imgs "$num_images" --expr "$exp")
# the BVH library exits with status 0 on CUDA errors, so check the marker written after the last epoch
# (an interrupted stage 1 resumes from its last checkpoint when it is started again)
train() { local stage=$1; shift; python train.py "$@" && [[ -f "$workspace/stage$stage.done" ]] || { echo "stage $stage did not finish" >&2; exit 1; }; }
run 13 train 1 "$datadir" "${TRAIN_ARGS[@]}"
run 14 train 2 "$datadir" "${TRAIN_ARGS[@]}" --trainMapping --modes Train_MM

# 15. rendering (novel views of the held-out frames)
run 15 python render.py "$datadir" "${TRAIN_ARGS[@]}" --trainMapping --modes Train_MM
