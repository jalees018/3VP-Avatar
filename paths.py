"""Central place for all file-system locations used by the 3VP-Avatar pipeline.

Every directory can be overridden with an environment variable, e.g.

    export VP_DATA_DIR=/data/3vp/data
    export VP_DECA_DIR=/opt/DECA

All constants end with a path separator, so they can be used as ``DIR + name``.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _dir(env, default):
    return os.path.join(os.path.abspath(os.environ.get(env, default)), "")


# Work directory: per-subject workspaces (<workspace>/meshes, <workspace>/reg_results, checkpoints)
# and the per-subject segment files (<subject>.json).
WORK_DIR = _dir("VP_WORK_DIR", ROOT)
IMAGES_DIR = _dir("VP_IMAGES_DIR", os.path.join(ROOT, "Images"))
DATA_DIR = _dir("VP_DATA_DIR", os.path.join(ROOT, "data"))
# Output of the template-based tracker: <TRACKER_DIR>/<exp><subject>/
TRACKER_DIR = _dir("VP_TRACKER_DIR", os.path.join(ROOT, "Tracker"))
# DECA (https://github.com/yfeng95/DECA) results: <DECA_DIR>/TestSamples/examples/results_<subject>/
DECA_DIR = _dir("VP_DECA_DIR", os.path.join(ROOT, "external", "DECA"))
DECA_RESULTS = DECA_DIR + "TestSamples/examples/results_"
# SMPL-X model files (https://smpl-x.is.tue.mpg.de), the folder that contains smplx/SMPLX_NEUTRAL.npz
SMPLX_MODEL_DIR = _dir("VP_SMPLX_DIR", os.path.join(ROOT, "models"))
# Template meshes / masks / adjacency files shipped with (or downloaded for) the repository.
ASSETS_DIR = _dir("VP_ASSETS_DIR", os.path.join(ROOT, "assets"))
