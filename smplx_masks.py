"""Per-vertex masks on the SMPL-X topology (10475 vertices).

The masks were originally stored as vertex colours of SMPL-X meshes. Only the colours (and our own
19022-face topology, ``canonical_template.npy``) are shipped in ``assets/smplx_masks.npz``; no SMPL-X
geometry is included. ``load_mask_mesh`` returns an object with the attributes the code reads:

* ``.visual.vertex_colors``  (N, 4) uint8 RGBA             -- for the former ``*.obj`` masks
* ``.vertices``              (N, 7) [0, 0, 0, r, g, b, a]  -- for the former ``*.off`` masks
* ``.faces``                 (F, 3) canonical topology     -- for the former ``*.off`` masks
"""

import os
from types import SimpleNamespace

import numpy as np

from paths import ASSETS_DIR

_CACHE = {}


def _archive(folder):
    path = os.path.join(folder, "smplx_masks.npz")
    if path not in _CACHE:
        _CACHE[path] = dict(np.load(path))
    return _CACHE[path]


def load_mask_mesh(name, folder=None):
    """Stand-in for ``trimesh.load(<folder>/<name>, process=False)`` of a mask mesh, e.g. ``"SMPL-X22.obj"``."""
    folder = folder or ASSETS_DIR
    data = _archive(folder)
    if name + "/vertex_colors" in data:
        colors = data[name + "/vertex_colors"]
        return SimpleNamespace(visual=SimpleNamespace(vertex_colors=colors))
    if name + "/columns" in data:
        cols = data[name + "/columns"]
        vertices = np.concatenate([np.zeros((len(cols), 3)), cols], axis=1)
        faces = np.load(os.path.join(folder, "canonical_template.npy"))
        return SimpleNamespace(vertices=vertices, faces=faces)
    raise KeyError(f"unknown mask '{name}' in {folder}smplx_masks.npz")
