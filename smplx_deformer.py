"""SMPL-X with per-vertex offsets and access to the per-vertex skinning transforms.

The paper code used a locally modified copy of the `smplx` package. This module re-creates the
three additions on top of the unmodified package (`pip install smplx`), so the SMPL-X code itself
is not redistributed:

* ``verts_d``      learnable per-vertex offset (1, V, 3) added to the template (kept at zero by
                   preprocessing/fit_smplx.py).
* ``forward2(...)`` returns ``(T, offsets, joints)``:
                   T        (B, V, 4, 4) per-vertex skinning transform  sum_k w_vk * A_k
                   offsets  (B, V, 3)    v_shaped - v_template  (shape, expression and verts_d),
                                         i.e. posed vertices = T @ (v_template + offsets)
                   joints   (B, J, 3)    joints + face landmarks, as returned by SMPLX.forward
* ``forward3(...)`` returns the posed vertices (B, V, 3).

Like the paper code, the pose-corrective blend shapes are NOT applied by default
(``pose_correctives=False``): this reproduces the original tracking results to float precision.
``forward()`` follows the same convention, so fitting, offset estimation and tracking stay consistent.

Usage: ``smplx_deformer.create(model_folder, model_type="smplx", ...)`` - same arguments as
``smplx.create`` plus an optional ``verts_d`` initial value.
"""
import os

import torch
import torch.nn as nn
import smplx
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints, vertices2landmarks
from smplx.lbs import find_dynamic_lmk_idx_and_bcoords


class SMPLXDeformer(smplx.SMPLX):
    def __init__(self, model_path, verts_d=None, pose_correctives=False, **kwargs):
        super().__init__(model_path, **kwargs)
        self.pose_correctives = pose_correctives
        if verts_d is None:
            verts_d = torch.zeros((1, self.get_num_verts(), 3), dtype=self.dtype)
        verts_d = torch.as_tensor(verts_d, dtype=self.dtype).reshape(1, -1, 3)
        self.register_parameter("verts_d", nn.Parameter(verts_d, requires_grad=True))

    # ------------------------------------------------------------------ helpers
    def _full_pose(self, global_orient=None, body_pose=None, left_hand_pose=None, right_hand_pose=None, jaw_pose=None,
                   leye_pose=None, reye_pose=None):
        global_orient = self.global_orient if global_orient is None else global_orient
        body_pose = self.body_pose if body_pose is None else body_pose
        left_hand_pose = self.left_hand_pose if left_hand_pose is None else left_hand_pose
        right_hand_pose = self.right_hand_pose if right_hand_pose is None else right_hand_pose
        jaw_pose = self.jaw_pose if jaw_pose is None else jaw_pose
        leye_pose = self.leye_pose if leye_pose is None else leye_pose
        reye_pose = self.reye_pose if reye_pose is None else reye_pose
        if self.use_pca:
            left_hand_pose = torch.einsum("bi,ij->bj", [left_hand_pose, self.left_hand_components])
            right_hand_pose = torch.einsum("bi,ij->bj", [right_hand_pose, self.right_hand_components])
        full_pose = torch.cat(
            [global_orient.reshape(-1, 1, 3), body_pose.reshape(-1, self.NUM_BODY_JOINTS, 3),
             jaw_pose.reshape(-1, 1, 3), leye_pose.reshape(-1, 1, 3), reye_pose.reshape(-1, 1, 3),
             left_hand_pose.reshape(-1, 15, 3), right_hand_pose.reshape(-1, 15, 3)],
            dim=1,
        ).reshape(-1, 165)
        return full_pose + self.pose_mean

    def _deform(self, betas=None, expression=None, verts_d=None, transl=None, **pose):
        """LBS as in smplx.lbs.lbs, but keeping the per-vertex transforms and blend offsets."""
        full_pose = self._full_pose(**pose)
        betas = self.betas if betas is None else betas
        expression = self.expression if expression is None else expression
        verts_d = self.verts_d if verts_d is None else verts_d
        B = full_pose.shape[0]
        if betas.shape[0] != B:
            betas = betas.expand(B, -1)
        if expression.shape[0] != B:
            expression = expression.expand(B, -1)

        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.expr_dirs], dim=-1)
        v_template = self.v_template.unsqueeze(0)

        v_shaped = v_template + verts_d.reshape(1, -1, 3) + blend_shapes(shape_components, shapedirs)
        J = vertices2joints(self.J_regressor, v_shaped)
        ident = torch.eye(3, dtype=self.dtype, device=full_pose.device)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(B, -1, 3, 3)
        v_posed = v_shaped
        if self.pose_correctives:
            pose_feature = (rot_mats[:, 1:, :, :] - ident).view(B, -1)
            v_posed = v_shaped + torch.matmul(pose_feature, self.posedirs).view(B, -1, 3)
        J_transformed, A = batch_rigid_transform(rot_mats, J, self.parents, dtype=self.dtype)

        W = self.lbs_weights.unsqueeze(0).expand(B, -1, -1)
        num_joints = self.J_regressor.shape[0]
        T = torch.matmul(W, A.view(B, num_joints, 16)).view(B, -1, 4, 4)
        ones = torch.ones([B, v_posed.shape[1], 1], dtype=self.dtype, device=v_posed.device)
        verts = torch.matmul(T, torch.cat([v_posed, ones], dim=2).unsqueeze(-1))[:, :, :3, 0]

        joints = self._joints(verts, J_transformed, full_pose)
        if transl is None and hasattr(self, "transl"):
            transl = self.transl
        if transl is not None:
            verts = verts + transl.unsqueeze(1)
            joints = joints + transl.unsqueeze(1)
        return T, v_posed - v_template, joints, verts

    def _joints(self, vertices, joints, full_pose):
        """Extra joints and face landmarks exactly as SMPLX.forward (without translation)."""
        B = vertices.shape[0]
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(0).expand(B, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(0).repeat(B, 1, 1)
        if self.use_face_contour:
            dyn_idx, dyn_bc = find_dynamic_lmk_idx_and_bcoords(
                vertices, full_pose, self.dynamic_lmk_faces_idx, self.dynamic_lmk_bary_coords,
                self.neck_kin_chain, pose2rot=True,
            )
            lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_idx], 1)
            lmk_bary_coords = torch.cat([lmk_bary_coords.expand(B, -1, -1), dyn_bc], 1)
        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        joints = self.vertex_joint_selector(vertices, joints)
        joints = torch.cat([joints, landmarks], dim=1)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints=joints, vertices=vertices)
        return joints

    # ------------------------------------------------------------------ public API
    def forward(self, betas=None, global_orient=None, body_pose=None, left_hand_pose=None, right_hand_pose=None,
                transl=None, expression=None, jaw_pose=None, leye_pose=None, reye_pose=None, verts_d=None,
                **kwargs):
        _, _, joints, verts = self._deform(
            betas=betas, expression=expression, verts_d=verts_d, transl=transl,
            global_orient=global_orient, body_pose=body_pose, left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose, jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose,
        )
        return smplx.utils.SMPLXOutput(vertices=verts, joints=joints, betas=betas, expression=expression,
                                       global_orient=global_orient, body_pose=body_pose, jaw_pose=jaw_pose)

    def forward2(self, transl=None, **kwargs):
        """(T, offsets, joints); `transl` is applied to the joints only (T excludes translation)."""
        T, offsets, joints, _ = self._deform(transl=transl, **kwargs)
        return T, offsets, joints

    def forward3(self, **kwargs):
        """Posed vertices (B, V, 3)."""
        return self._deform(**kwargs)[3]


def create(model_path, model_type="smplx", **kwargs):
    """Drop-in replacement for smplx.create(..., model_type="smplx") returning an SMPLXDeformer."""
    assert model_type == "smplx", "only SMPL-X is supported"
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "smplx")
    return SMPLXDeformer(model_path, **kwargs)
