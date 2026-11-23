import os
import glob
import pickle as pkl
import tqdm
import random
import tensorboardX

import numpy as np

import time

from .perceptual_loss import ResNetLOSS
import cv2
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.optim as optim

import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

from .NN import load_model
from .projection.map import SurfaceAlignedConverter
from .projection.map_utils import load_model as load_model2
from pytorch3d.structures import Meshes  # , Pointclouds
from paths import ASSETS_DIR, DATA_DIR, DECA_RESULTS, IMAGES_DIR, SMPLX_MODEL_DIR, TRACKER_DIR, WORK_DIR
from smplx_masks import load_mask_mesh

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
import json

pose_111 = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
pose_1111 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# """ from .projection.map3 import project_position
# def warp_position(triangles, trisc, xyzs, tryii=None):


def backproject(depth, focalLengthx, focalLengthy, center_x, center_y, pose, scaling_factor=1.0):
    color_points = []
    a = np.zeros((depth.shape[0], depth.shape[1], 3))
    for v in range(depth.shape[1]):
        for u in range(depth.shape[0]):
            z = depth[u, v] / scaling_factor
            # if z == 0:   # here np.where can be used prior to the loop
            x = (u - center_x) * z / focalLengthx  # u and v are needed, no clue how to do this
            y = (v - center_y) * z / focalLengthy

            a[u, v, 0] = y
            a[u, v, 1] = x
            a[u, v, 2] = z

            a[u, v, :] = np.matmul(pose_111[0:3, 0:3], a[u, v, :])
            a[u, v, :] = np.matmul(pose[0:3, 0:3], a[u, v, :]) + pose[0:3, 3]

    return a


def dist_to_rgb_jet(errors, min_dist=0.0, max_dist=1.0):

    norm = mpl.colors.Normalize(vmin=min_dist, vmax=max_dist)
    cmap = cm.get_cmap(name="jet")
    colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    return colormapper.to_rgba(errors)


def dist_to_rgb(errors):
    h, w, d = errors.shape
    scale = 2.0
    errors = np.clip(0, scale, errors)
    errors = errors.reshape(h * w)
    heat = dist_to_rgb_jet(errors, 0, scale)[:, 0:3]
    heat = heat.reshape(h, w, 3)
    heat = np.minimum(np.maximum(heat * 255, 0), 255).astype(np.uint8)
    return heat


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x**0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
def get_rays_patches(poses, intrinsics, H, W, N=-1, error_map=None):
    """get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    rand_xx = torch.randint(0, W - 32, size=[1], device=device)  # may duplicate
    rand_yy = torch.randint(0, H - 32, size=[1], device=device)  # may duplicate
    i, j = custom_meshgrid(
        torch.linspace(rand_xx[0], rand_xx[0] + int(np.sqrt(N)) - 1, int(np.sqrt(N)), device=device),
        torch.linspace(rand_yy[0], rand_yy[0] + int(np.sqrt(N)) - 1, int(np.sqrt(N)), device=device),
    )
    i = i.t().reshape([1, N]).expand([B, N]) + 0.5
    j = j.t().reshape([1, N]).expand([B, N]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])

        else:
            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), 1, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // H, inds_coarse % H  # `//` will throw a warning in torch 1.10... anyway.
            inds_x = (inds_x + torch.rand(B, 1, device=device)).long().clamp(max=H - 1 - 32)
            inds_y = (inds_y + torch.rand(B, 1, device=device)).long().clamp(max=W - 1 - 32)
            i, j = custom_meshgrid(
                torch.linspace(inds_x[0, 0], inds_x[0, 0] + int(np.sqrt(N)) - 1, int(np.sqrt(N)), device=device),
                torch.linspace(inds_y[0, 0], inds_y[0, 0] + int(np.sqrt(N)) - 1, int(np.sqrt(N)), device=device),
            )
            i = i.t().reshape([1, N]).expand([B, N]) + 0.5
            j = j.t().reshape([1, N]).expand([B, N]) + 0.5

        results["inds"] = torch.stack((i, j), dim=-1)  # inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    return results


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    """get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device), torch.linspace(0, H - 1, H, device=device))
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H * W)

        if error_map is None:
            inds = torch.randint(0, H * W, size=[N], device=device)  # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False)  # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128  # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results["inds_coarse"] = inds_coarse  # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results["inds"] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results["rays_o"] = rays_o
    results["rays_d"] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths)  # [B, N, 3] or [B, H, W, 3], range[0, 1]

        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f"PSNR = {self.measure():.6f}"


class Trainer(object):
    def __init__(
        self,
        name,  # name of this experiment
        opt,  # extra conf
        model,  # network
        criterion=None,  # loss function, if None, assume inline implementation in train_step
        optimizer=None,  # optimizer
        ema_decay=None,  # if use EMA, set the decay
        lr_scheduler=None,  # scheduler
        metrics=[],  # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
        local_rank=0,  # which GPU am I
        world_size=1,  # total num of GPUs
        device=None,  # device to use, usually setting to None is OK. (auto choose device)
        mute=False,  # whether to mute all print
        fp16=False,  # amp optimize level
        eval_interval=1,  # eval once every $ epoch
        max_keep_ckpt=2,  # max num of saved ckpts in disk
        workspace="workspace",  # workspace to save logs & ckpts
        best_mode="min",  # the smaller/larger result, the better
        use_loss_as_metric=True,  # use loss as the first metric
        report_metric_at_train=False,  # also report metrics at training
        use_checkpoint="latest",  # which ckpt to use at init time
        use_tensorboardX=True,  # whether to use tensorboard for logging
        scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
        model2=None,
    ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.model2 = model2
        self.device = (
            device if device is not None else torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        )
        self.patch_size = int(np.sqrt(self.opt.num_rays))
        print("Patch Size:", self.patch_size)
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = "min"

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            self.best_path_cond = f"{self.ckpt_path}/{self.name}_cond.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}'
        )
        self.log(f"[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                if self.opt.modes == "single":
                    self.load_checkpoint()
                elif self.opt.modes == "Double_train":
                    self.load_checkpoint_cond2()
                elif self.opt.modes == "Train_MM":
                    print("Training Mapping Network!")
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # clip loss prepare
        if opt.rand_pose >= 0:  # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss

            self.clip_loss = CLIPLoss(self.device)
            self.clip_loss.prepare_text([self.opt.clip_text])  # only support one text prompt now...

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def train_step_correctives(
        self,
        data,
        model2,
        rays_o,
        rays_d,
        converter,
        ver,
        face_id,
        local_coordinates_meshes,
        triangles_meshes,
        z_val,
        adj,
        iddd,
        trisc,
        trisd,
        nears=None,
        fars=None,
        msk_depth=None,
        face_msk=None,
    ):

        images = data["images2"]  # [B, N, 3/4]

        B, N, C = images.shape

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            bg_color = torch.rand_like(images[..., :3])  # [N, 3], pixel-wise random.

        if msk_depth == None:
            outputs = self.model.render_cond(
                rays_o,
                rays_d,
                model2,
                converter,
                ver,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_val,
                adj,
                data,
                iddd,
                trisc,
                trisd,
                staged=False,
                bg_color=bg_color,
                perturb=True,
                force_all_rays=True,
                **vars(self.opt),
            )
        else:
            outputs = self.model.render_cond_sample_depth(
                rays_o,
                rays_d,
                model2,
                converter,
                ver,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_val,
                adj,
                data,
                iddd,
                trisc,
                trisd,
                nears,
                fars,
                msk_depth,
                face_msk,
                self.epoch,
                staged=False,
                bg_color=bg_color,
                perturb=True,
                force_all_rays=True,
                **vars(self.opt),
            )
        pred_rgb = outputs["image"]

        if self.opt.color_space == "linear":
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        ########################################################################################################################

        pred_depth2 = outputs["depth2"]
        pred_weights_sum = outputs["depth3"] + 1e-5

        losss = (self.criterion(pred_rgb, gt_rgb)).mean(
            -1
        )  # [B, N, 3] --> [B, N] #loss = self.criterion(pred_rgb*mskk_bkgrnd, gt_rgb*mskk_bkgrnd).mean(-1)  # [B, N, 3] --> [B, N]

        # special case for CCNeRF's rank-residual training
        if len(losss.shape) == 3:  # [K, B, N]
            losss = losss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data["index"]  # [B]
            inds = data["inds_coarse"]  # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index]  # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:

            error = losss.detach().to(error_map.device)  # [B, N], already in [0, 1]

            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map
        pred_rgb1 = pred_rgb.reshape(-1, self.patch_size, self.patch_size, 3).permute(0, 3, 1, 2)
        predicted_images = -1.0 + 2 * pred_rgb1
        gt_rgb1 = gt_rgb.reshape(-1, self.patch_size, self.patch_size, 3).permute(0, 3, 1, 2)
        target_images = -1.0 + 2 * gt_rgb1
        perc_loss = self.perceptual_loss(predicted_images, target_images).mean()

        msk_new = pred_weights_sum != 0.0
        depth_gt = self.depth.cuda()
        beta_loss = -0.1 * (pred_weights_sum * torch.log(pred_weights_sum)).mean()
        wts_reg = 10.0

        if self.epoch <= 39:
            if msk_new.sum() != 0.0:  # outputs['deform'] is not None:
                loss = losss.mean() + beta_loss
            else:
                loss = losss.mean() + beta_loss
        else:
            if msk_new.sum() != 0.0:  # outputs['deform'] is not None:
                loss = (
                    0.35 * perc_loss
                    + 0.0 * losss.mean()
                    + 0.035 * torch.mean(abs(self.m_pool(gt_rgb1) - self.m_pool(pred_rgb1)))
                    + beta_loss
                )
            else:
                loss = (
                    0.35 * perc_loss
                    + 0.0 * losss.mean()
                    + 0.035 * torch.mean(abs(self.m_pool(gt_rgb1) - self.m_pool(pred_rgb1)))
                    + beta_loss
                )
        # """ perc_loss = self.perceptual_loss(predicted_images, target_images).mean() if \
        return pred_rgb, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step2(
        self,
        data,
        rays_o,
        rays_d,
        converter,
        ver,
        face_id,
        local_coordinates_meshes,
        triangles_meshes,
        z_val=None,
        adj=None,
        iddd=None,
        model2=None,
        abcdefghhhh=None,
        trisc=None,
        trisd=None,
        nears=None,
        fars=None,
        msk_depth=None,
        face_msk=None,
        epoch=10,
        bg_color=None,
        perturb=False,
    ):

        H, W = data["H"], data["W"]

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        # if model2 is None:
        if msk_depth == None:
            outputs = self.model.render_cond_test(
                rays_o,
                rays_d,
                model2,
                converter,
                ver,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_val,
                adj,
                iddd,
                abcdefghhhh,
                trisc,
                trisd,
                staged=True,
                bg_color=bg_color,
                perturb=perturb,
                **vars(self.opt),
            )
        else:
            outputs = self.model.render_cond_test_depth_sample(
                rays_o,
                rays_d,
                model2,
                converter,
                ver,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_val,
                adj,
                iddd,
                abcdefghhhh,
                trisc,
                trisd,
                nears,
                fars,
                msk_depth,
                face_msk,
                epoch,
                staged=True,
                bg_color=bg_color,
                perturb=perturb,
                **vars(self.opt),
            )
        pred_rgb = outputs["image"].reshape(-1, H, W, 3)
        pred_depth = outputs["depth"].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def trainMapping(self, train_loader, valid_loader, max_epochs, model2=None):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch_correctivesMappingNetwork(train_loader, model2)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint_cond_Mapping(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.save_checkpoint_cond_Mapping(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def train_correctives(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map

        model2 = None

        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch_correctivesSampleDepth(train_loader, model2)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint_cond(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.save_checkpoint_cond(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def animateSampleDepth(self, loader, model2=None, model3=None, save_path=None, name=None):

        if model2 is not None:
            model2 = model2.to(self.device)
            model2 = self.load_checkpoint_cond3(model2)
            model2.eval()
        if save_path is None:
            save_path = os.path.join(self.workspace, self.opt.view)

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(
            total=len(loader) * loader.batch_size,
            bar_format="{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        if model3 is not None:
            self.model = self.load_checkpoint_cond3_Mapping(model3.to(self.device))

        self.model.eval()
        with torch.no_grad():
            M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

            path1 = TRACKER_DIR + self.opt.expr + self.opt.subject + "/"
            obj_path1 = WORK_DIR + self.opt.workspace + "/reg_results/" + self.opt.subject + "_s.obj"
            obj_path2 = WORK_DIR + self.opt.workspace + "/meshes/" + self.opt.subject + ".obj"
            facs = np.load(ASSETS_DIR + "canonical_template.npy")
            verts1, faces, faces_idx, verts_uvs, faces_t, device = load_model2(obj_path1)
            faces_idx = torch.tensor(facs).cuda()
            converter = SurfaceAlignedConverter(verts1, faces, faces_idx, verts_uvs, faces_t, device)
            abcccdef = load_mask_mesh("SMPL-X22_new_888.off")
            iddxxx = abcccdef.vertices[:, 5] == 255
            mask = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
            mask[iddxxx] = True
            msk_face = abcccdef.faces
            scale = 1.0
            face_id = torch.tensor(mask[msk_face].all(axis=1)).cuda()
            verts1, faces, faces_idx = load_model(obj_path1)
            faces_idx = torch.tensor(facs).cuda()
            verts = verts1.unsqueeze(0).cuda()

            # Render the scene
            W = 512  # 1000
            H = 512  # 800
            faces = (faces_idx[face_id, :])[None, ...].repeat(len(verts), 1, 1).cuda()
            meshes = Meshes(verts=(verts) * scale, faces=faces)
            triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]
            local_coordinates_meshes = None

            faces1 = faces
            ####################################################################################################################################
            verts2 = verts1.unsqueeze(0).cuda()
            iddxxx2 = abcccdef.vertices[:, 5] == 0
            mask2 = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
            mask2[iddxxx2] = True
            msk_face2 = abcccdef.faces
            face_id2 = torch.tensor(mask2[msk_face2].all(axis=1)).cuda()
            faces_idx2 = torch.tensor(facs).cuda()
            faces2 = (faces_idx2[face_id2, :])[None, ...].repeat(len(verts2), 1, 1).cuda()
            ####################################################################################################################################

            faces = torch.cat((faces1, faces2), dim=1)

            meshesss = Meshes(verts=verts, faces=faces)
            verts_packed = meshesss.verts_packed()
            faces_packed = meshesss.faces_packed()
            trisc = verts_packed[faces_packed]
            with open(WORK_DIR + self.opt.subject + ".json") as ff:
                fr = json.load(ff)
            if self.opt.trainMapping:
                restart = self.opt.train_imgs
            else:
                restart = fr[11]
            i = restart
            num_imgs = 1251 * 4

            num_imgss = restart + 1000
            trisdd = []
            for iit in tqdm.tqdm(range(restart, num_imgss - 100)):
                if self.opt.drive:
                    verts11, _, _ = load_model(
                        TRACKER_DIR + "Meshes/" + self.opt.driving_folder + "/With_Mean/Mesh_" + str(iit) + ".obj"
                    )
                else:
                    verts11, _, _ = load_model(path1 + "result_final_Real_tt_" + str(iit) + "_s.obj")
                meshesss = Meshes(verts=verts11.unsqueeze(0), faces=faces.cpu())
                verts_packed = meshesss.verts_packed()
                faces_packed = meshesss.faces_packed()
                trisdd.append(verts_packed[faces_packed])
            trisdd = torch.stack(trisdd)
            iddxxx0 = (abcccdef.vertices[:, 4] == 0) & (abcccdef.vertices[:, 5] == 255)
            mask0 = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
            mask0[iddxxx0] = True
            msk_face0 = faces
            scale = 1.0
            face_id0 = torch.tensor(mask0[msk_face0[0].cpu().numpy()].all(axis=1)).cuda()
            # if True:
            iddxxx0x = ((abcccdef.vertices[:, 4] == 0) & (abcccdef.vertices[:, 5] == 0)) & (
                abcccdef.vertices[:, 3] == 0
            )
            mask0x = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
            mask0x[iddxxx0x] = True
            scale = 1.0
            face_id0x = torch.tensor(mask0x[msk_face0[0].cpu().numpy()].all(axis=1)).cuda()
            new_adj = torch.tensor(np.load(ASSETS_DIR + "Adjacency_SMPL-X_Edges_3333.npy")).cuda()
            new_adj2 = torch.zeros((faces2.shape[1], 19)).cuda()
            new_adj = torch.cat((new_adj, new_adj2), dim=0)
            local_coordinates_meshes = (face_id0x | face_id0).tolist()
            for iik in range(faces1.shape[1], faces.shape[1]):
                new_adj[iik, 18] = iik + 1

            lsttte = os.listdir(IMAGES_DIR + self.opt.subject + "/Images/_results/")
            incrementor_111 = int(min([int(((iii).split("/")[-1])[:-4]) for iii in lsttte]))  # 1560
            print(incrementor_111)
            if self.opt.notrainposes:
                poses = []
                poses2 = []
                num_b = 1000  # 838
                from .provider import nerf_matrix_to_ngp, camera_path_pos_horizontals

                ptss = camera_path_pos_horizontals(-3.9, num_b, 1.29)
                for kk1 in range(0, num_b):
                    poses3 = np.transpose(ptss[kk1])
                    posee = np.zeros_like(poses3)
                    posee[2, 0:4] = poses3[1, 0:4]
                    posee[1, 0:4] = poses3[0, 0:4]
                    posee[0, 0:4] = poses3[2, 0:4]
                    pose2 = posee[0:3, 0:4].astype(np.float32)
                    pose = nerf_matrix_to_ngp(
                        posee[0:3, 0:4].astype(np.float32), scale=self.opt.scale, offset=self.opt.offset
                    )
                    poses.append(pose)
                    poses2.append(pose2)
                poses = torch.from_numpy(np.stack(poses, axis=0))  # [N, 4, 4]
                poses2 = torch.from_numpy(np.stack(poses2, axis=0))  # [N, 4, 4]
            flags = pyrender.RenderFlags.SKIP_CULL_FACES
            jj = self.opt.driving_restart
            for ii in range(0, num_imgs):  # , data in enumerate(loader):

                ver = None
                if self.opt.drive:
                    subject2 = self.opt.driver
                    incrementor_222 = self.opt.driver_incrementor  # 1100 #1412 #1100 #1412 #incrementor_111
                    neut = self.opt.driver_neutral  # 751 #19
                    my_neut = self.opt.my_neutral  # 5451 #19 #8672 #751 #5451 #

                    jawpose_mean = np.load(
                        DECA_RESULTS + subject2 + "/" + str(neut + incrementor_222).zfill(4) + "_pose.npy"
                    )
                    exp_mean = np.load(
                        DECA_RESULTS + subject2 + "/" + str(neut + incrementor_222).zfill(4) + "_expressions.npy"
                    )[:, 0:13]
                    jawpose_my_mean = np.load(
                        DECA_RESULTS + self.opt.subject + "/" + str(my_neut + incrementor_111).zfill(4) + "_pose.npy"
                    )
                    exp_my_mean = np.load(
                        DECA_RESULTS
                        + self.opt.subject
                        + "/"
                        + str(my_neut + incrementor_111).zfill(4)
                        + "_expressions.npy"
                    )[:, 0:13]

                    abcdefghhhh = {}
                    abcdefghhhh["jaw_pose"] = (
                        np.load(DECA_RESULTS + subject2 + "/" + str(jj + incrementor_222).zfill(4) + "_pose.npy")
                        - jawpose_mean
                    ) + jawpose_my_mean
                    abcdefghhhh["expressions"] = (
                        np.load(
                            DECA_RESULTS + subject2 + "/" + str(jj + incrementor_222).zfill(4) + "_expressions.npy"
                        )[:, 0:13]
                        - exp_mean
                    ) + exp_my_mean
                    abcdefghhhh["jaw_posei"] = (
                        np.load(DECA_RESULTS + subject2 + "/" + str(jj + incrementor_222 - 1).zfill(4) + "_pose.npy")
                        - jawpose_mean
                    ) + jawpose_my_mean
                    abcdefghhhh["expressionsi"] = (
                        np.load(
                            DECA_RESULTS + subject2 + "/" + str(jj + incrementor_222 - 1).zfill(4) + "_expressions.npy"
                        )[:, 0:13]
                        - exp_mean
                    ) + exp_my_mean
                    abcdefghhhh["jaw_posef"] = (
                        np.load(DECA_RESULTS + subject2 + "/" + str(jj + incrementor_222 + 1).zfill(4) + "_pose.npy")
                        - jawpose_mean
                    ) + jawpose_my_mean
                    abcdefghhhh["expressionsf"] = (
                        np.load(
                            DECA_RESULTS + subject2 + "/" + str(jj + incrementor_222 + 1).zfill(4) + "_expressions.npy"
                        )[:, 0:13]
                        - exp_mean
                    ) + exp_my_mean
                    abcdefghhhh["json_file"] = fr

                    jj = jj + 1

                elif self.opt.expression_source == "tracker":
                    # tracked expression / jaw (the mapping network was trained on these), frames i-1, i, i+1
                    abcdefghhhh = {"json_file": fr}
                    for key, n in (("", i), ("i", i - 1), ("f", i + 1)):
                        with open(path1 + "result_final_Real_tt_" + str(n) + ".pkl", "rb") as fpkl:
                            tracked = pkl.load(fpkl)
                        abcdefghhhh["jaw_pose" + key] = np.asarray(tracked["jaw_pose"], dtype=np.float32).reshape(1, 3)
                        abcdefghhhh["expressions" + key] = np.asarray(
                            tracked["expressions"], dtype=np.float32
                        ).reshape(1, -1)[:, 0:13]

                else:
                    subject2 = self.opt.subject
                    abcdefghhhh = {}
                    abcdefghhhh["jaw_pose"] = np.load(
                        DECA_RESULTS + subject2 + "/" + str(i + incrementor_111).zfill(4) + "_pose.npy"
                    )
                    abcdefghhhh["expressions"] = np.load(
                        DECA_RESULTS + subject2 + "/" + str(i + incrementor_111).zfill(4) + "_expressions.npy"
                    )[:, 0:13]
                    abcdefghhhh["jaw_posei"] = np.load(
                        DECA_RESULTS + subject2 + "/" + str(i + incrementor_111 - 1).zfill(4) + "_pose.npy"
                    )
                    abcdefghhhh["expressionsi"] = np.load(
                        DECA_RESULTS + subject2 + "/" + str(i + incrementor_111 - 1).zfill(4) + "_expressions.npy"
                    )[:, 0:13]
                    abcdefghhhh["jaw_posef"] = np.load(
                        DECA_RESULTS + subject2 + "/" + str(i + incrementor_111 + 1).zfill(4) + "_pose.npy"
                    )
                    abcdefghhhh["expressionsf"] = np.load(
                        DECA_RESULTS + subject2 + "/" + str(i + incrementor_111 + 1).zfill(4) + "_expressions.npy"
                    )[:, 0:13]
                    abcdefghhhh["json_file"] = fr

                ver = None

                trisd = trisdd[i - restart].cuda()
                # update grid
                #     with torch.cuda.amp.autocast(enabled=self.fp16):
                epochh = 1

                if i == restart:
                    kk = 0
                    for iiidd2, data2 in enumerate(loader):
                        if kk == 0:
                            data = data2
                        else:
                            break
                        kk = kk + 1
                    intr = data["intrinsics"]

                    # Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
                    camera = pyrender.camera.IntrinsicsCamera(intr[0], intr[1], intr[2], intr[3])

                if not self.opt.notrainposes:
                    camera_pose111 = data["poses"].cpu().numpy()
                    camera_pose111 = np.concatenate(
                        (camera_pose111[0], np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)), axis=0
                    )
                else:
                    camera_pose111 = (poses2)[[i - restart]].cpu().numpy()
                    camera_pose111 = np.concatenate(
                        (camera_pose111[0], np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)), axis=0
                    )
                xf = np.linalg.inv(M) @ camera_pose111
                mesh_111 = trimesh.load(
                    path1 + "Results_Deformed_tt_" + str(i) + ".obj", process=False
                )
                trimesh.repair.fix_inversion(mesh_111)
                mesh_111.merge_vertices(merge_tex=True, merge_norm=True)
                mesh_1111 = trimesh.Trimesh(
                    vertices=(mesh_111.vertices) + 0.015 * mesh_111.vertex_normals, faces=mesh_111.faces
                )
                scene = pyrender.Scene(ambient_light=np.array([1.0, 1.0, 1.0, 1.0]), bg_color=np.array([0.0, 0.0, 0.0]))
                scene.add(pyrender.Mesh.from_trimesh(mesh_1111, smooth=False))
                nc = pyrender.Node(camera=camera, matrix=xf)
                scene.add_node(nc)
                r = pyrender.OffscreenRenderer(W, H)
                col, depth = r.render(scene, flags=flags)
                depth = torch.tensor(depth)
                msk_depth = depth != 0.0

                r.delete()
                nears = depth.clone()
                fars = depth.clone()
                nears[msk_depth] -= 0.05
                fars[msk_depth] += 0.80

                z_vals11 = None

                # with torch.cuda.amp.autocast(enabled=self.fp16):
                if i == restart and not self.opt.notrainposes:
                    rays_0 = data["rays_o2"]  # [B, N, 3]
                    rays_d = data["rays_d2"]  # [B, N, 3]
                elif self.opt.notrainposes:
                    poses3 = poses[[i - restart]].cuda()
                    rays2 = get_rays(poses3, intr, H, W, -1, None)
                    rays_0 = rays2["rays_o"]
                    rays_d = rays2["rays_d"]

                preds, preds_depth = self.test_step2(
                    data,
                    rays_0,
                    rays_d,
                    converter,
                    ver,
                    face_id,
                    local_coordinates_meshes,
                    triangles_meshes,
                    z_vals11,
                    new_adj,
                    i,
                    model2,
                    abcdefghhhh,
                    trisc,
                    trisd,
                    nears.reshape(-1, 1),
                    fars.reshape(-1, 1),
                    msk_depth.reshape(-1, 1),
                    None,
                    epochh,
                )

                path = os.path.join(save_path, f"{name}_{i:04d}.png")
                path_depth = os.path.join(save_path, f"{name}_{i:04d}_heat.png")

                if self.opt.color_space == "linear":
                    preds = linear_to_srgb(preds)

                pred = np.clip(preds[0].detach().cpu().numpy(), 0.0, 1.0, out=None)

                cv2.imwrite(path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                fake = pred * 255
                target = cv2.resize(
                    cv2.imread(
                        IMAGES_DIR + self.opt.subject + "/Images/" + str(i + incrementor_111).zfill(4) + ".png",
                        cv2.IMREAD_COLOR,
                    ),
                    (W, H),
                    interpolation=cv2.INTER_AREA,
                )
                errors = np.linalg.norm((target / 255.0 - fake / 255.0), axis=2, keepdims=True, ord=1)
                heat = dist_to_rgb(errors)
                cv2.imwrite(path_depth, cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))

                pbar.update(loader.batch_size)
                i = i + 1

    def train_one_epoch_correctivesMappingNetwork(self, loader, model2):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")
        if self.epoch == 1:
            if model2 is not None:
                model2 = model2.to(self.device)
                self.model2 = self.load_checkpoint_cond3(model2)
                self.model2.eval()
        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        self.local_step = 0

        for data in loader:
            indx = data["index"]
            length = data["length"]

            if indx < length:
                iddd1 = 0
            else:
                iddd1 = indx - length

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            de = torch.cat((data["expressions"], data["jaw_pose"]), dim=2)

            out = self.model(de)
            gtt = self.model2.latentxx[iddd1 : iddd1 + 1, 0:11, 0, 0]

            loss = torch.mean((out - gtt) ** 2)

            loss.backward()
            self.optimizer.step()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                # if self.report_metric_at_train:
                #     for metric in self.metrics:

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def train_one_epoch_correctivesSampleDepth(self, loader, model2):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(
                total=len(loader) * loader.batch_size,
                bar_format="{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

        self.local_step = 0

        M = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        path1 = TRACKER_DIR + self.opt.expr + self.opt.subject + "/"
        obj_path1 = WORK_DIR + self.opt.workspace + "/reg_results/" + self.opt.subject + "_s.obj"
        obj_path2 = WORK_DIR + self.opt.workspace + "/meshes/" + self.opt.subject + ".obj"
        model_folder = SMPLX_MODEL_DIR

        facs = np.load(ASSETS_DIR + "canonical_template.npy")
        verts1, faces, faces_idx, verts_uvs, faces_t, device = load_model2(obj_path1)
        faces_idx = torch.tensor(facs).cuda()
        converter = None
        abcccdef = load_mask_mesh("SMPL-X22_new_888.off")
        abcccdefm = load_mask_mesh("SMPL-X22_new_888m.off")
        iddxxx = abcccdef.vertices[:, 5] == 255
        mask = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
        mask[iddxxx] = True
        msk_face = abcccdef.faces
        scale = 1.0
        face_id = torch.tensor(mask[msk_face].all(axis=1)).cuda()
        verts1, faces, faces_idx = load_model(obj_path1)
        ####################################################################################################################################
        verts2 = verts1.unsqueeze(0).cuda()
        iddxxx2 = abcccdef.vertices[:, 5] == 0
        mask2 = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
        mask2[iddxxx2] = True
        msk_face2 = abcccdef.faces
        face_id2 = torch.tensor(mask2[msk_face2].all(axis=1)).cuda()
        faces_idx2 = torch.tensor(facs).cuda()
        faces2 = (faces_idx2[face_id2, :])[None, ...].repeat(len(verts2), 1, 1).cuda()
        ####################################################################################################################################

        faces_idx = torch.tensor(facs).cuda()
        verts = verts1.unsqueeze(0).cuda()

        # Render the scene
        W = 512  # 1000
        H = 512  # 800
        faces1 = (faces_idx[face_id, :])[None, ...].repeat(len(verts), 1, 1).cuda()
        meshes = Meshes(verts=(verts) * scale, faces=faces1)
        triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]  # [batch*13776, 3, 3]
        local_coordinates_meshes = None

        faces = torch.cat((faces1, faces2), dim=1)
        meshesss = Meshes(verts=verts, faces=faces)
        verts_packed = meshesss.verts_packed()
        faces_packed = meshesss.faces_packed()
        trisc = verts_packed[faces_packed]

        ###############################################
        msk_face0 = faces
        # if True:
        iddxxx0xm = (abcccdefm.vertices[:, 4] == 255) & (abcccdefm.vertices[:, 5] == 0)
        mask0xm = np.zeros(len(abcccdefm.vertices[:, 0:3]), dtype=bool)
        mask0xm[iddxxx0xm] = True
        scale = 1.0
        face_id0xm = torch.tensor(mask0xm[msk_face0[0].cpu().numpy()].all(axis=1)).cuda()
        iddxxx0 = (abcccdef.vertices[:, 4] == 0) & (abcccdef.vertices[:, 5] == 255)
        mask0 = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
        mask0[iddxxx0] = True
        scale = 1.0
        face_id0 = torch.tensor(mask0[msk_face0[0].cpu().numpy()].all(axis=1)).cuda()
        # if True:
        iddxxx0x = ((abcccdef.vertices[:, 4] == 0) & (abcccdef.vertices[:, 5] == 0)) & (abcccdef.vertices[:, 3] == 0)
        mask0x = np.zeros(len(abcccdef.vertices[:, 0:3]), dtype=bool)
        mask0x[iddxxx0x] = True
        scale = 1.0
        face_id0x = torch.tensor(mask0x[msk_face0[0].cpu().numpy()].all(axis=1)).cuda()
        new_adj = torch.tensor(np.load(ASSETS_DIR + "Adjacency_SMPL-X_Edges_3333.npy")).cuda()
        new_adj2 = torch.zeros((faces2.shape[1], 19)).cuda()
        new_adj = torch.cat((new_adj, new_adj2), dim=0)
        local_coordinates_meshes = (face_id0x | face_id0).tolist()
        for iik in range(faces1.shape[1], faces.shape[1]):
            new_adj[iik, 18] = iik + 1
        arr = np.array([[0.0, 0.0, 0.0, 1.0]]).astype(np.float32)
        if not hasattr(self, "trisddd"):  # first epoch of this run (also when resuming)
            num_imgs = len(
                [
                    name
                    for name in os.listdir(IMAGES_DIR + self.opt.subject + "/Images/")
                    if os.path.isfile(os.path.join(IMAGES_DIR + self.opt.subject + "/Images/", name))
                ]
            )
            trisdd = []
            self.nearsh = []
            self.farsh = []
            self.msk_depthh = []
            self.face_msk = []
            for iit in tqdm.tqdm(range(0, self.opt.train_imgs)):
                verts11, _, _ = load_model(path1 + "result_final_Real_tt_" + str(iit) + "_s.obj")
                meshesss = Meshes(verts=verts11.unsqueeze(0).cpu(), faces=faces.cpu())
                verts_packed = meshesss.verts_packed()
                faces_packed = meshesss.faces_packed()
                trisdd.append(verts_packed[faces_packed])
            self.trisddd = torch.stack(trisdd)
            self.root_path = DATA_DIR + self.opt.subject
            lstt_111 = []
            with open(os.path.join(self.root_path, f"transforms_train.json"), "r") as f:
                transform = json.load(f)
            frames = transform["frames"]
            # load intrinsics
            focal_scale = H / int(transform["h"])
            print("Focal_Scale:", focal_scale)
            if "fl_x" in transform or "fl_y" in transform:
                fl_x = (transform["fl_x"] if "fl_x" in transform else transform["fl_y"]) * focal_scale  # / downscale
                fl_y = (transform["fl_y"] if "fl_y" in transform else transform["fl_x"]) * focal_scale  # / downscale
            elif "camera_angle_x" in transform or "camera_angle_y" in transform:
                # blender, assert in radians. already downscaled since we use H/W
                fl_x = self.W / (2 * np.tan(transform["camera_angle_x"] / 2)) if "camera_angle_x" in transform else None
                fl_y = self.H / (2 * np.tan(transform["camera_angle_y"] / 2)) if "camera_angle_y" in transform else None
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
            with open(os.path.join(self.root_path, f"trajectory.json"), "r") as f:
                meta111 = json.load(f)
            for f in tqdm.tqdm(frames, desc=f"Loading data"):
                f_path = os.path.join(self.root_path, f["file_path"])
                lstt_111.append(int(f_path[-8:-4]))
            lstt_222 = list(lstt_111)
            lstt_222.sort()
            id_111 = np.where(np.array(lstt_111) == np.array(lstt_111).max())
            j = lstt_111[int(id_111[0])]

            print("Length:", len(lstt_111))
            for iit in tqdm.tqdm(range(0, len(lstt_111) + self.opt.train_imgs), desc=f"Loading Depth Maps"):
                npzfiles = np.load(IMAGES_DIR + self.opt.subject + "/Images/Depth_Maps/" + "Depth_" + str(iit) + ".npz")
                nears = torch.tensor(npzfiles["near"])
                fars = torch.tensor(npzfiles["far"])
                msk_depth = torch.tensor(npzfiles["msk_depthh"])

                self.nearsh.append(nears)
                self.farsh.append(fars)
                self.msk_depthh.append(msk_depth)

        restart = 0
        i = restart

        for data in loader:
            indx = data["index"]
            length = data["length"]
            if indx < length:
                trisd = trisc
            else:
                trisd = self.trisddd[indx - length]

            if i == restart:
                kk = 0
                for iiidd2, data2 in enumerate(loader):
                    if kk == 0:
                        data3 = data2
                    else:
                        break
                    kk = kk + 1
                intr = data3["intrinsics"]
                length = data3["length"]
                print(length)

                self.m_pool = nn.AvgPool2d(self.patch_size // 4, stride=self.patch_size // 4)
                self.perceptual_loss = ResNetLOSS().to(self.device)
                z_vals11 = None

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                # with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state(trisc.cuda(), trisd.cuda(), data, new_adj)

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            rays_0 = data["rays_o2"]  # [B, N, 3]
            rays_d = data["rays_d2"]  # [B, N, 3]
            self.inxcices = data["inxcices"]
            # pixel indices of the sampled rays; on the CPU like the depth buffers they index (required by torch >= 1.13)
            iy, ix = self.inxcices[0, :, 1].long().cpu(), self.inxcices[0, :, 0].long().cpu()

            nears = self.nearsh[indx]
            nears2 = nears[iy, ix]
            fars = self.farsh[indx]
            mskdepth = self.msk_depthh[indx]
            self.depth = nears2
            mskdepth2 = mskdepth[iy, ix].reshape(-1, 1)
            self.depth = self.depth[mskdepth[iy, ix]] + 0.09
            if i == restart:
                face_mskk = None
            i = i + 1
            if mskdepth2.sum() == 0:
                continue

            preds, truths, loss = self.train_step_correctives(
                data,
                model2,
                rays_0,
                rays_d,
                converter,
                None,
                face_id,
                local_coordinates_meshes,
                triangles_meshes,
                z_vals11,
                new_adj,
                None,
                trisc.cuda(),
                trisd.cuda(),
                nears2.reshape(-1, 1).cuda(),
                fars[iy, ix].reshape(-1, 1).cuda(),
                mskdepth2.cuda(),
                face_id0xm.tolist(),
            )
            loss.backward()
            self.optimizer.step()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}"
                    )
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def save_checkpoint_cond_Mapping(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f"{self.name}_ep_Mapping_{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
        }

        if self.model.cuda_ray:
            state["mean_count"] = self.model.mean_count
            state["mean_density"] = self.model.mean_density

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:

            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}_cond_Mapping.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if "density_grid" in state["model"]:
                        del state["model"]["density_grid"]

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path_cond)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def save_checkpoint_cond(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f"{self.name}_ep{self.epoch:04d}"

        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "stats": self.stats,
        }

        if self.model.cuda_ray:
            state["mean_count"] = self.model.mean_count
            state["mean_density"] = self.model.mean_density

        if full:
            state["optimizer"] = self.optimizer.state_dict()
            state["lr_scheduler"] = self.lr_scheduler.state_dict()
            state["scaler"] = self.scaler.state_dict()
            if self.ema is not None:
                state["ema"] = self.ema.state_dict()

        if not best:

            state["model"] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}_cond.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state["model"] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if "density_grid" in state["model"]:
                        del state["model"]["density_grid"]

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path_cond)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep[0-9]*.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict["model"], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and "ema" in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict["ema"])

        if self.model.cuda_ray:
            if "mean_count" in checkpoint_dict:
                self.model.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                self.model.mean_density = checkpoint_dict["mean_density"]

        if model_only:
            return

        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"]
        self.global_step = checkpoint_dict["global_step"]
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and "optimizer" in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and "lr_scheduler" in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict["lr_scheduler"])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and "scaler" in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict["scaler"])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")

    def load_checkpoint_cond2(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*_cond.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location="cpu")

        if "model" not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict["model"], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        #    """  if self.ema is not None and 'ema' in checkpoint_dict:

        if self.model.cuda_ray:
            if "mean_count" in checkpoint_dict:
                self.model.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                self.model.mean_density = checkpoint_dict["mean_density"]

        if model_only:
            return

        # resume after the last finished epoch
        self.stats = checkpoint_dict["stats"]
        self.epoch = checkpoint_dict["epoch"] + 1
        self.global_step = checkpoint_dict["global_step"]
        for key in ("optimizer", "lr_scheduler", "scaler", "ema"):
            target = getattr(self, key)
            if target is not None and key in checkpoint_dict:
                target.load_state_dict(checkpoint_dict[key])
        self.log(f"[INFO] resuming at epoch {self.epoch}, global step {self.global_step}")

    def load_checkpoint_cond3_Mapping(self, model, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep_Mapping_*_cond_Mapping.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_dict["model"], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        #    """  if self.ema is not None and 'ema' in checkpoint_dict:

        if model.cuda_ray:
            if "mean_count" in checkpoint_dict:
                model.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                model.mean_density = checkpoint_dict["mean_density"]

        if model_only:
            return

        # if self.optimizer and 'optimizer' in checkpoint_dict:
        #     except:

        # if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
        #     except:

        # if self.scaler and 'scaler' in checkpoint_dict:
        #     except:
        return model

    def load_checkpoint_cond3(self, model, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f"{self.ckpt_path}/{self.name}_ep*_cond.pth"))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if "model" not in checkpoint_dict:
            model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_dict["model"], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        #    """  if self.ema is not None and 'ema' in checkpoint_dict:

        if model.cuda_ray:
            if "mean_count" in checkpoint_dict:
                model.mean_count = checkpoint_dict["mean_count"]
            if "mean_density" in checkpoint_dict:
                model.mean_density = checkpoint_dict["mean_density"]

        if model_only:
            return

        # if self.optimizer and 'optimizer' in checkpoint_dict:
        #     except:

        # if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
        #     except:

        # if self.scaler and 'scaler' in checkpoint_dict:
        #     except:
        return model
