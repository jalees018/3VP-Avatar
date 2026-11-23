import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import pickle as pkl

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, get_rays_patches

from scipy.spatial.transform import Rotation as R
from paths import IMAGES_DIR, TRACKER_DIR, WORK_DIR


def camera_path_pos_floating(radius, num_pts, theta=2 * np.pi, neutr_cam=np.eye(3, 4)):
    neutr_cam[2, 3] = radius
    pose = []
    for i in range(0, num_pts):
        t = i / num_pts
        pitch = 0.25 * np.cos(t * 2 * np.pi)
        yaw = 0.45 * np.sin(t * 2 * np.pi)
        pitch1 = pitch * np.array([1, 0, 0])
        yaw1 = yaw * np.array([0, 1, 0])
        pitch1[1] = yaw
        yaww = R.from_rotvec(pitch1).as_matrix()
        camera = np.matmul(yaww, neutr_cam)
        camera = camera_path_dir(camera[:, 3], num_pts, to=None)
        pose.append(camera)
    return np.array(pose).astype(np.float32)


def camera_path_pos_horizontals(radius, num_pts, height=0.0, theta=2 * np.pi, neutr_cam=np.eye(3, 4)):
    neutr_cam[1, 3] = height
    neutr_cam[2, 3] = radius
    t = 2.0 * theta / num_pts
    k = 0
    anchor = theta
    pose = []
    for i in range(0, num_pts):
        yaw1 = anchor - k
        yaw = R.from_rotvec(yaw1 * np.array([0, 1, 0])).as_matrix()
        camera = np.matmul(yaw, neutr_cam)
        camera = camera_path_dir(camera[:, 3], num_pts, to=None)
        k = k + t
        pose.append(camera)
    return np.array(pose).astype(np.float32)


def camera_path_dir(pos, num_pts, to=None):
    if to is None:
        to = np.zeros((1, 3))
    dirs = pos - to
    directions = dirs / np.linalg.norm(pos - to, axis=1, keepdims=True)
    temp = np.zeros_like(to)
    temp[:, 1] = 1
    right = np.cross(temp, directions)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)
    c2w = np.zeros((4, 4))
    up = np.cross(directions, right)
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    c2w[0, 0:3] = right
    c2w[1, 0:3] = up
    c2w[2, 0:3] = directions
    c2w[3, 0:3] = pos
    c2w[3, 3] = 1
    c2w[3, 0:3] = pos
    return c2w.astype(np.float32)


#     for i in range(0,num_pts):
#     for i in range(0,num_pts):
#     if to is None:
# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array(
        [
            [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
            [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
            [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return new_pose


def rand_poses(size, device, radius=1, theta_range=[np.pi / 3, 2 * np.pi / 3], phi_range=[0, 2 * np.pi]):
    """generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    """

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack(
        [
            radius * torch.sin(thetas) * torch.sin(phis),
            radius * torch.cos(thetas),
            radius * torch.sin(thetas) * torch.cos(phis),
        ],
        dim=-1,
    )  # [B, 3]

    # lookat
    forward_vector = -normalize(centers)
    up_vector = (
        torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    )  # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDatasetCond:
    def __init__(self, opt, device, type="train", downscale=1, n_test=10):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload  # preload data into GPU
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset  # camera offset
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16  # if preload, load into fp16.

        self.training = self.type in ["train", "all", "trainval"]
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.notrain_poses = opt.notrainposes

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, "transforms.json")):
            self.mode = "colmap"  # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, "transforms_train.json")):
            self.mode = "blender"  # provided split
        else:
            raise NotImplementedError(f"[NeRFDataset] Cannot find transforms*.json under {self.root_path}")

        # load nerf-compatible format data.
        if self.mode == "colmap":
            with open(os.path.join(self.root_path, "transforms.json"), "r") as f:
                transform = json.load(f)
        elif self.mode == "blender":
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == "all":
                transform_paths = glob.glob(os.path.join(self.root_path, "*.json"))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, "r") as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform["frames"].extend(tmp_transform["frames"])
            # load train and val split
            elif type == "trainval":
                with open(os.path.join(self.root_path, f"transforms_train.json"), "r") as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f"transforms_val.json"), "r") as f:
                    transform_val = json.load(f)
                transform["frames"].extend(transform_val["frames"])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f"transforms_{type}.json"), "r") as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f"unknown dataset mode: {self.mode}")

        # load image size
        if "h" in transform and "w" in transform:
            self.H = 512
            self.W = 512
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]
        with open(os.path.join(self.root_path, f"trajectory.json"), "r") as f:
            meta111 = json.load(f)

        # for colmap, manually interpolate a test set.
        if self.mode == "colmap" and type == "test":

            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(
                np.array(f0["transform_matrix"], dtype=np.float32), scale=self.scale, offset=self.offset
            )  # [4, 4]
            pose1 = nerf_matrix_to_ngp(
                np.array(f1["transform_matrix"], dtype=np.float32), scale=self.scale, offset=self.offset
            )  # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == "colmap":
                if type == "train":
                    frames = frames[1:]
                elif type == "val":
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames

            self.poses = []
            self.poses2 = []
            self.images = []
            self.lstt_111 = []
            self.jp = []
            self.expr = []
            kkkk = 0
            lsttte = os.listdir(IMAGES_DIR + self.opt.subject + "/Images/_results/")
            incrementor_111 = int(min([int(((iii).split("/")[-1])[:-4]) for iii in lsttte]))
            print("Incrementor: ", incrementor_111)
            for f in tqdm.tqdm(frames, desc=f"Loading {type} data"):
                f_path = os.path.join(self.root_path, f["file_path"])
                if self.mode == "blender" and "." not in os.path.basename(f_path):
                    f_path += ".png"
                self.lstt_111.append(int(f_path[-8:-4]))
                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                f_path2 = (
                    IMAGES_DIR + self.opt.subject + "/Images/_results/" + str(kkkk + incrementor_111).zfill(4) + ".png"
                )
                path1 = TRACKER_DIR + self.opt.expr + self.opt.subject + "/"
                abcdefghhhh = pkl.load(open(path1 + "result_final_Real_tt_" + str(kkkk) + ".pkl", "rb"))
                jpp = abcdefghhhh["jaw_pose"]
                exprs = abcdefghhhh["expressions"]
                self.jp.append(jpp)
                self.expr.append(exprs)
                pose = np.array(f["transform_matrix"], dtype=np.float32)  # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image = cv2.imread(f_path2, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

                image = image.astype(np.float32) / 255  # [H, W, 3/4]

                kkkk = kkkk + 1
            kkkk2 = 0
            for iikkk in tqdm.tqdm(range(0, 300)):  # f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path2 = (
                    IMAGES_DIR + self.opt.subject + "/Images/_results/" + str(iikkk + incrementor_111).zfill(4) + ".png"
                )
                image = cv2.imread(f_path2, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

                image = image.astype(np.float32) / 255  # [H, W, 3/4]
                self.images.append(image)
                kkkk2 = kkkk2 + 1
            ###for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
            print("Loading Trajectory Data")
            self.train_poses = not self.notrain_poses
            kk1 = 0
            lstt_222 = list(self.lstt_111)
            lstt_222.sort()
            for i in self.lstt_111:  # for t, frame in enumerate(meta111[::skip]):
                frame = meta111[lstt_222.index(i)]
                if self.train_poses:
                    posee = np.array(frame["transform_matrix"], dtype=np.float32)
                else:
                    pts = camera_path_pos_floating(-4.0, len(self.lstt_111), 10 * np.pi / 180)
                    poses3 = np.transpose(pts[kk1])
                    posee = np.zeros_like(poses3)
                    posee[2, 0:4] = poses3[1, 0:4]
                    posee[1, 0:4] = poses3[0, 0:4]
                    posee[0, 0:4] = poses3[2, 0:4]
                pose2 = posee[0:3, 0:4].astype(np.float32)
                pose = nerf_matrix_to_ngp(posee[0:3, 0:4].astype(np.float32), scale=self.scale, offset=self.offset)
                self.poses.append(pose)
                self.poses2.append(pose2)
                kk1 = kk1 + 1
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))  # [N, 4, 4]
        self.poses2 = torch.from_numpy(np.stack(self.poses2, axis=0))  # [N, 4, 4]
        id_111 = np.where(np.array(self.lstt_111) == np.array(self.lstt_111).max())

        poses0 = self.poses[[id_111[0]]].repeat((300, 1, 1))
        poses20 = self.poses2[[id_111[0]]].repeat((300, 1, 1))

        self.poses = torch.cat((self.poses, poses0), 0)
        self.poses2 = torch.cat((self.poses2, poses20), 0)
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0))  # [N, H, W, C]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones(
                [self.images.shape[0], 128 * 128], dtype=torch.float
            )  # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.

        # [debug] uncomment to view examples of randomly generated poses.

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != "linear":
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        focal_scale = self.H / int(transform["h"])
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

        cx = (transform["cx"] * focal_scale) if "cx" in transform else (self.W / 2)
        cy = (transform["cy"] * focal_scale) if "cy" in transform else (self.H / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):

        B = len(index)  # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays)  # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                "H": rH,
                "W": rW,
                "rays_o": rays["rays_o"],
                "rays_d": rays["rays_d"],
                "intrinsics": self.intrinsics,
                "poses": poses,
            }

        poses = self.poses[index].to(self.device)  # [B, 4, 4]
        diff = np.array(self.lstt_111).max() - np.array(self.lstt_111).min()
        offf = (diff) / len(self.lstt_111)
        if self.opt.view == "results_side":
            limm = np.array(self.lstt_111).min() + int((2 / 8) * (diff))
            id_111 = np.where(np.sort(np.array(self.lstt_111)) >= limm + offf)
        elif self.opt.view == "results_side2":
            limm = np.array(self.lstt_111).min() + int((6 / 8) * (diff))
            id_111 = np.where(np.sort(np.array(self.lstt_111)) >= limm + offf)
        elif self.opt.view == "results_side7":
            limm = np.array(self.lstt_111).min() + int((3 / 8) * (diff))
            id_111 = np.where(np.sort(np.array(self.lstt_111)) >= limm + offf)
        else:
            id_111 = np.where(np.array(self.lstt_111) == np.array(self.lstt_111).max())
        if self.train_poses:
            poses2 = self.poses2[[(id_111[0])[0:1]]].to(
                self.device
            )  # [B, 4, 4] #poses2 = self.poses2[[id_111[0]]].to(self.device) # [B, 4, 4]
            poses3 = self.poses[[(id_111[0])[0:1]]].to(
                self.device
            )  # [B, 4, 4] #poses3 = self.poses[[id_111[0]]].to(self.device) # [B, 4, 4]

            # if False:
        else:
            poses2 = self.poses2.to(
                self.device
            )  # [B, 4, 4] #poses2 = self.poses2[[id_111[0]]].to(self.device) # [B, 4, 4]
            poses3 = self.poses.to(
                self.device
            )  # [B, 4, 4] #poses3 = self.poses[[id_111[0]]].to(self.device) # [B, 4, 4]
        error_map = None if self.error_map is None else self.error_map[index]

        if self.train_poses:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map)
            rays2 = get_rays(poses3, self.intrinsics, self.H, self.W, self.num_rays, error_map)

        if self.training:
            results = {
                "H": self.H,
                "W": self.W,
                "rays_o": rays["rays_o"],
                "rays_d": rays["rays_d"],
                "intrinsics": self.intrinsics,
                "poses": poses2,
                "rays_o2": rays2["rays_o"],
                "rays_d2": rays2["rays_d"],
                "inxcices": rays2["inds"],
                "jaw_pose": torch.tensor(self.jp[index[0]]).to(self.device),
                "expressions": torch.tensor(self.expr[index[0]]).to(self.device),
                "index": index[0],
                "length": len(self.lstt_111),
            }
        else:
            if self.train_poses:
                results = {
                    "H": self.H,
                    "W": self.W,
                    "intrinsics": self.intrinsics,
                    "poses": poses2,
                    "rays_o2": rays2["rays_o"],
                    "rays_d2": rays2["rays_d"],
                    "jaw_pose": torch.tensor(self.jp[index[0]]).to(self.device),
                    "expressions": torch.tensor(self.expr[index[0]]).to(self.device),
                    "length": len(self.lstt_111),
                }
            else:
                results = {
                    "H": self.H,
                    "W": self.W,
                    "intrinsics": self.intrinsics,
                    "poses": poses2,
                    "rays_o2": poses3,
                    "jaw_pose": torch.tensor(self.jp[index[0]]).to(self.device),
                    "expressions": torch.tensor(self.expr[index[0]]).to(self.device),
                    "length": len(self.lstt_111),
                }
        if self.images is not None:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays2["inds"]], -1))  # [B, N, 3/4]
            results["images"] = images

        # need inds to update error_map
        if error_map is not None:
            results["index"] = index
            results["inds_coarse"] = rays2["inds_coarse"]

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose  # index >= size means we use random pose.
        loader = DataLoader(
            list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0
        )
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader


class NeRFDatasetCondOr:
    def __init__(self, opt, device, type="train", downscale=1, n_test=10, freeze=False, can_only=False):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload  # preload data into GPU
        self.scale = opt.scale  # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset  # camera offset
        self.bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16  # if preload, load into fp16.

        self.training = self.type in ["train", "all", "trainval"]
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose
        self.notrain_poses = opt.notrainposes

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, "transforms.json")):
            self.mode = "colmap"  # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, "transforms_train.json")):
            self.mode = "blender"  # provided split
        else:
            raise NotImplementedError(f"[NeRFDataset] Cannot find transforms*.json under {self.root_path}")

        # load nerf-compatible format data.
        if self.mode == "colmap":
            with open(os.path.join(self.root_path, "transforms.json"), "r") as f:
                transform = json.load(f)
        elif self.mode == "blender":
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == "all":
                transform_paths = glob.glob(os.path.join(self.root_path, "*.json"))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, "r") as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform["frames"].extend(tmp_transform["frames"])
            # load train and val split
            elif type == "trainval":
                with open(os.path.join(self.root_path, f"transforms_train.json"), "r") as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f"transforms_val.json"), "r") as f:
                    transform_val = json.load(f)
                transform["frames"].extend(transform_val["frames"])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f"transforms_{type}.json"), "r") as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f"unknown dataset mode: {self.mode}")

        # load image size
        if "h" in transform and "w" in transform:
            self.H = 512
            self.W = 512
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]
        with open(os.path.join(self.root_path, f"trajectory.json"), "r") as f:
            meta111 = json.load(f)

        with open(WORK_DIR + self.opt.subject + ".json") as ff:
            fr = json.load(ff)
        # for colmap, manually interpolate a test set.
        if self.mode == "colmap" and type == "test":

            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(
                np.array(f0["transform_matrix"], dtype=np.float32), scale=self.scale, offset=self.offset
            )  # [4, 4]
            pose1 = nerf_matrix_to_ngp(
                np.array(f1["transform_matrix"], dtype=np.float32), scale=self.scale, offset=self.offset
            )  # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == "colmap":
                if type == "train":
                    frames = frames[1:]
                elif type == "val":
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames

            self.poses = []
            self.poses2 = []
            self.images0 = []
            self.images = []
            self.lstt_111 = []
            self.jp = []
            self.expr = []
            self.error_map2 = []
            kkkk = 0
            lsttte = os.listdir(IMAGES_DIR + self.opt.subject + "/Images/_results/")
            incrementor_111 = int(min([int(((iii).split("/")[-1])[:-4]) for iii in lsttte]))  # 1560
            print("Incrementor: ", incrementor_111)
            for f in tqdm.tqdm(frames, desc=f"Loading {type} data"):
                f_path = os.path.join(self.root_path, f["file_path"])
                if self.mode == "blender" and "." not in os.path.basename(f_path):
                    f_path += ".png"
                self.lstt_111.append(int(f_path[-8:-4]))
                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                path1 = TRACKER_DIR + self.opt.expr + self.opt.subject + "/"
                abcdefghhhh000 = pkl.load(open(path1 + "result_final_Real_tt_" + str(0) + ".pkl", "rb"))
                jpp = np.expand_dims(abcdefghhhh000["jaw_pose"], axis=0)
                exprs = np.expand_dims((abcdefghhhh000["expressions"]), axis=0)[:, 0:13]
                self.jp.append(jpp)
                self.expr.append(exprs)
                pose = np.array(f["transform_matrix"], dtype=np.float32)  # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                image0 = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                if self.H is None or self.W is None:
                    self.H = image0.shape[0] // downscale
                    self.W = image0.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image0.shape[-1] == 3:
                    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGBA)
                else:
                    image0 = cv2.cvtColor(image0, cv2.COLOR_BGRA2RGBA)

                if image0.shape[0] != self.H or image0.shape[1] != self.W:
                    image0 = cv2.resize(image0, (self.W, self.H), interpolation=cv2.INTER_AREA)

                # kept as uint8 (converted to float in collate) to reduce memory
                baar = (
                    np.load(IMAGES_DIR + self.opt.subject + "/Images/error_maps/" + str(kkkk).zfill(4) + ".npy").astype(
                        np.float32
                    )
                )[:, :, 0].reshape(1, -1)[0]
                baar[baar == 4.0] = 2.0
                self.error_map2.append(baar)  # [B, 128 * 128], flattened for easy indexing, fixed resolution...

                self.images0.append(image0)
                kkkk = kkkk + 1
            kkkk2 = 0
            # one preallocated uint8 buffer for all frames (360° turn + sequence) to keep peak memory low
            n0 = len(self.images0)
            self.images = np.empty((n0 + self.opt.train_imgs,) + self.images0[0].shape, dtype=np.uint8)
            self.images[:n0] = np.stack(self.images0, axis=0)
            del self.images0
            for iikkk in tqdm.tqdm(
                range(0, self.opt.train_imgs)
            ):  # f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path2 = (
                    IMAGES_DIR + self.opt.subject + "/Images/_results/" + str(iikkk + incrementor_111).zfill(4) + ".png"
                )
                image = cv2.imread(f_path2, cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
                abcdefghhhh000 = pkl.load(open(path1 + "result_final_Real_tt_" + str(iikkk) + ".pkl", "rb"))
                jpp = np.expand_dims(abcdefghhhh000["jaw_pose"], axis=0)
                exprs = np.expand_dims((abcdefghhhh000["expressions"]), axis=0)[:, 0:13]
                self.jp.append(jpp)
                self.expr.append(exprs)

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

                # kept as uint8 (converted to float in collate)
                aaar = (
                    np.load(
                        IMAGES_DIR
                        + self.opt.subject
                        + "/Images/error_maps/"
                        + str(iikkk + len(self.lstt_111)).zfill(4)
                        + ".npy"
                    ).astype(np.float32)
                )[:, :, 0].reshape(1, -1)[0]
                if (iikkk) < fr[11]:
                    aaar[aaar == 4.0] = 6.0
                else:
                    aaar[aaar == 4.0] = 8.0

                self.error_map2.append(aaar)  # [B, 128 * 128], flattened for easy indexing, fixed resolution...

                self.images[n0 + kkkk2] = image
                kkkk2 = kkkk2 + 1
            ###for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
            print("Loading Trajectory Data")
            self.train_poses = not self.notrain_poses
            kk1 = 0
            lstt_222 = list(self.lstt_111)
            lstt_222.sort()
            for i in self.lstt_111:  # for t, frame in enumerate(meta111[::skip]):
                frame = meta111[lstt_222.index(i)]
                if self.train_poses:
                    posee = np.array(frame["transform_matrix"], dtype=np.float32)
                else:
                    pts = camera_path_pos_floating(4.0, len(self.lstt_111), 10 * np.pi / 180)
                    poses3 = np.transpose(camera_path_dir(pts, len(self.lstt_111)), (0, 2, 1))[kk1]
                    posee = np.zeros_like(poses3)
                    posee[2, 0:4] = poses3[1, 0:4]
                    posee[1, 0:4] = poses3[0, 0:4]
                    posee[0, 0:4] = poses3[2, 0:4]
                pose2 = posee[0:3, 0:4].astype(np.float32)
                pose = nerf_matrix_to_ngp(posee[0:3, 0:4].astype(np.float32), scale=self.scale, offset=self.offset)
                self.poses.append(pose)
                self.poses2.append(pose2)
                kk1 = kk1 + 1
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0))  # [N, 4, 4]
        self.poses2 = torch.from_numpy(np.stack(self.poses2, axis=0))  # [N, 4, 4]
        id_111 = np.where(np.array(self.lstt_111) == np.array(self.lstt_111).max())
        self.id_1112 = id_111

        poses0 = self.poses[[id_111[0]]].repeat((self.opt.train_imgs, 1, 1))
        poses20 = self.poses2[[id_111[0]]].repeat((self.opt.train_imgs, 1, 1))

        self.poses = torch.cat((self.poses, poses0), 0)
        self.poses2 = torch.cat((self.poses2, poses20), 0)

        self.jp = torch.tensor(np.stack(self.jp, axis=0)).to(self.device).unsqueeze(-1).unsqueeze(-1)
        self.expr = torch.tensor(np.stack(self.expr, axis=0)).to(self.device).unsqueeze(-1).unsqueeze(-1)
        print(self.expr.shape)
        if self.images is not None:
            self.images = torch.from_numpy(self.images)  # [N, H, W, C] uint8
            self.error_map2 = torch.from_numpy(np.stack(self.error_map2, axis=0))  # [N, H, W, C]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones(
                [self.images.shape[0], 128 * 128], dtype=torch.float
            )  # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None
        # [debug] uncomment to view all training poses.

        # [debug] uncomment to view examples of randomly generated poses.

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != "linear":
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = torch.from_numpy(self.images.numpy().astype(np.float32) / 255).to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        focal_scale = self.H / int(transform["h"])
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

        cx = (transform["cx"] * focal_scale) if "cx" in transform else (self.W / 2)
        cy = (transform["cy"] * focal_scale) if "cy" in transform else (self.H / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        self.listr = [10.0] * len(self.lstt_111)
        self.listr[0:11] = [1.0] * 11
        self.listr[-6:] = [1.0] * 6
        self.listr[0 : len(self.lstt_111)] = [61.0] * len(self.lstt_111)
        self.listr += (0.0 * np.load(WORK_DIR + self.opt.workspace + "/Probs222.npy")).tolist()
        self.listr[len(self.lstt_111) + 0 : len(self.lstt_111) + fr[10][1]] = [1.0] * fr[10][1]
        self.listr[len(self.lstt_111) + fr[11] : len(self.lstt_111) + self.opt.train_imgs] = [2.0] * (
            self.opt.train_imgs - fr[11]
        )

        self.listr[len(self.lstt_111) + fr[2][0] : len(self.lstt_111) + fr[2][1]] = [1.0] * (fr[2][1] - fr[2][0])

        self.listr[len(self.lstt_111) + 0 : len(self.lstt_111) + fr[0][0]] = [0.0] * fr[0][0]  # 5 #2
        self.listr[len(self.lstt_111) + fr[0][1] : len(self.lstt_111) + fr[1][0]] = [0.0] * (
            fr[1][0] - fr[0][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[1][1] : len(self.lstt_111) + fr[2][0]] = [0.0] * (
            fr[2][0] - fr[1][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[2][1] : len(self.lstt_111) + fr[3][0]] = [0.0] * (
            fr[3][0] - fr[2][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[3][1] : len(self.lstt_111) + fr[4][0]] = [0.0] * (
            fr[4][0] - fr[3][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[4][1] : len(self.lstt_111) + fr[5][0]] = [0.0] * (
            fr[5][0] - fr[4][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[5][1] : len(self.lstt_111) + fr[6][0]] = [0.0] * (
            fr[6][0] - fr[5][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[6][1] : len(self.lstt_111) + fr[7][0]] = [0.0] * (
            fr[7][0] - fr[6][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[7][1] : len(self.lstt_111) + fr[8][0]] = [0.0] * (
            fr[8][0] - fr[7][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[8][1] : len(self.lstt_111) + fr[9][0]] = [0.0] * (
            fr[9][0] - fr[8][1]
        )  # 5 #2
        self.listr[len(self.lstt_111) + fr[9][1] : len(self.lstt_111) + fr[10][0]] = [0.0] * (
            fr[10][0] - fr[9][1]
        )  # 5 #2

        if self.opt.trainMapping:
            self.listr[0 : len(self.lstt_111)] = [0.0] * len(self.lstt_111)
            self.listr[len(self.lstt_111) + 0 : len(self.lstt_111) + fr[11]] = [0.0] * fr[11]

            self.listr[len(self.lstt_111) + fr[0][0] + 0 : len(self.lstt_111) + fr[0][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[1][0] + 0 : len(self.lstt_111) + fr[1][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[2][0] + 0 : len(self.lstt_111) + fr[2][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[3][0] + 0 : len(self.lstt_111) + fr[3][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[4][0] + 0 : len(self.lstt_111) + fr[4][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[5][0] + 0 : len(self.lstt_111) + fr[5][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[6][0] + 0 : len(self.lstt_111) + fr[6][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[7][0] + 0 : len(self.lstt_111) + fr[7][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[8][0] + 0 : len(self.lstt_111) + fr[8][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[9][0] + 0 : len(self.lstt_111) + fr[9][0] + 1] = [1.0] * 1
            self.listr[len(self.lstt_111) + fr[10][0] + 0 : len(self.lstt_111) + fr[10][0] + 1] = [1.0] * 1

            self.listr[len(self.lstt_111) + fr[11] : len(self.lstt_111) + self.opt.train_imgs] = [0.04] * (
                self.opt.train_imgs - fr[11]
            )  # 0.04 sampling

        if can_only:
            self.listr[0 : len(self.lstt_111)] = [81.0] * len(self.lstt_111)
            self.listr[len(self.lstt_111) + 0 : len(self.lstt_111) + fr[10][1]] = [1.0] * fr[10][1]
            self.listr[len(self.lstt_111) + fr[11] : len(self.lstt_111) + self.opt.train_imgs] = [2.0] * (
                self.opt.train_imgs - fr[11]
            )

            self.listr[len(self.lstt_111) + fr[2][0] : len(self.lstt_111) + fr[2][1]] = [1.0] * (fr[2][1] - fr[2][0])

            self.listr[len(self.lstt_111) + 0 : len(self.lstt_111) + fr[0][0]] = [0.0] * fr[0][0]  # 5 #2
            self.listr[len(self.lstt_111) + fr[0][1] : len(self.lstt_111) + fr[1][0]] = [0.0] * (
                fr[1][0] - fr[0][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[1][1] : len(self.lstt_111) + fr[2][0]] = [0.0] * (
                fr[2][0] - fr[1][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[2][1] : len(self.lstt_111) + fr[3][0]] = [0.0] * (
                fr[3][0] - fr[2][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[3][1] : len(self.lstt_111) + fr[4][0]] = [0.0] * (
                fr[4][0] - fr[3][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[4][1] : len(self.lstt_111) + fr[5][0]] = [0.0] * (
                fr[5][0] - fr[4][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[5][1] : len(self.lstt_111) + fr[6][0]] = [0.0] * (
                fr[6][0] - fr[5][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[6][1] : len(self.lstt_111) + fr[7][0]] = [0.0] * (
                fr[7][0] - fr[6][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[7][1] : len(self.lstt_111) + fr[8][0]] = [0.0] * (
                fr[8][0] - fr[7][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[8][1] : len(self.lstt_111) + fr[9][0]] = [0.0] * (
                fr[9][0] - fr[8][1]
            )  # 5 #2
            self.listr[len(self.lstt_111) + fr[9][1] : len(self.lstt_111) + fr[10][0]] = [0.0] * (
                fr[10][0] - fr[9][1]
            )  # 5 #2
        self.threshold = 760  # 160
        if freeze:
            self.listr[0 : len(self.lstt_111)] = [0.0] * len(self.lstt_111)
            self.listr[len(self.lstt_111) + 0 : len(self.lstt_111) + self.opt.train_imgs] = [0.0] * self.opt.train_imgs

            self.listr[len(self.lstt_111) + fr[11] : len(self.lstt_111) + self.opt.train_imgs] = [1.0] * (
                self.opt.train_imgs - fr[11]
            )

            self.threshold = 1
            print("Freeze!")

        self.json_file = fr

    def collate(self, index):

        B = len(index)  # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays)  # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                "H": rH,
                "W": rW,
                "rays_o": rays["rays_o"],
                "rays_d": rays["rays_d"],
                "intrinsics": self.intrinsics,
                "poses": poses,
            }

        if self.opt.view == "results_side":
            id_111 = np.where(np.array(self.lstt_111) == (np.array(self.lstt_111).min() + 200))
        elif self.opt.view == "results_side2":
            id_111 = np.where(np.array(self.lstt_111) == (np.array(self.lstt_111).min() + 600))
        else:
            id_111 = self.id_1112
        if self.train_poses:
            if index[0] < len(self.lstt_111) and self.training:
                poses2 = self.poses2[index].to(
                    self.device
                )  # [B, 4, 4] #poses2 = self.poses2[[id_111[0]]].to(self.device) # [B, 4, 4]
                poses3 = self.poses[index].to(
                    self.device
                )  # [B, 4, 4] #poses3 = self.poses[[id_111[0]]].to(self.device) # [B, 4, 4]
            else:
                poses2 = self.poses2[[id_111[0]]].to(
                    self.device
                )  # [B, 4, 4] #poses2 = self.poses2[[id_111[0]]].to(self.device) # [B, 4, 4]
                poses3 = self.poses[[id_111[0]]].to(
                    self.device
                )  # [B, 4, 4] #poses3 = self.poses[[id_111[0]]].to(self.device) # [B, 4, 4]
        else:
            poses2 = self.poses2[index].to(
                self.device
            )  # [B, 4, 4] #poses2 = self.poses2[[id_111[0]]].to(self.device) # [B, 4, 4]
            poses3 = self.poses[index].to(
                self.device
            )  # [B, 4, 4] #poses3 = self.poses[[id_111[0]]].to(self.device) # [B, 4, 4]
        error_map = None if self.error_map is None else self.error_map[index]
        error_map2 = self.error_map2[index].to(self.device)

        rays2 = get_rays_patches(poses3, self.intrinsics, self.H, self.W, self.num_rays, error_map2)

        if self.training:
            results = {
                "H": self.H,
                "W": self.W,
                "length": len(self.lstt_111),
                "intrinsics": self.intrinsics,
                "poses": poses2,
                "rays_o2": rays2["rays_o"],
                "rays_d2": rays2["rays_d"],
                "inxcices": rays2["inds"],
                "jaw_pose": self.jp[index],
                "expressions": self.expr[index],
                "index": index[0],
                "json_file": self.json_file,
                "threshold": self.threshold,
            }
        else:
            results = {
                "H": self.H,
                "W": self.W,
                "length": len(self.lstt_111),
                "intrinsics": self.intrinsics,
                "poses": poses2,
                "rays_o2": rays2["rays_o"],
                "rays_d2": rays2["rays_d"],
                "jaw_pose": torch.tensor(self.jp[0]).to(self.device),
                "expressions": torch.tensor(self.expr[0]).to(self.device),
                "index": index[0],
                "json_file": self.json_file,
                "threshold": self.threshold,
            }
        if self.images is not None:
            if self.images.dtype == torch.uint8:  # stored as uint8, same conversion as numpy astype(float32) / 255
                images = torch.from_numpy(self.images[index].numpy().astype(np.float32) / 255).to(self.device)
            else:
                images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
            results["images"] = images
            results["images2"] = images[:, (rays2["inds"])[0, :, 1].long(), (rays2["inds"])[0, :, 0].long(), :].view(
                B, -1, C
            )

        # need inds to update error_map
        if error_map is not None:
            results["index"] = index
            results["inds_coarse"] = rays2["inds_coarse"]

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose  # index >= size means we use random pose.
        loader = DataLoader(
            list(range(size)),
            # weights are defined for the full sequence; keep only those of the loaded frames
            sampler=torch.utils.data.WeightedRandomSampler(self.listr[:size], size, replacement=True),
            batch_size=1,
            collate_fn=self.collate,
            num_workers=0,
        )
        loader._data = self  # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader
