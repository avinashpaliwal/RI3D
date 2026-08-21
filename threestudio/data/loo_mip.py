import math
import os
import random
import json
import torch
import numpy as np
import cv2
import pytorch_lightning as pl
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset

import threestudio
from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.typing import *
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat, rotmat2qvec
from utils.camera_utils import resize_mask_image, load_raw_depth, transform_poses_pca, generate_ellipse_path_from_poses, invert_transform_poses_pca, generate_ellipse_path_from_camera_infos
from utils.graphics_utils import getWorld2View2, focal2fov
from .random_camera_sampler import RandomCameraSampler
from scene.dataset_readers import readColmapCameras
from scene.dataset_readers_flow import readMipTransforms
from utils.camera_utils import focus_point_fn

from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, focal2fov
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, transform_pcd, getWorld2View, geom_transform_points

class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: np.ndarray
    FovX: np.ndarray
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int

def getNerfppNorm(cam_centers):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center
    return {"translate": translate.astype(np.float32), "radius": float(radius)}

# def generate_ellipse_path_from_camera_infos(
#         extrinsics,
#         n_frames: int = 120,
#         const_speed: bool = False,
#         z_variation: float = 0.,
#         z_phase: float = 0.
#     ):
#     poses = np.array([np.linalg.inv(getWorld2View2(R, T))[:3, :4] for R, T in extrinsics])
#     poses[:, :, 1:3] *= -1
#     poses, transform, scale_factor = transform_poses_pca(poses)
#     render_poses = generate_ellipse_path_from_poses(poses, n_frames, const_speed, z_variation, z_phase)
#     render_poses = invert_transform_poses_pca(render_poses, transform, scale_factor)
#     render_poses[:, :, 1:3] *= -1
#     ret_cam_infos = []
#     for uid, pose in enumerate(render_poses):
#         R = pose[:3, :3]
#         c2w = np.eye(4)
#         c2w[:3, :4] = pose
#         T = np.linalg.inv(c2w)[:3, 3]
#         ret_cam_infos.append((R, T))
#     return ret_cam_infos

# def add_mask_dep(poses, mask, depth):

#     for idx, pose in enumerate(poses):

#         poses[idx] = poses[idx]._replace(mask=mask)
#         poses[idx] = poses[idx]._replace(mono_depth=depth)
    
    return poses

@dataclass
class LooDataModuleConfig:
    batch_size: int = 1
    data_dir: str = ''
    eval_camera_distance: float = 6.
    resolution: int = 1
    prompt: str = ''
    sparse_num: int = 0
    bg_white: bool = False
    length: int = 1500
    around_gt_steps: int = 750
    refresh_interval: int = 100
    refresh_size: int = 20



from collections import deque

def layered_midpoints(N):
    indices = []
    queue = deque([(0, N - 1)])

    while queue:
        level_size = len(queue)
        current_level = []

        # Process each segment at the current level
        for _ in range(level_size):
            start, end = queue.popleft()
            if start <= end:
                midpoint = (start + end) // 2
                current_level.append(midpoint)

                # Add left and right segments to the queue for the next level
                queue.append((start, midpoint - 1))
                queue.append((midpoint + 1, end))
        
        # Add current level midpoints to indices
        indices.extend(current_level)

    return indices

@register("loo-dataset")
class LooDataset(Dataset):
    def _load_depth(self, cam_info):
        """Locate a view's metric depth prior.

        Mirrors the candidate chain in scene/dataset_readers_flow.py so both
        entry points accept the same layouts -- the depth writer emits into the
        scene root and the image directory, and which one exists depends on how
        the scene was prepared.
        """
        img_dir = os.path.dirname(cam_info.image_path)
        base_name = os.path.splitext(cam_info.image_name)[0]
        candidates = [
            os.path.join(d, f"{prefix}{base_name}_{self.sparse_num}.npy")
            for d in (
                os.path.join(img_dir, "depth_rel"),
                os.path.join(os.path.dirname(img_dir), "depth_rel"),
                os.path.join(self.data_dir, "depth_rel"),
            )
            for prefix in ("inpv2", "inp_dust3r")
        ]
        for c in candidates:
            if os.path.exists(c):
                return np.load(c)
        raise FileNotFoundError(
            f"No depth prior for {base_name} ({self.sparse_num} views). Looked in:\n  "
            + "\n  ".join(candidates)
        )

    def _load_watermark_mask(self, cam_info, height, width):
        """Photometric-loss mask written by the SfM stage, or None.

        Same candidate chain as _load_depth. One mask per scene at image
        resolution -- watermarks are composited at fixed pixel coordinates, which
        is what makes a single mask valid for every view. Returns a (H, W) float
        tensor, 1 = supervise, 0 = ignore.
        """
        img_dir = os.path.dirname(cam_info.image_path)
        sfm_dir = os.path.dirname(img_dir)
        candidates = [
            os.path.join(d, "watermark_mask.png")
            for d in (
                os.path.join(sfm_dir, "masks"),
                os.path.join(self.data_dir, "masks"),
                os.path.join(img_dir, "masks"),
                # inference.py's `wm` stage writes one level above <backend>_sfm/
                os.path.dirname(sfm_dir),
                os.path.dirname(self.data_dir),
            )
        ]
        seen = set()
        for c in candidates:
            c = os.path.normpath(c)
            if c in seen or not os.path.exists(c):
                continue
            seen.add(c)
            m = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[:2] != (height, width):
                # Skip, do not abort -- a stale pointmap-resolution mask must not
                # shadow a correct full-resolution one later in the chain.
                print(f"[!] {c} is {m.shape[1]}x{m.shape[0]} but images are {width}x{height}; "
                      "skipping it.")
                continue
            print(f"[i] Watermark loss mask {c}: ignoring {(m > 127).mean() * 100:.2f}% of pixels")
            return torch.from_numpy((m <= 127).astype(np.float32))
        return None

    def __init__(self, cfg: LooDataModuleConfig, split: str = 'train', sparse_num: int = 0):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.data_dir = self.cfg.data_dir
        self.resolution = self.cfg.resolution
        self.sparse_num = sparse_num
        self.length = self.cfg.length
        self.around_gt_steps = self.cfg.around_gt_steps
        self.refresh_interval = self.cfg.refresh_interval
        self.refresh_size = self.cfg.refresh_size
        self.min_depth = 10000000000.0

        self.max_depth = 0


        cam_infos_unsorted = readMipTransforms(self.data_dir, resolution=cfg.resolution)
        # reading_dir = "images" if images == None else images
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        self.render_cam_infos = generate_ellipse_path_from_camera_infos(cam_infos)#[::-1]

        
        self.fov_scale = 1.1
        
        # FoVy = focal2fov(fov2focal(cam_infos[0].FovX, cam_infos[0].height) * fov_scale, cam_infos[0].height)
        # FoVx = focal2fov(fov2focal(cam_infos[0].FovX, cam_infos[0].width)  * fov_scale, cam_infos[0].width)

        # for i, cam_info in enumerate(self.render_cam_infos):
            
        #     self.render_cam_infos[i] = self.render_cam_infos[i]._replace(FovX=FoVx)
        #     self.render_cam_infos[i] = self.render_cam_infos[i]._replace(FovY=FoVy)

        poses = np.array([np.linalg.inv(getWorld2View2(cam_info.R, cam_info.T))[:3, :4] for cam_info in cam_infos])
        poses[:, :, 1:3] *= -1
        self.scene_center = torch.tensor(focus_point_fn(poses)).cuda()


        # self.masks = np.load('data/mipnerf360/flowers/confs3.npy')
        # self.masks = torch.tensor(self.masks).float().cuda()
        self.confs = np.load(f'{self.data_dir}/confs{self.sparse_num}.npy')
        self.confs = torch.tensor(self.confs).float().cuda()
        # print(self.masks.shape, "++++++++++++++++++++")
        # exit()
        
        with open(f'{self.data_dir}/train_test_split_{self.sparse_num}.json') as json_data:
            data = json.load(json_data)
            train_ids = data['train_ids']

        for train_id in train_ids:

            cam_infos[train_id] = cam_infos[train_id]._replace(image=Image.open(cam_infos[train_id].image_path))

        train_cam_infos = [cam_infos[i] for i in train_ids]

        
        world_view_transformtr = torch.tensor(getWorld2View(train_cam_infos[0].R, train_cam_infos[0].T)).transpose(0, 1).cuda().float()
        train_camera_center = world_view_transformtr.inverse()[3, :3] 
        min_dist = 100000000
        min_index = -1
        for idx, ren_cam in enumerate(self.render_cam_infos):
            world_view_transform = torch.tensor(getWorld2View(ren_cam.R, ren_cam.T)).transpose(0, 1).cuda().float()
            ren_camera_center = world_view_transform.inverse()[3, :3] 
            diff = (train_camera_center - ren_camera_center)
            # print(diff, diff.shape)
            dist = torch.pow(torch.pow(diff, 2).sum(dim=0), 0.5)
            if dist < min_dist:
                min_dist = dist
                min_index = idx

        # self.inp_index = (min_index + 60 - 15) % len(self.render_cam_infos)
        self.inp_index = ( min_index ) % len(self.render_cam_infos)

        # self.first_indices = [0, 15, 8, 24, 4, 12, 20, 28, 2, 6, 10, 14, 18, 22, 26, 1,  3,  5,  7,  9, 11, 13, 17, 19, 21, 23, 25, 27, 29]
        # self.first_indices = [0, 10, 5, 15, 2, 7, 12, 17, 1, 4, 8, 11, 14, 18]
        self.first_indices = layered_midpoints(len(self.render_cam_infos)//(self.refresh_size//2))
        print(self.first_indices, "++++++++++++++++++++")
        self.first_indices = [x + self.inp_index for x in self.first_indices]
        self.first_indices = self.first_indices * 10
        self.first_indices.reverse()
        # [self.inp_index, self.inp_index + 15, self.inp_index + len(self.render_cam_infos) // 2, self.inp_index + 3 * len(self.render_cam_infos) // 4]

        self.Rs, self.Ts, self.heights, self.widths, self.fovxs, self.fovys, self.images, self.masks, self.depths = [], [], [], [], [], [], [], [], []
        cam_c = []
        for cam_info in (train_cam_infos):

            cam_c.append(np.linalg.inv(getWorld2View2(cam_info.R, cam_info.T))[:3, 3:4])
            self.Rs.append(cam_info.R)
            self.Ts.append(cam_info.T)
            self.fovxs.append(cam_info.FovX)
            self.fovys.append(cam_info.FovY)

            # Metric z-depth in SfM world units, aligned to the MASt3R pointmaps
            # by utils/depth_align.py -- used as-is, no rescaling.
            depth_rel = self._load_depth(cam_info)
            if depth_rel.ndim == 3 and depth_rel.shape[-1] == 3:
                depth_rel = depth_rel[..., 0]

            finite = depth_rel[np.isfinite(depth_rel) & (depth_rel > 0)]
            if finite.size:
                self.min_depth = min(self.min_depth, float(finite.min()))
                self.max_depth = max(self.max_depth, float(finite.max()))

            # mask image
            image = (torch.from_numpy(np.array(cam_info.image))/255.).permute(2, 0, 1) # C, H, W

            # 1 = supervise, 0 = ignore. Was unconditionally torch.ones_like, which
            # meant every consumer's `mask > 0.5` gate was always true -- so a
            # burned-in watermark was supervised as scene content here too.
            wm = self._load_watermark_mask(cam_info, image.shape[-2], image.shape[-1])
            mask = torch.ones_like(image)[0].squeeze() if wm is None else wm

            self.images.append(image)
            self.masks.append(mask)
            self.depths.append(depth_rel.squeeze())
            print(image.shape, "++++++++++++++++++++", cam_info.image_path)

            self.heights.append(image.shape[-2])
            self.widths.append(image.shape[-1])
        # exit()
        all_Rs = []
        all_Ts = []
        cam_c = []
        for cam_info in train_cam_infos:
            R = cam_info.R
            T = cam_info.T

            cam_c.append(np.linalg.inv(getWorld2View2(R, T))[:3, 3:4])
            all_Rs.append(R)
            all_Ts.append(T)

        self.cameras_extent = getNerfppNorm(cam_c)
        self.camera_sampler = RandomCameraSampler(self.Rs, self.Ts, all_Rs, all_Ts)

        self.cnt = 0
        self.random_poses = []

        # for i in range(10):
        #     self.next_inp_index()
        
        # exit()

    def next_inp_index(self):
        step_size = int(len(self.render_cam_infos) * self.refresh_interval / self.around_gt_steps) #Assuming 6000 iterations
        self.inp_index = (self.inp_index + 2*step_size -1) % len(self.render_cam_infos)
        print(self.inp_index, int(len(self.render_cam_infos) * self.refresh_interval / self.around_gt_steps), len(self.random_poses), self.refresh_interval, self.around_gt_steps)

    # def refresh_random_poses(self):
    #     self.random_poses = []
    #     dis_from_gt = 0.8
    #     threestudio.info(f'refresh random poses with dis_drom_gt={dis_from_gt} at step {self.cnt}')
    #     self.random_poses = []
    #     while len(self.random_poses) < self.refresh_size:
    #         self.random_poses.extend(self.camera_sampler.sample_away_from_gt(dis_from_gt))
    #     self.random_poses = self.random_poses[:self.refresh_size]

    
    # def refresh_random_render_poses(self):
    #     threestudio.info(f'refresh random render poses at step {self.cnt}')
    #     random_indices = random.sample(range(len(self.render_poses) - 1), self.refresh_size)
    #     self.random_poses = []
    #     for random_idx in random_indices:
    #         self.random_poses.append(self.render_poses[random_idx])
    #     self.random_poses = self.random_poses[:self.refresh_size]


    def refresh_random_render_poses(self):
        threestudio.info(f'refresh random render poses at step {self.cnt}')

        first_random = self.first_indices.pop()
        step_size    = len(self.render_cam_infos) // (self.refresh_size)
        # random_indices = [self.inp_index] + [first_random + i * step_size for i in range(self.refresh_size - 1)]
        random_indices = [(first_random + i * step_size) % len(self.render_cam_infos) for i in range(self.refresh_size)]
        print("random_indices: ", random_indices)
        self.random_poses = []
        for random_idx in random_indices:
            self.random_poses.append(self.render_cam_infos[random_idx])
        # self.random_poses = self.random_poses[:self.refresh_size]
        # self.next_inp_index()

    def __len__(self):
        if self.split == 'train':
            return self.cfg.length
        elif len(self.images):
            return len(self.images)
        else:
            return len(self.Rs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.split == 'train':
            idx = random.randint(0, len(self.images) - 1)
            random_index = random.randint(0, self.refresh_size - 1)
            if self.cnt < self.around_gt_steps:
                if self.cnt % self.cfg.refresh_interval== 0:
                    # if self.cnt < self.around_gt_steps//7:
                    #     self.refresh_random_poses()
                    # else:
                    self.refresh_random_render_poses()
                    # self.random_poses = add_mask_dep(self.random_poses, self.masks[0], self.depths[0])
                # print(self.random_poses[random_index], "++++++++++++++++++++", self.random_poses)
                cam_info = self.random_poses[random_index]
            else:
                # random_R, random_T = self.camera_sampler.sample(None)
                cam_info = self.random_poses[0]
            self.cnt += 1
        else:
            # theta = 2 * math.pi * idx / len(self)
            # random_index = idx
            # random_R, random_T = self.camera_sampler.sample(theta)
            cam_info = self.random_poses[0]

        # print(cam_info.width, cam_info.height, "++++++++++++++++++++")
        # exit()
        ret = {
            "index": idx,
            "R": self.Rs[idx],
            "T": self.Ts[idx],
            "height": self.heights[idx],
            "width": self.widths[idx],
            "fovx": self.fovxs[idx],
            "fovy": self.fovys[idx],
            "image": self.images[idx],
            "mask": self.masks[idx],
            "depth": self.depths[idx],
            "txt": self.cfg.prompt,
            "random_index": random_index,
            "random_R": cam_info.R,
            "random_T": cam_info.T,
            'random_height': cam_info.height//4,
            'random_width': cam_info.width//4,
            'random_fovx': cam_info.FovX,
            'random_fovy': cam_info.FovY,
            "random_poses": [[p.R, p.T] for p in self.random_poses],
            'random_distance': cam_info.distance,
            'distance': self.render_cam_infos[0].distance,
            "gt_images": self.images,
            "gt_Ts": self.Ts,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "fov_scale": self.fov_scale,
            "conf": self.confs[idx],
        }
        # print(ret, idx)
        # print(ret)
        return ret

    def get_scene_extent(self):
        return self.cameras_extent

    def norm_to_pc(self, center):
        self.Ts = [(T - center) for T in self.Ts]


@register("loo-datamodule")
class LooDataModuleFromConfig(pl.LightningDataModule):
    cfg: LooDataModuleConfig
    train_dataset: Optional[LooDataset] = None
    val_dataset: Optional[LooDataset] = None
    test_dataset: Optional[LooDataset] = None

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(LooDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = LooDataset(self.cfg, "train", sparse_num=self.cfg.sparse_num)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = LooDataset(self.cfg, "val", sparse_num=self.cfg.sparse_num)
        if stage in [None, "test", "predict"]:
            self.test_dataset = LooDataset(self.cfg, "test", sparse_num=self.cfg.sparse_num)

    def norm_to_pc(self, center):
        if self.train_dataset is not None:
            self.train_dataset.norm_to_pc(center)
        if self.val_dataset is not None:
            self.val_dataset.norm_to_pc(center)
        if self.test_dataset is not None:
            self.test_dataset.norm_to_pc(center)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1
        )
