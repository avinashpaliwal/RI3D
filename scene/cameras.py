#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2_tensor, fov2focal, focal2fov

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, mono_depth,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", white_background=False, distance=None,
                 loss_mask=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.distance = distance

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.mask = gt_alpha_mask.to(self.data_device)
            self.original_image[(self.mask <= 0.5).expand_as(self.original_image)] = 1.0 if white_background else 0.0
        else:
            self.mask = None

        # Photometric-loss mask: 1 = supervise, 0 = ignore. Unlike `mask` above it
        # never touches `original_image` -- the point is to leave those pixels
        # unsupervised, not to replace them with a background colour the losses
        # would then dutifully reproduce.
        self.loss_mask = loss_mask.to(self.data_device) if loss_mask is not None else None

        if mono_depth is not None:
            # self.mono_depth = mono_depth.to(self.data_device) if mono_depth is not None else None
            self.mono_depth = mono_depth.to(self.data_device)
            # self.mono_depth[(self.mask <= 0.5)] = 0.0
        else:
            self.mono_depth = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(self.data_device).float()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device).float()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # self.world_view_transform: w2c ^ T
        # self.projection_matrix: p^T
        # full_proj_transform: P = K * w2c, note this is [-1, 1] space.

    def move_to_device(self):
        # move all tensors to device
        self.original_image = self.original_image.to(self.data_device)
        self.world_view_transform = self.world_view_transform.to(self.data_device)
        self.projection_matrix = self.projection_matrix.to(self.data_device)
        self.full_proj_transform = self.full_proj_transform.to(self.data_device)
        self.camera_center = self.camera_center.to(self.data_device)

    def refactor(self, factor, fov_scale=None):

        if fov_scale is not None:
            self.FoVy = focal2fov(fov2focal(self.FoVy, self.image_height) * fov_scale, self.image_height)
            self.FoVx = focal2fov(fov2focal(self.FoVx, self.image_width)  * fov_scale, self.image_width)

            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.factor = factor
        self.image_width = int(self.factor*self.original_image.shape[2])
        self.image_height = int(self.factor*self.original_image.shape[1])


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class Render_Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, image, gt_alpha_mask=None, mono_depth=None, loss_mask=None,
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 white_background=False, distance=None
                 ):
        super(Render_Camera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.distance = distance

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.mask = gt_alpha_mask.to(self.data_device)
            self.original_image[(self.mask <= 0.5).expand_as(self.original_image)] = 1.0 if white_background else 0.0
        else:
            self.mask = None

        # Photometric-loss mask: 1 = supervise, 0 = ignore. Never touches
        # original_image -- see the note in Camera above.
        if loss_mask is not None:
            self.loss_mask = loss_mask.to(self.data_device)
            if self.loss_mask.dim() == 2:
                self.loss_mask = self.loss_mask[None]
        else:
            self.loss_mask = None

        if mono_depth is not None:
            self.mono_depth = mono_depth.to(self.data_device)
            # self.mono_depth[(self.mask <= 0.5)] = 0.0
        else:
            self.mono_depth = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans.float()
        self.scale = scale

        self.world_view_transform = getWorld2View2_tensor(R, T).transpose(0, 1).to(self.data_device).float()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device).float()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def refactor(self, factor, fov_scale=None):

        if fov_scale is not None:
            self.FoVy = focal2fov(fov2focal(self.FoVy, self.image_height) * fov_scale, self.image_height)
            self.FoVx = focal2fov(fov2focal(self.FoVx, self.image_width)  * fov_scale, self.image_width)

            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
            self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.factor = factor
        self.image_width = int(self.factor*self.original_image.shape[2])
        self.image_height = int(self.factor*self.original_image.shape[1])
