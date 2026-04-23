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
import torch.nn.functional as F
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from utils.graphics_utils import BasicPointCloud, z_score_from_percentage
from utils.general_utils import strip_symmetric, build_scaling_rotation
from kornia.geometry.depth import depth_to_3d
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, transform_pcd, getWorld2View, geom_transform_points
from scene.xfields import XfieldsFlow
from scene.posenc import get_embedder
from scene.cameras import Camera

from torch.optim.lr_scheduler import CosineAnnealingLR

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, num_cameras : int = 0, mono_d_so_enable : bool = False, spherical_gaussians : bool = False, enable_learned_opac : bool = False, device: str = 'cuda'):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._backup_attributes = {}
        self.setup_functions()

        self.device = device
        self.mono_d_so_enable = mono_d_so_enable#False
        self.train = False

        if mono_d_so_enable:
            self.num_cameras = num_cameras
            self.monodepth_scaling = torch.nn.Parameter(torch.ones(self.num_cameras).to(self.device).requires_grad_(True))
            self.monodepth_offset = torch.nn.Parameter(torch.zeros(self.num_cameras).to(self.device).requires_grad_(True))
            self.monoso_optimizer = torch.optim.Adam([{'params': [self.monodepth_scaling], 'lr': 0.1}, {'params': [self.monodepth_offset], 'lr': 0.5}])
            self.monoso_scheduler = CosineAnnealingLR(self.monoso_optimizer, T_max=10000)

        
        self.num_cameras = num_cameras
        self.radius_mult = 1.3

        self.spherical_gaussians = spherical_gaussians

        
        # inp = np.linspace(-1, 1, num=self.num_cameras, dtype=np.float32)
        embed_fn = get_embedder(10)
        inp = np.linspace(0, 1, num=self.num_cameras, dtype=np.float32)
        inp = embed_fn(inp)
        self.inp = torch.tensor(inp)[..., None, None].to(self.device)


        # self.enable_decoder = True
        self.enable_learned_opac = enable_learned_opac

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def cache(self):
        return [self._xyz.detach(),
                self._features_dc.detach(),
                self._features_rest.detach() if self._features_rest.numel() > 0 else torch.empty(0),
                self._scaling.detach(),
                self._rotation.detach(),
                self._opacity.detach()]

    @property
    def get_scaling(self):
        if self.spherical_gaussians:
            scaling = self._scaling[:, :1].repeat(1, 3)
            return self.scaling_activation(scaling)
        else:
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    

    def set_ref_camera(self, ref_camera, ngf=10, outChannels=5):
        _, self.height, self.width = ref_camera.original_image.shape
        self.ref_camera = ref_camera

        self.decoder = XfieldsFlow(self.height, self.width, ngf=8, inChannels=20, outChannels=outChannels).to(self.device)
        self.decoder_optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=1e-6)
        self.decoder_scheduler = torch.optim.lr_scheduler.CyclicLR(self.decoder_optimizer, base_lr=1e-6, max_lr=1e-4, mode='triangular2', step_size_up=1000, cycle_momentum=False)

        self.decoder_opac = XfieldsFlow(self.height, self.width, ngf=5, inChannels=20, outChannels=outChannels).to(self.device)
        self.decoder_opac_optimizer = torch.optim.AdamW(self.decoder_opac.parameters(), lr=1e-6)
        self.decoder_opac_scheduler = torch.optim.lr_scheduler.CyclicLR(self.decoder_opac_optimizer, base_lr=1e-6, max_lr=1e-5, mode='triangular2', step_size_up=2000, cycle_momentum=False)

    
    def get_clean_indices(self, camera, render=False, scale=1):
        
        if render:
            threshold = camera.distance * scale
            world_view_transform = torch.tensor(getWorld2View(camera.R, camera.T)).transpose(0, 1).to(self.device).float()
        else:
            threshold = (camera.distance * scale).to(self.device)
            world_view_transform = torch.tensor(getWorld2View(camera.R.cpu().numpy(), camera.T.cpu().numpy())).transpose(0, 1).to(self.device).float()
        depth = geom_transform_points(self.get_xyz, world_view_transform)[:, 2]
        
        # depth[depth < 0] = 100
        mask = torch.where(depth < threshold, 1, 0)

        clean_indices = mask.nonzero()

        # print(floater_indices, floater_indices.shape, depth.shape, mask.shape, depth[:2])
        return clean_indices[:, 0]


    def remove_floaters(self, min_depth: float, cameras, scene_center):
        threshold = 0.4 * min_depth
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        
        masks = []
        for camera in cameras:
            
            world_view_transform = torch.tensor(getWorld2View(camera.R, camera.T)).transpose(0, 1).to(self.device).float()
            camera_center = world_view_transform.inverse()[3, :3]
            diff = (self.get_xyz - camera_center)
            z_dir = (scene_center - camera_center)
            dist = torch.pow(torch.pow(diff, 2).sum(dim=1), 0.5)
            # dotp = (diff * z_dir).sum(dim=1)
            # dist[dotp < 0] = 100
            # print(z_dir.shape, diff.shape, dotp.shape)
            # exit()
            # print(dist.max(), dist.min(), dist[:5], dotp[:5], z_dir[:5], dist.shape)
            mask = torch.where(dist < threshold, True, False)


            # world_view_transform = torch.tensor(getWorld2View(camera.R, camera.T)).transpose(0, 1).to(self.device).float()
            # depth = geom_transform_points(geom_transform_points(self.get_xyz, world_view_transform), camera.
            # # print(depth.max(), depth.min(), depth[:5], depth.shape)

            # depth[depth < 0] = 100
            # mask = torch.where(depth < threshold, True, False)

            # print(depth.max(), depth.min(), depth[:5], mask[:5], depth.shape)
            # print(mask.shape)
            # print((self.get_xyz - camera.camera_center)[:5], dist[:5], mask[:5])
            # exit()
            masks.append(mask)

        # masks = torch.stack(masks, dim=1)
        # print(masks.shape, masks.nonzero().shape)
        # masks = masks.sum(dim=1)
        # print(masks.shape, masks.nonzero().shape)
        # mask = torch.clamp(masks, 0, 1)
        # print(mask.shape, mask.nonzero().shape, min_depth)
        # print(mask[:5])
        # exit()
        mask = masks[0]
        for m in masks:

            mask = torch.logical_or(mask, m)
        print(f"Removed {mask.sum()} floaters")

        self.prune_points(mask)


    def get_residual(self):

        delta = (self.decoder(self.inp))
        # delta = delta * self.dep_mask / (torch.sum(self.dep_mask, dim=1, keepdim=True) + 1e-10)
        # delta = torch.sum(delta, dim=1, keepdim=True)

        delta = 5 * delta.reshape(-1, 1).contiguous()#[..., None]

        return delta
    
    
    def get_learned_opac(self):

        # opac = F.tanh(F.relu(self.decoder_opac(self.inp)))
        opac = (self.decoder_opac(self.inp))
        # opac = opac * self.dep_mask / (torch.sum(self.dep_mask, dim=1, keepdim=True) + 1e-10)
        # opac = torch.sum(opac, dim=1, keepdim=True)

        opac = 100 * opac.reshape(-1, 1).contiguous()#[..., None]

        return opac

    @property
    def get_xyz(self):
        if not self.train:
            return self._xyz

        else:
        
            if self.mono_d_so_enable:
                # scaling = self.monodepth_scaling.repeat_interleave(self.height*self.width)[..., None]
                # offset = self.monodepth_offset.repeat_interleave(self.height*self.width)[..., None]

                # z = self._z * scaling + offset + self.get_residual()
                
                z = self._z #+ self.get_residual()
                
                xyz_homo = torch.cat((self._xy * z, z, torch.ones_like(z)), dim=1)[..., None]
                # xyz_world = xyz_homo[..., :3, 0]
                xyz_world_homo = torch.bmm(self.c2w, xyz_homo)
                xyz_world = (xyz_world_homo[:, :3] / xyz_world_homo[:, 3:])[..., 0]

                return xyz_world
            
            else:
                return self._xyz

    
    
    @property
    def get_z(self):
        # return self._xyz

        
        if self.mono_d_so_enable:
            # scaling = self.monodepth_scaling.repeat_interleave(self.height*self.width)[..., None]
            # offset = self.monodepth_offset.repeat_interleave(self.height*self.width)[..., None]

            # z = self._z * scaling + offset
            
            z = self._z + self.get_residual()
        else:
            z = self._z

        return z
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        if self.enable_learned_opac:
            opacity = self.get_learned_opac() + self._opacity
            ret_opac = self.opacity_activation(opacity)

            return ret_opac
        else:
            return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1.0):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, train_cameras : list, flows : list):
        
        self.flows, self.masks, radii2 = flows

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        # print(pcd.colors[:10], fused_color[:10])
        # exit()
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0


        # print("Number of points at initialisation : ", fused_point_cloud.shape[0], flows.shape)
        print(fused_point_cloud.shape, features.shape)

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(self.device)), 0.0000001)
        dist2 = torch.tensor(radii2).float().to(self.device)
        # dist2 = torch.tensor(np.asarray(flows)).float().to(self.device)torch.log(torch.sqrt(radii2) * radius_mult)
        scales = torch.log(torch.sqrt(dist2) * 1.4)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        # opacities = inverse_sigmoid(0.99 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        if self.mono_d_so_enable:

            w2cs = torch.stack([x.world_view_transform.transpose(0, 1) for x in train_cameras], dim=0)
            c2w = w2cs.inverse().clone().repeat_interleave(fused_point_cloud.shape[0]//self.num_cameras, dim=0)

            self._xy = (fused_point_cloud[:, :2] / fused_point_cloud[:, 2:]).contiguous()
            self._z = fused_point_cloud[:, 2:].contiguous()
        else:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))


        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self._features_dc.shape[0]), device=self.device)


        self.flows = torch.tensor(self.flows).float().to(self.device)
        self.masks = torch.tensor(self.masks).float().to(self.device)

        if self.mono_d_so_enable:
            self.c2w = c2w.contiguous()
            C, H, W = train_cameras[0].original_image.shape
            self.height = H
            self.width  = W

            
            # self.sparse_dep = sparse_dep
            # sparse_dep_mask = [torch.zeros_like(x).to(self.device) for x in sparse_dep]
            # for x, y in zip(sparse_dep, sparse_dep_mask):
            #     y[x != 0] = 1
            # self.sparse_dep_mask = sparse_dep_mask


        self.train = True
        # print(self.width, self.height)
        # exit(0)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._features_dc.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self._features_dc.shape[0], 1), device=self.device)

        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if not self.mono_d_so_enable:
            l.append({'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)


    def reset_learning_rates(self, training_args):

        l = [
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': 0, "name": "scaling"},
            {'params': [self._rotation], 'lr': 0, "name": "rotation"}
        ]

        if not self.mono_d_so_enable:
            l.append({'params': [self._xyz], 'lr': 0, "name": "xyz"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups: # pyright: ignore[reportOptionalMemberAccess]
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            

    def set_learning_rate_xyz(self, lr_xyz=None, lr_scaling=None):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups: # pyright: ignore[reportOptionalMemberAccess]
            if param_group["name"] == "xyz":
                param_group['lr'] = lr_xyz
                # return lr
            
            if param_group["name"] == "scaling":
                param_group['lr'] = lr_scaling
                # return lr

    def update_spatial_lr_scale(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.scaling_inverse_activation(self.get_scaling).detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)#[self.masks.cpu().numpy().reshape(-1) > 0]
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, val=0.1):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*val))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reduce_opacity(self, val=0.6):
        opacities_new = inverse_sigmoid(torch.max(self.get_opacity*val, torch.ones_like(self.get_opacity)*0.1))
        # opacities_new = inverse_sigmoid(self.get_opacity*0.6)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
    def load_ply(self, path, requires_grad: bool = True, opac_offset: float = 0.0):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis] + opac_offset

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(requires_grad))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(requires_grad))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=self.device).transpose(1, 2).contiguous().requires_grad_(requires_grad))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(requires_grad))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=self.device).requires_grad_(requires_grad))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(requires_grad))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

        self.active_sh_degree = self.max_sh_degree

        
    def load_init_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        return xyz, SH2RGB(features_dc)
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def depth_densify(self, blended_depth, inpainted_image, inpaint_mask, camera, enable_postfix=True, opac=0.999):

        inpainted_image = torch.tensor(inpainted_image / 255.).float().to(self.device).reshape(-1, 3)
        inpaint_mask = torch.tensor(inpaint_mask[..., 0]).float().to(self.device).reshape(-1, 1)
        print(inpaint_mask.shape, inpainted_image.shape)
        inpaint_mask = inpaint_mask.nonzero()[:, 0]
        print(inpaint_mask.shape, inpainted_image.shape)
        
        dep = torch.tensor(blended_depth).float()[None, None].to(self.device)

        # torch.save(torch.index_select(dep.reshape(-1, 1), 0, inpaint_mask), 'inp_dep.pt')
        # exit()


        K = torch.eye(3)[None].to(self.device)
        K[:, 0, 0] = fov2focal(camera.FoVx, camera.image_width)
        K[:, 0, 2] = camera.image_width / 2.0
        K[:, 1, 1] = fov2focal(camera.FoVy, camera.image_height)
        K[:, 1, 2] = camera.image_height / 2.0

        print("camera: ..... ", K)

        camera3d = depth_to_3d(dep, K)
        xyz_cam = camera3d[0].permute(1, 2, 0).reshape(-1, 3)
        xyz_cam_homo = torch.cat((xyz_cam, torch.ones_like(xyz_cam[:, :1])), dim=1)[..., None]
        c2w = camera.world_view_transform.transpose(0,1).inverse()[None]
        xyz_world_homo = torch.matmul(c2w, xyz_cam_homo)
        xyz_world = (xyz_world_homo[:, :3] / xyz_world_homo[:, 3:])[..., 0]
        xyz_world = torch.index_select(xyz_world, 0, inpaint_mask)

        rgb = torch.index_select(inpainted_image, 0, inpaint_mask)
        rgb = RGB2SH(rgb)
        features = torch.zeros((rgb.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(self.device)
        features[:, :3, 0 ] = rgb
        features[:, 3:, 1:] = 0.0

        radii = np.tan(0.5 * float(camera.FoVy))  * dep / camera.image_height
        radii2 =  torch.index_select((radii**2).reshape(-1, 1), 0, inpaint_mask)[:, 0]
        # dist2 = torch.clamp_min(distCUDA2(xyz_world), 0.0000001)
        scales = torch.log(torch.sqrt(radii2))[...,None].repeat(1, 3)
        rots = torch.zeros((xyz_world.shape[0], 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(opac * torch.ones((xyz_world.shape[0], 1), dtype=torch.float, device=self.device))
        
        new_xyz = xyz_world
        new_features_dc = features[:,:,0:1].transpose(1, 2).contiguous()
        new_features_rest = features[:,:,1:].transpose(1, 2).contiguous()
        new_opacities = opacities
        new_scaling = scales
        new_rotation = rots

        print(opacities.shape)

        if enable_postfix:
            print("Postfixing")
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask], device=self.device).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity <= min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size # too big on screen
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent # scale too big
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        print(f'grads.max(): {grads.max()}')
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        print("Pruning: {} points removed".format(prune_mask.sum()))
        # print(prune_mask[:5], prune_mask.shape, prune_mask.max(), prune_mask.min())
        # exit()
        self.prune_points(prune_mask)

    def remove_outliers(self, opt, step, linear=False, removing_ratio=0, remaining_rate=0.0):
        """Assume as Gaussian distribution
        removing_ratio: the removing_ratio of removing points
        remain_rate: the ratio of remaining points in ratio
        """
        xyz = self.get_xyz.detach()
        num_points = xyz.shape[0]
        if not linear:
            lambda_sigma = float(z_score_from_percentage(removing_ratio)) if removing_ratio > 0 else 1 # default 1
        else: 
            # in this case, we believe that the pointcloud are more and more dense,
            # so the std_nearest_k_distance + mean_nearest_k_distance are smaller and smaller
            # thus the lambda_sigma should be bigger and bigger, from 1 -> z_score_from_percentage(1)
            iter_start = opt.densify_from_iter
            iter_end = opt.densify_until_iter
            init_lambda_sigma = 1
            final_lambda_sigma = float(z_score_from_percentage(1))
            lambda_sigma = init_lambda_sigma + (final_lambda_sigma - init_lambda_sigma) * (step - iter_start) / (iter_end - iter_start)
        K = int(num_points**0.5)
        # Chunked KNN to avoid OOM: each chunk allocates [chunk_size, N] floats
        mem_per_row = num_points * 4  # 4 bytes per float32
        chunk_size = max(1, min(4096, int(2e9 / mem_per_row)))  # target ~2GB per chunk
        nearest_k_distance = torch.empty(num_points, K, device=xyz.device)
        for i in range(0, num_points, chunk_size):
            end = min(i + chunk_size, num_points)
            nearest_k_distance[i:end] = torch.cdist(xyz[i:end], xyz, p=2).pow(2).topk(K, dim=-1, largest=False).values
        nearest_k_distance = nearest_k_distance.unsqueeze(0)
        mean_nearest_k_distance, std_nearest_k_distance = nearest_k_distance.mean(), nearest_k_distance.std()
        mask = nearest_k_distance.mean(dim = -1) >= (mean_nearest_k_distance + lambda_sigma * std_nearest_k_distance)
        mask = mask.squeeze()

        if remaining_rate > 0: # since the gs cannot generate point from None, we may need to keep some points incase of empty
            num_remove_points = mask.sum()
            num_remain_points = int(num_remove_points * remaining_rate)
            remove_indices = torch.where(mask)[0]
            # Randomly select indices to keep
            keep_indices = remove_indices[torch.randperm(remove_indices.size(0))[:num_remain_points]]
            # Update the mask: set selected points to False (to keep)
            mask[keep_indices] = False

        self.prune_points(mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # gsplat stores absolute gradients in .absgrad [C, N, 2] after backward
        grad = getattr(viewspace_point_tensor, 'absgrad', None)
        if grad is None:
            grad = viewspace_point_tensor.grad
        if grad is not None:
            if grad.dim() == 3:
                grad = grad[0]  # [C, N, 2] -> [N, 2]
            self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def add_densification_stats_no_grad(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def restore_noise(self):
        # Restore the attributes from backup
        for attr, value in self._backup_attributes.items():
            setattr(self, attr, value)

    def add_statistics_noise(self, statistics_info, noise_dropout: float = 0., std_scale: float = 0.7):
        # List of attributes to add noise to '_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'
        attributes_to_noise = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']

        # get the means and vars of statistics_info
        means, stds = {}, {}

        for key in statistics_info[0].keys():
            means[key] = np.mean([info[key][0] for info in statistics_info], axis=0)
            stds[key] = np.mean([info[key][1] for info in statistics_info], axis=0)

        # Backup and add noise
        for attr in attributes_to_noise:
            self._backup_attributes[attr] = getattr(self, attr)
            if attr in ['_xyz', '_scaling', '_rotation', '_opacity']:
                cur_std = torch.from_numpy(stds[attr]).float().to(self._backup_attributes[attr].device)
                cur_mean = torch.from_numpy(means[attr]).float().to(self._backup_attributes[attr].device)

                # generate noise with mean and std
                noise = torch.randn_like(getattr(self, attr)) * cur_std + cur_mean
                if attr == '_scaling' or attr == '_opacity' or attr == '_rotation':
                    noise = torch.clamp(noise, cur_mean - std_scale * cur_std, cur_mean + std_scale * cur_std)

                noise[torch.rand_like(getattr(self, attr)) < noise_dropout] = 0
                setattr(self, attr, getattr(self, attr) + noise)
