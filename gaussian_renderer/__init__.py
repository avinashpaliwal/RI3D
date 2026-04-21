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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, fixed_pc : GaussianModel = None, clean_indices : torch.Tensor = None, test=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if test:
        pool_op = lambda x: x
    else:
        viewpoint_camera.refactor(2)
        pool_op = torch.nn.AvgPool2d(2).to(pc.device)

    # print(clean_indices, clean_indices.shape, fixed_pc.get_xyz.shape)
    if fixed_pc is not None:
        xyz      = torch.cat((pc.get_xyz, fixed_pc.get_xyz), dim=0) if clean_indices is None else torch.cat((pc.get_xyz, fixed_pc.get_xyz[clean_indices]), dim=0)
        opacity  = torch.cat((pc.get_opacity, fixed_pc.get_opacity), dim=0) if clean_indices is None else torch.cat((pc.get_opacity, fixed_pc.get_opacity[clean_indices]), dim=0)
        scaling  = torch.cat((pc.get_scaling, fixed_pc.get_scaling), dim=0) if clean_indices is None else torch.cat((pc.get_scaling, fixed_pc.get_scaling[clean_indices]), dim=0)
        rotation = torch.cat((pc.get_rotation, fixed_pc.get_rotation), dim=0) if clean_indices is None else torch.cat((pc.get_rotation, fixed_pc.get_rotation[clean_indices]), dim=0)
        features = torch.cat((pc.get_features, fixed_pc.get_features), dim=0) if clean_indices is None else torch.cat((pc.get_features, fixed_pc.get_features[clean_indices]), dim=0)
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points_pc = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.device) + 0
        try:
            screenspace_points_pc.retain_grad()
        except:
            pass
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points_fpc = torch.zeros_like(fixed_pc.get_xyz, dtype=fixed_pc.get_xyz.dtype, requires_grad=True, device=pc.device) + 0
        screenspace_points = torch.cat((screenspace_points_pc, screenspace_points_fpc), dim=0)
    else:
        xyz      = pc.get_xyz #if clean_indices is None else pc.get_xyz[clean_indices]
        opacity  = pc.get_opacity #if clean_indices is None else pc.get_opacity[clean_indices]
        scaling  = pc.get_scaling #if clean_indices is None else pc.get_scaling[clean_indices]
        rotation = pc.get_rotation #if clean_indices is None else pc.get_rotation[clean_indices]
        features = pc.get_features #if clean_indices is None else pc.get_features[clean_indices]
        if clean_indices is not None:
            with torch.no_grad():
                opacity_copy = opacity.detach().clone()
                opacity_copy_copy = opacity_copy.clone()
                opacity_copy.index_fill_(0, clean_indices, 0)
                opacity.data = opacity_copy
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.device) + 0

        try:
            screenspace_points.retain_grad()
        except:
            pass
    
    # print(xyz.shape, opacity.shape, scaling.shape, rotation.shape, features.shape)
    # print(pc.get_xyz.shape)
        # print('hjeere')
    
    # exit()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points
    # opacity = opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = None#pc.get_covariance(scaling_modifier)
    else:
        scales = scaling
        rotations = rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)[:4]
    
    if clean_indices is not None:
        opacity.data = opacity_copy_copy
    # print(screenspace_points_pc.grad, means3D.grad, opacity.grad, scaling.grad, rotation.grad, features.grad)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(rendered_image.device, pc.device)
    return {"render": pool_op(rendered_image),
            "viewspace_points": screenspace_points_pc if fixed_pc is not None else screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rendered_depth": pool_op(rendered_depth), # depth
            "rendered_alpha": pool_op(rendered_alpha), # acc
    }
