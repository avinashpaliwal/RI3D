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
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import fov2focal

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, fixed_pc : GaussianModel = None, clean_indices : torch.Tensor = None, test=False):
    """
    Render the scene using gsplat.

    Background tensor (bg_color) must be on GPU!
    """
    if test:
        pool_op = lambda x: x
    else:
        viewpoint_camera.refactor(2)
        pool_op = torch.nn.AvgPool2d(2).to(pc.device)

    if fixed_pc is not None:
        xyz      = torch.cat((pc.get_xyz, fixed_pc.get_xyz), dim=0) if clean_indices is None else torch.cat((pc.get_xyz, fixed_pc.get_xyz[clean_indices]), dim=0)
        opacity  = torch.cat((pc.get_opacity, fixed_pc.get_opacity), dim=0) if clean_indices is None else torch.cat((pc.get_opacity, fixed_pc.get_opacity[clean_indices]), dim=0)
        scaling  = torch.cat((pc.get_scaling, fixed_pc.get_scaling), dim=0) if clean_indices is None else torch.cat((pc.get_scaling, fixed_pc.get_scaling[clean_indices]), dim=0)
        rotation = torch.cat((pc.get_rotation, fixed_pc.get_rotation), dim=0) if clean_indices is None else torch.cat((pc.get_rotation, fixed_pc.get_rotation[clean_indices]), dim=0)
        features = torch.cat((pc.get_features, fixed_pc.get_features), dim=0) if clean_indices is None else torch.cat((pc.get_features, fixed_pc.get_features[clean_indices]), dim=0)
        screenspace_points_pc = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.device) + 0
        try:
            screenspace_points_pc.retain_grad()
        except:
            pass
        screenspace_points_fpc = torch.zeros_like(fixed_pc.get_xyz, dtype=fixed_pc.get_xyz.dtype, requires_grad=True, device=pc.device) + 0
        screenspace_points = torch.cat((screenspace_points_pc, screenspace_points_fpc), dim=0)
    else:
        xyz      = pc.get_xyz
        opacity  = pc.get_opacity
        scaling  = pc.get_scaling
        rotation = pc.get_rotation
        features = pc.get_features
        if clean_indices is not None:
            with torch.no_grad():
                opacity_copy = opacity.detach().clone()
                opacity_copy_copy = opacity_copy.clone()
                opacity_copy.index_fill_(0, clean_indices, 0)
                opacity.data = opacity_copy
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    # gsplat expects viewmats as [C, 4, 4] W2C (not transposed)
    viewmat = viewpoint_camera.world_view_transform.T[None]  # [1, 4, 4]

    # Build intrinsics matrix from FoV
    fx = fov2focal(viewpoint_camera.FoVx, W)
    fy = fov2focal(viewpoint_camera.FoVy, H)
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=xyz.device)[None]  # [1, 3, 3]

    # Prepare colors: SH coefficients or precomputed
    if override_color is not None:
        colors = override_color
        sh_degree_to_use = None
    elif pipe.convert_SHs_python:
        shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        dir_pp = (xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
        sh_degree_to_use = None
    else:
        # gsplat expects SH coeffs as [N, K, 3] where K = (degree+1)^2
        colors = features  # already [N, K, 3]
        sh_degree_to_use = pc.active_sh_degree

    # gsplat expects opacities as [N] not [N, 1]
    opacities_flat = opacity.squeeze(-1)

    render_colors, render_alphas, meta = rasterization(
        means=xyz,
        quats=rotation,
        scales=scaling * scaling_modifier,
        opacities=opacities_flat,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=sh_degree_to_use,
        backgrounds=bg_color[None],
        render_mode="RGB+D",
        packed=False,
        absgrad=True,
    )

    # render_colors: [1, H, W, 4] (RGB + depth in last channel)
    # render_alphas: [1, H, W, 1]
    rendered_image = render_colors[0, :, :, :3].permute(2, 0, 1)  # [3, H, W]
    rendered_depth = render_colors[0, :, :, 3:4].permute(2, 0, 1)  # [1, H, W]
    rendered_alpha = render_alphas[0, :, :, 0:1].permute(2, 0, 1)  # [1, H, W]

    # gsplat returns radii as [C, N, 2] (x/y radii per camera); take max and flatten to [N]
    radii = meta.get("radii", torch.zeros(xyz.shape[0], dtype=torch.int32, device=xyz.device))
    if radii.dim() == 3:
        radii = radii[0].amax(dim=-1)  # [C, N, 2] -> [N]
    elif radii.dim() == 2:
        radii = radii[0]  # [C, N] -> [N]

    # gsplat stores absgrad on meta["means2d"] after backward; pass the original tensor through
    # so consumers can access .absgrad for densification
    means2d_meta = meta.get("means2d", None)
    viewspace_points_out = means2d_meta if means2d_meta is not None else screenspace_points

    if clean_indices is not None:
        opacity.data = opacity_copy_copy

    return {"render": pool_op(rendered_image),
            "viewspace_points": viewspace_points_out,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rendered_depth": pool_op(rendered_depth),
            "rendered_alpha": pool_op(rendered_alpha),
    }
