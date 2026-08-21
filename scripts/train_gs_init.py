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

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
from argparse import ArgumentParser, Namespace
from random import randint, shuffle
from typing import Optional
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics.functional.regression import pearson_corrcoef
from utils.arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, monodisp, l2_loss
from torch.utils.tensorboard.writer import SummaryWriter

TENSORBOARD_FOUND = True

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, args.sparse_view_num, mono_d_so_enable=True, enable_learned_opac=True)
    scene = Scene(dataset, gaussians, extra_opts=args)
    gaussians.training_setup(opt)
    if checkpoint:
        try:
            (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        except TypeError:
            (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    
    Ll1 = torch.tensor([0])

    viewpoint_stack, augview_stack = None, None
    # Pick a random Camera
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()

    viewpoint_cam = viewpoint_stack[0]
    C, H, W = viewpoint_cam.original_image.shape
    print(viewpoint_cam.original_image.shape)
    gaussians.set_ref_camera(viewpoint_cam, outChannels=1)

    # Watermark pixels back-project to geometry that no loss will ever correct:
    # the photometric losses mask them out, so those Gaussians get zero gradient
    # and are therefore never pushed toward real content *and* never pruned
    # (pruning is opacity-driven, opacity is gradient-driven). They freeze wearing
    # the watermark's colour and float wherever the depth prior put them. Kill
    # them here instead -- masking the loss stops the watermark being corrected,
    # only this stops it being created.
    #
    # Zero the opacity rather than dropping the points: create_from_pcd under
    # mono_d_so_enable=True reshapes by `shape[0] // num_cameras` and needs equal
    # per-view counts (the same reason --depth_conf_thr is unsupported here, see
    # docs/point_map.md 4). Stage 1b's first densify_and_prune then removes them,
    # since it prunes below opacity 0.005.
    if getattr(gaussians, "masks", None) is not None:
        dead = gaussians.masks.reshape(-1) <= 0
        n_dead = int(dead.sum())
        if n_dead:
            gaussians._opacity.data[dead] = -10.0  # sigmoid(-10) ~ 4.5e-5
            print(f"[i] Watermark mask: zeroed opacity on {n_dead:,} of "
                  f"{dead.numel():,} init points ({n_dead / dead.numel() * 100:.2f}%)")

    scene.save(first_iter)
    exit()
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record() # type: ignore
        gaussians.update_learning_rate(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        cam_id = viewpoint_cam.uid

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        loss, Ll1 = cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, tb_writer=tb_writer, iteration=iteration)

        # mask = torch.where(gaussians.masks[cam_id] < 4, 0, 1)
        # loss += l2_loss(render_pkg["rendered_depth"] * mask, gaussians.flows[cam_id] * mask) * 0.0

        # sin_depth = gaussians.get_z.reshape(-1, gaussians.height, gaussians.width)[cam_id]

        # disp_mono = 1/viewpoint_cam.mono_depth.clamp(1e-6).reshape(-1) # shape: [N]
        # disp_render = 1/sin_depth.clamp(1e-6).reshape(-1) # shape: [N]
        # depth_loss = 10 * monodisp(disp_mono, disp_render, 'l1')[-1]
        # # # depth_loss = (1 - pearson_corrcoef(disp_render.reshape(-1), -disp_mono.reshape(-1))).mean()

        # loss += depth_loss #* args.mono_depth_weight * 10

        loss.backward()
        iter_end.record()  # type: ignore

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            num_gauss = len(gaussians._xyz)
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f"{ema_loss_for_log:.{7}f}",  'n': f"{num_gauss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # # Densification
            # if iteration < opt.densify_until_iter and num_gauss < opt.max_num_splats:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            #     if iteration % opt.remove_outliers_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.remove_outliers(opt, iteration, linear=True)

            # Optimizer step
            if iteration < opt.iterations:

                gaussians.decoder_optimizer.step()
                gaussians.decoder_optimizer.zero_grad()
                gaussians.decoder_scheduler.step()

                gaussians.decoder_opac_optimizer.step()
                gaussians.decoder_opac_optimizer.zero_grad()
                gaussians.decoder_opac_scheduler.step()


                # gaussians.monoso_optimizer.step()
                # gaussians.monoso_optimizer.zero_grad()
                # gaussians.monoso_scheduler.step()

                
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
            args.model_path = os.path.join("./output/", unique_str)
        else:
            unique_str = str(uuid.uuid4())
            args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

def cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid", tb_writer=None, iteration=0):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.
    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)
    Optional: [silhouette loss, monodepth loss]
    """
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()
    # if opt.random_background:
    #     gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
    Ll1 = l1_loss(image, gt_image)
    Lssim = (1.0 - ssim(image, gt_image))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
    if tb_writer is not None:
        tb_writer.add_scalar('loss/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('loss/ssim_loss', Lssim, iteration)

    # if hasattr(args, "use_mask") and args.use_mask:
    #     if silhouette_loss_type == "bce":
    #         silhouette_loss = F.binary_cross_entropy(render_pkg["rendered_alpha"], viewpoint_cam.mask)
    #     elif silhouette_loss_type == "mse":
    #         silhouette_loss = F.mse_loss(render_pkg["rendered_alpha"], viewpoint_cam.mask)
    #     else:
    #         raise NotImplementedError
    #     loss = loss + opt.lambda_silhouette * silhouette_loss
    #     if tb_writer is not None:
    #         tb_writer.add_scalar('loss/silhouette_loss', silhouette_loss, iteration)

    if hasattr(viewpoint_cam, "mono_depth") and viewpoint_cam.mono_depth is not None:
        # print('here')
        if mono_loss_type == "mid":
            # we apply masked monocular loss
            disp_mono = 1/viewpoint_cam.mono_depth.clamp(1e-6).reshape(-1) # shape: [N]
            disp_render = 1/render_pkg["rendered_depth"].clamp(1e-6).reshape(-1) # shape: [N]
            # print(viewpoint_cam.mono_depth.shape, render_pkg['rendered_depth'].shape, disp_mono.shape, disp_render.shape)
            depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]
        elif mono_loss_type == "pearson":
            # Both sides are metric depth in SfM world units, so they correlate
            # positively -- the negation here dated from mono_depth holding
            # Depth-Anything disparity.
            depth_mono = viewpoint_cam.mono_depth.clamp(1e-6).reshape(-1, 1) # shape: [N]
            depth_render = render_pkg["rendered_depth"].clamp(1e-6).reshape(-1, 1) # shape: [N]
            depth_loss = (1 - pearson_corrcoef(depth_render, depth_mono)).mean()
        else:
            raise NotImplementedError

        loss = loss + args.mono_depth_weight * depth_loss
        if tb_writer is not None:
            tb_writer.add_scalar('loss/depth_loss', depth_loss, iteration)

    return loss, Ll1

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                        help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                        else use dense view. In sparse setting, sparse views will be used as training data, \
                        others will be used as testing data.")
    parser.add_argument("--use_mask", default=False, help="Use masked image, by default True")
    parser.add_argument("--init_pcd_name", default='origin', type=str, 
                        help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--transform_the_world", action="store_true", help="Transform the world to the origin")
    parser.add_argument('--mono_depth_weight', type=float, default=0.005, help="The rate of monodepth loss")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.is_renderrr = False

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
