import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid
from argparse import ArgumentParser, Namespace
from utils.arguments import ModelParams, PipelineParams, OptimizationParams
from random import randint
import json

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
from tqdm import tqdm
from torchmetrics.functional.regression import pearson_corrcoef

from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, monodisp, masked_l1_loss, masked_ssim
from utils.image_utils import psnr
from gaussian_renderer import render
from scene import Scene, GaussianModel

# try:
#     from torch.utils.tensorboard.writer import SummaryWriter
#     TENSORBOARD_FOUND = True
# except ImportError:
#     TENSORBOARD_FOUND = False

def leave_one_out_training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, train_id):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, args.sparse_view_num, spherical_gaussians=True)
    scene = Scene(dataset, gaussians, shuffle=False, extra_opts=args) # make sure no shuffle
    gaussians.load_ply(args.ply_path)
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

    num_id, image_id = train_id
    viewpoint_stack = None

    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    
    white_bg = torch.ones_like(background).cuda()
    black_bg = torch.zeros_like(background).cuda()

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record() # type: ignore

        if iteration < opt.densify_until_iter:
            gaussians.update_learning_rate(iteration)
        else:
            gaussians.reset_learning_rates(opt)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 10000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            if iteration <= opt.densify_until_iter:
                viewpoint_stack = scene.getTrainCameras().copy()[:num_id] + scene.getTrainCameras().copy()[num_id+1:] # leave one out
            else:
                viewpoint_stack = scene.getTrainCameras().copy() # after the densify_until_iter, use all the cameras
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # bg = torch.rand((3), device="cuda") if opt.random_background else background
        # bg = white_bg if randint(0, 1) else black_bg
        bg = white_bg

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        loss, Ll1 = cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg)

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

            # Log
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            # Save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity(val=0.01)

                # # if iteration % (opt.opacity_reset_interval // 2) == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                # if iteration % opt.remove_outliers_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.remove_outliers(opt, iteration, linear=True)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # cache the rendered images in the rendering progress.
            if iteration > opt.densify_until_iter and not viewpoint_stack: 
                infer_cam = scene.getTrainCameras().copy()[num_id]
                # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                # bg_color = [1, 0, 1]
                background = torch.tensor(bg, dtype=torch.float32, device="cuda")
                rendering = render(infer_cam, gaussians, pipe, background)["render"]
                gt = infer_cam.original_image[0:3, :, :]
                render_path = os.path.join(args.model_path, 'left_image')
                os.makedirs(render_path, exist_ok=True)
                torchvision.utils.save_image(rendering, os.path.join(render_path, f'sample_{iteration}.png'))
                if not os.path.exists(os.path.join(args.model_path, 'gt.png')):
                    torchvision.utils.save_image(gt, os.path.join(args.model_path, 'gt.png'))

    ### in the end, we need store the gaussians.cache for latter use
    torch.save(gaussians.cache, os.path.join(args.model_path, 'gaussians_cache.pth'))

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
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

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
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid"):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.
    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)
    Optional: [silhouette loss, monodepth loss]
    """
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()
    # loss_mask is 1 = supervise, 0 = ignore; carries burned-in watermarks so they
    # are not learned as scene content. See scene/dataset_readers_flow.py.
    loss_mask = getattr(viewpoint_cam, "loss_mask", None)
    Ll1 = masked_l1_loss(image, gt_image, loss_mask)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - masked_ssim(image, gt_image, loss_mask))

    # if hasattr(args, "use_mask") and args.use_mask:
    #     if silhouette_loss_type == "bce":
    #         silhouette_loss = F.binary_cross_entropy(render_pkg["rendered_alpha"], viewpoint_cam.mask)
    #     elif silhouette_loss_type == "mse":
    #         silhouette_loss = F.mse_loss(render_pkg["rendered_alpha"], viewpoint_cam.mask)
    #     else:
    #         raise NotImplementedError
    #     loss = loss + opt.lambda_silhouette * silhouette_loss
        
    if hasattr(viewpoint_cam, "mono_depth") and  viewpoint_cam.mono_depth is not None:
        if mono_loss_type == "mid":
            disp_mono = 1 / viewpoint_cam.mono_depth.clamp(1e-6).reshape(-1) # shape: [N]
            disp_render = 1 / render_pkg["rendered_depth"].clamp(1e-6).reshape(-1) # shape: [N]
            depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]
        elif mono_loss_type == "pearson":
            # mono_depth is metric depth in SfM world units, so it correlates
            # directly with the render. The old -depth / 1/(depth+200) pair were
            # two ways of flipping Depth-Anything disparity into a depth-like
            # ordering, and both anti-correlate now.
            mono_depth = viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6)
            rendered_depth = render_pkg["rendered_depth"][viewpoint_cam.mask > 0.5].clamp(1e-6)
            depth_loss = 1 - pearson_corrcoef(mono_depth, rendered_depth)
        else:
            raise NotImplementedError

        loss = loss + args.mono_depth_weight * depth_loss

    return loss, Ll1

def train_3dgs(args, ids):
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    pipeline = pp.extract(args)
    model_path_root = args.model_path
    
    for num_id, image_id in zip(range(args.sparse_view_num), ids): # num_id: leave one out id, image_id: the id of the image to be infered
        args.model_path = os.path.join(model_path_root, f'leave_{image_id}')
        dataset.model_path = args.model_path
        os.makedirs(args.model_path, exist_ok=True)
        leave_one_out_training(args,
                                dataset,
                                op.extract(args),
                                pipeline,
                                args.test_iterations,
                                args.save_iterations,
                                args.checkpoint_iterations,
                                args.start_checkpoint,
                                args.debug_from,
                                train_id = (num_id, image_id))


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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_0000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[6000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                    help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                    else use dense view. In sparse setting, sparse views will be used as training data, \
                    others will be used as testing data.")
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")
    parser.add_argument("--init_pcd_name", default='origin', type=str, help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument('--mono_depth_weight', type=float, default=0.005, help="The rate of monodepth loss")
    parser.add_argument('--ply_path', type=str, help="path to the ply file")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    args.is_renderrr = False
    args.scaling_lr = 0
    print(args)
    
    # bg_color_dict = {'bonsai': [1, 1, 0],  'bicycle': [1, 0, 1], 'counter': [1, 0, 1],
    #                  'flowers': [0, 0, 1], 'garden': [1, 0, 1],  'kitchen': [0, 0, 1],
    #                  'room': [1, 0, 1],    'stump': [1, 0, 1],   'treehill': [1, 0, 1]}
    # bg_color_ren = bg_color_dict[args.source_path.split('/')[-1]]

    assert args.sparse_view_num > 0, 'leave_one_out is for sparse view training'
    assert os.path.exists(os.path.join(args.source_path, f"train_test_split_{args.sparse_view_num}.json")), f"sparse_{args.sparse_view_num}.txt not found!"
    # assert os.path.exists(os.path.join(args.source_path, f"visual_hull_{args.sparse_view_num}.ply")), f"visual_hull_{args.sparse_view_num}.ply not found!"

    # ids = np.loadtxt(os.path.join(args.source_path, f"train_test_split_{args.sparse_view_num}.json"), dtype=np.int32).tolist()
    
    with open(f'{args.source_path}/train_test_split_{args.sparse_view_num}.json') as json_data:
        data = json.load(json_data)
        ids = data['train_ids']

    train_3dgs(args, ids)

    # All done
    print("\nAll training complete.")
