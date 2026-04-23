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

import json
import os
from argparse import ArgumentParser
from os import makedirs

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm
import imageio
from matplotlib import cm

from utils.arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel, render
import lpips
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir/ "gt"
            renders_dir = method_dir / "renders"
            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                render = renders[idx].cuda()
                gt = gts[idx].cuda()

                ssims.append(ssim(render, gt))
                psnrs.append(psnr(render, gt))
                lpipss.append(lpips_fn(render, gt))

            print("==FROM 3DGS==")
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                    "PSNR": torch.tensor(psnrs).mean().item(),
                                                    "LPIPS": torch.tensor(lpipss).mean().item()})
            per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                        "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


from utils.depth_layering import get_depth_bins

def get_bgmask(depth_rel, num_bins=5, start=6, start_depth=None):
    print('depth_rel: ', depth_rel.shape, depth_rel.min(), depth_rel.max())

    
    bins = get_depth_bins(depth=depth_rel, num_bins=num_bins)
    # bins = [1 / x for x in bins]
    # bins.reverse()
    
    dep = depth_rel[0, 0]
    if start_depth is not None:
        mask = np.where((dep >= start_depth) & (dep <= bins[-1]), 255, 0).astype(np.uint8) / 255.
    else:
        mask = np.where((dep >= bins[start]) & (dep <= bins[-1]), 255, 0).astype(np.uint8) / 255.

    print('bins: ', bins)

    return mask



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, save_images=True, not_generate_video=False, viewR=None):
    model_path = model_path + postfix
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    ssims = 0.
    psnrs = 0.
    lpipss = 0.
    depths = []
    average_gt_color_rgb = torch.zeros(3).cuda()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        gt = view.original_image[0:3, :, :]

        if name == 'train':
            bgmask = get_bgmask(torch.clamp(view.mono_depth.cpu(), min=1.5e-2)[None], start=3)
            bgmask = torch.tensor(bgmask, device='cuda').float()[None]
            num_values = bgmask.sum(dim=(1, 2))
            sum_values = (bgmask * gt).sum(dim=(1, 2))
            average_gt_color_rgb += sum_values/num_values

        view.distance = viewR.distance
        clean_indices = gaussians.get_clean_indices(view, scale=0.5, render=True)
        render_pkg = render(view, gaussians, pipeline, background, clean_indices=clean_indices)
        rendering = render_pkg["render"]
        depths.append(render_pkg["rendered_depth"].cpu().numpy()[0])
        ssims += ssim(rendering, gt).mean().item()
        psnrs += psnr(rendering, gt).mean().item()
        lpipss += lpips_fn(rendering, gt).item()
        if save_images:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    if name == 'train':
        average_gt_color_rgb /= len(views)
        print(f'{name} Average GT color: {average_gt_color_rgb}')

    print(f'{name} SSIM: {ssims / len(views)}')
    print(f'{name} PSNR: {psnrs / len(views)}')
    print(f'{name} LPIPS: {lpipss / len(views)}')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "results.json"), 'w') as fp:
        json.dump({"SSIM": ssims / len(views), "PSNR": psnrs / len(views), "LPIPS": lpipss / len(views)}, fp, indent=True)

    return average_gt_color_rgb

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_all : bool, extra_opts=None):
    with torch.no_grad():
        load_ply = None if extra_opts.load_ply == 'origin' else extra_opts.load_ply
        gaussians = GaussianModel(dataset.sh_degree, extra_opts.sparse_view_num)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, extra_opts=extra_opts, load_ply=load_ply)

        bg_color = [1, 1, 1]#[1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        bg_train = None

        if not skip_train:
            bg_train = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, not_generate_video=extra_opts.not_generate_video, save_images=not extra_opts.not_saveimages, viewR=scene.getRenderCameras()[0])
        # bg_train = background
        test_bg = bg_train if bg_train is not None else background
        if not skip_test and len(scene.getTestCameras()) > 0:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, test_bg, not_generate_video=extra_opts.not_generate_video, save_images=not extra_opts.not_saveimages, viewR=scene.getRenderCameras()[0])
    
    return bg_train
        # if not skip_all:
        #     render_set(dataset.model_path, "all", scene.loaded_iter, scene.getAllCameras(), gaussians, pipeline, background, not_generate_video=extra_opts.not_generate_video, save_images=not extra_opts.not_saveimages)

@torch.no_grad()
def render_path(dataset : ModelParams, iteration : int, pipeline : PipelineParams, extra_opts=None, bg=None):
    load_ply = None if extra_opts.load_ply == 'origin' else extra_opts.load_ply
    print("load_ply", load_ply)
    gaussians = GaussianModel(dataset.sh_degree, extra_opts.sparse_view_num)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, extra_opts=extra_opts, load_ply=load_ply)

    iteration = scene.loaded_iter

    if bg is None:
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    else:
        background = bg

    model_path = dataset.model_path + postfix
    name = "render"

    views = scene.getRenderCameras()

    render_dir = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    makedirs(render_dir, exist_ok=True)

    renders_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    writer = imageio.get_writer(f"{renders_path}/video.mp4", fps=30)
    writerD = imageio.get_writer(f"{renders_path}/videoD.mp4", fps=30)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        clean_indices = gaussians.get_clean_indices(view, scale=0.5, render=True)
        render_pkg = render(view, gaussians, pipeline, background, clean_indices=clean_indices, test=True)
        rendering = torch.clamp(render_pkg["render"], 0, 1)

        rendering_np = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        writer.append_data(rendering_np)

        dep = render_pkg["rendered_depth"].cpu().numpy()[0]
        dep = ((dep - dep.min()) / (dep.max() - dep.min()))
        dep = cm.get_cmap('turbo')(dep)[..., :3]
        writerD.append_data((dep * 255).astype('uint8'))

    print(renders_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_all", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--not_saveimages", action="store_true")
    parser.add_argument("--not_generate_video", "-ng", action="store_true")
    parser.add_argument("--is_eval", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--render_resize_method", default="crop", type=str)
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                        help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                        else use dense view. In sparse setting, sparse views will be used as training data, \
                        others will be used as testing data.")
    parser.add_argument("--init_pcd_name", default='origin', type=str, 
                        help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")
    parser.add_argument("--transform_the_world", action="store_true", help="Transform the world to the origin")
    parser.add_argument("--load_ply", default="origin", type=str, help="Load other ply as init")
    parser.add_argument("--postfix", default="_stage1", type=str, help="Postfix for output directory (e.g. _stage1 for repair, _stage2 for inpainting)")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.is_renderrr = True

    postfix = args.postfix if hasattr(args, 'postfix') else '_den'

    # sometimes we only want to render the images, and do not want to evaluate the metrics
    if args.is_eval:
        with torch.no_grad():
            evaluate([args.model_path])
        exit()

    bg = None

    if not args.render_path:
        bg = render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_all, extra_opts=args)

    if args.render_path:
        render_path(model.extract(args), args.iteration, pipeline.extract(args), extra_opts = args, bg=bg)
