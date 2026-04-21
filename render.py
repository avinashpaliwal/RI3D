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
import subprocess
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm
import imageio
from matplotlib import cm

from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel, render
import lpips
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix


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


from depth_layering import get_depth_bins

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

    # viewR = scene.getRenderCameras()[0]
    

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
            # print(gt.dtype, gt.shape, gt.min(), gt.max(), bgmask.shape, (bgmask * gt).mean(dim=(1, 2)).shape, (bgmask * gt).shape, torch.nonzero(bgmask * gt).shape, torch.nonzero(bgmask * gt).dtype)#.mean(dim=(0)).shape, torch.nonzero(bgmask * gt).mean(dim=(0)).dtype)
            average_gt_color_rgb += sum_values/num_values  #torch.nonzero(bgmask * gt).mean(dim=(0))
            # average_gt_color_rgb += gt.mean(dim=(1, 2))
            # print(gt.mean(dim=(1, 2)).shape)
        # cv2.imwrite('bgmask7.png', bgmask * 255)
        # exit()
        view.distance = viewR.distance
        clean_indices = gaussians.get_clean_indices(view, scale=0.5, render=True)
        render_pkg = render(view, gaussians, pipeline, background, clean_indices=clean_indices)
        rendering = render_pkg["render"]
        depths.append(render_pkg["rendered_depth"].cpu().numpy()[0])
        ssims += ssim(rendering, gt).mean().item()
        psnrs += psnr(rendering, gt).mean().item()
        lpipss += lpips_fn(rendering, gt).item() # NCHW
        if save_images:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    if name == 'train':
        average_gt_color_rgb /= len(views)
        print(f'{name} Average GT color: {average_gt_color_rgb}')
    # exit()
    print(f'{name} SSIM: {ssims / len(views)}')
    print(f'{name} PSNR: {psnrs / len(views)}')
    print(f'{name} LPIPS: {lpipss / len(views)}')

    # since the eval is done in the render function, just dump the results to json
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "results.json"), 'w') as fp:
        json.dump({"SSIM": ssims / len(views), "PSNR": psnrs / len(views), "LPIPS": lpipss / len(views)}, fp, indent=True)

    return average_gt_color_rgb
    # # Use ffmpeg to output video
    # if not not_generate_video:
    #     renders_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4")
    #     gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt.mp4")
    #     combined_path = os.path.join(model_path, name, "ours_{}".format(iteration), "combined.mp4")
    #     # Use ffmpeg to output video
    #     subprocess.run(["ffmpeg", "-y", 
    #                 "-framerate", "24",
    #                 "-i", os.path.join(render_path, "%05d.png"), 
    #                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    #                 "-c:v", "libx264", 
    #                 "-crf", "23", 
    #                 # "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
    #                 renders_path], 
    #                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    #                 )
    #     subprocess.run(["ffmpeg", "-y", 
    #                 "-framerate", "24",
    #                 "-i", os.path.join(gts_path, "%05d.png"), 
    #                 "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    #                 "-c:v", "libx264", 
    #                 "-crf", "23", 
    #                 # "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
    #                 gt_path], 
    #                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    #                 )
    #     # Concatenate the videos vertically using the `concat` filter
    #     command = [
    #         "ffmpeg","-y",
    #         "-i",renders_path,
    #         "-i",gt_path,
    #         "-filter_complex","[0:v][1:v]hstack=inputs=2[v]",
    #         "-map","[v]",
    #         "-c:v","libx264",
    #         "-crf","23",
    #         "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
    #         combined_path
    #     ]
    #     # Run the command
    #     subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL

    #     # Remove the original videos
    #     if os.path.exists(renders_path):
    #         os.remove(renders_path)
    #     os.remove(gt_path)

    #     # use opencv generate depth video
    #     # import pdb
    #     # pdb.set_trace()
    #     depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth.mp4")
    #     depth_video = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'mp4v'), 24, (depths[0].shape[1], depths[0].shape[0]), False)
    #     for depth in depths:
    #         # opencv need to convert to uint8
    #         if depth.max() > 0:
    #             depth[depth <= 0] = depth[depth>0].min()
    #             depth_normalized = cv2.normalize(depth, depth, 0.0, 1.0, cv2.NORM_MINMAX)
    #         else:
    #             depth_normalized = np.zeros_like(depth)
    #         depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    #         depth_uint8 = np.uint8(depth_normalized)
    #         depth_video.write(depth_uint8)
    #     depth_video.release()
    #     # due to some bug, we need to use ffmpeg to convert the depth video to mp4
    #     subprocess.run(["ffmpeg", "-y", "-i", depth_path, "-c:v", "libx264", "-crf", "23", depth_path.replace(".mp4", "_compressed.mp4")], 
    #                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #     os.remove(depth_path)

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
        if not skip_test and len(scene.getTestCameras()) > 0:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, not_generate_video=extra_opts.not_generate_video, save_images=not extra_opts.not_saveimages, viewR=scene.getRenderCameras()[0])
    
    return bg_train
        # if not skip_all:
        #     render_set(dataset.model_path, "all", scene.loaded_iter, scene.getAllCameras(), gaussians, pipeline, background, not_generate_video=extra_opts.not_generate_video, save_images=not extra_opts.not_saveimages)

from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2_tensor, fov2focal, focal2fov
@torch.no_grad()
def render_path(dataset : ModelParams, iteration : int, pipeline : PipelineParams, extra_opts=None, bg=None):
    load_ply = None if extra_opts.load_ply == 'origin' else extra_opts.load_ply
    print("load_ply", load_ply)
    gaussians = GaussianModel(dataset.sh_degree, extra_opts.sparse_view_num)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, extra_opts=extra_opts, load_ply=load_ply)

    # gaussians_den = GaussianModel(dataset.sh_degree, extra_opts.sparse_view_num)
    # gaussians_den.load_ply(f'output_den{extra_opts.sparse_view_num}/gaussian_object/{dataset.source_path.split("/")[-1]}_{extra_opts.sparse_view_num}/save/last.ply')

    # gaussians.training_setup(extra_opts)
    # arr = np.load('arr.npy', allow_pickle=True).item()
    # blended_depth, inpainted_image, inpaint_mask, camera = arr['blended_depth'], arr['best_controlnet_out'], arr['mask'], arr['viewpoint_cam']
    # gaussians.depth_densify(blended_depth, inpainted_image, inpaint_mask, camera)

    iteration = scene.loaded_iter

    if bg is None:
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    else:
        background = bg

    model_path = dataset.model_path + postfix
    name = "render"

    views = scene.getRenderCameras()
    viewT = scene.getTrainCameras()[0]
    # print(views[0].distance)
    # exit()
    
    # min_dist = 100000000
    # min_index = -1
    # for idx, view in enumerate(views):
    #     diff = (view.camera_center - viewT.camera_center)
    #     # print(diff, diff.shape)
    #     dist = torch.pow(torch.pow(diff, 2).sum(dim=0), 0.5)
    #     if dist < min_dist:
    #         min_dist = dist
    #         min_index = idx
        # print(dist, min_dist, min_index)

    # print(len(views))
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    # Use ffmpeg to output video
    renders_path = os.path.join(model_path, name, "ours_{}".format(iteration))#, "renders.mp4")
    writer = imageio.get_writer(f"{renders_path}/video.mp4", fps=30)
    # writer = imageio.get_writer(renders_path, fps=30)
    writerD = imageio.get_writer(f"{renders_path}/videoD.mp4", fps=30)

    depths = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if args.render_resize_method == 'crop':
        #     image_size = 512
        # elif args.render_resize_method == 'pad':
        #     image_size = max(view.image_width, view.image_height)
        # else:
        #     raise NotImplementedError
        # view.original_image = torch.zeros((3, image_size, image_size), device=view.original_image.device)
        # focal_length_x = fov2focal(view.FoVx, view.image_width)
        # focal_length_y = fov2focal(view.FoVy, view.image_height)
        # view.image_width = image_size
        # view.image_height = image_size
        # view.FoVx = focal2fov(focal_length_x, image_size)
        # view.FoVy = focal2fov(focal_length_y, image_size)
        # view.projection_matrix = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda().float()
        # view.full_proj_transform = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        # M = np.array([[ 1,  1, 0 ],
        #               [ 1, 0, 0 ],
        #               [ 0,  0, 1 ]])
        
        # # print(view.R)
        # view.R = torch.tensor(view.R @ M)
        # view.world_view_transform = getWorld2View2_tensor(view.R, torch.tensor(view.T)).transpose(0, 1).cuda().float()
        # view.projection_matrix    = getProjectionMatrix(znear=view.znear, zfar=view.zfar, fovX=view.FoVx, fovY=view.FoVy).transpose(0,1).cuda().float()
        # view.full_proj_transform  = (view.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))).squeeze(0)
        # # print(view.R)

        # view.refactor(0.5, 0.75)
        # view.refactor(1, 0.5)
        # clean_indices = None
        clean_indices = gaussians.get_clean_indices(view, scale=0.5, render=True)
        # render_pkg = render(view, gaussians, pipeline, background, fixed_pc=gaussians_den, clean_indices=clean_indices)
        render_pkg = render(view, gaussians, pipeline, background, clean_indices=clean_indices, test=True)
        rendering = torch.clamp(render_pkg["render"], 0, 1)
        depths.append(render_pkg["rendered_depth"].cpu().numpy()[0])
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(render_pkg["rendered_alpha"], os.path.join(render_path, '{0:05d}'.format(idx) + "_alpha.png"))
        rendering = (rendering.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        # cv2.imwrite(f'ren_images/{idx}.png', rendering[..., ::-1])
        # print(rendering.shape, rendering.max(), rendering.min())
        writer.append_data(rendering)
        dep = render_pkg["rendered_depth"].cpu().numpy()[0]
        dep = ((dep - dep.min()) / (dep.max() - dep.min()))
        dep = cm.get_cmap('turbo')(dep)[..., :3]
        writerD.append_data((dep * 255).astype('uint8'))

    print(renders_path)
    # # Use ffmpeg to output video
    # subprocess.run(["ffmpeg", "-y", 
    #             "-framerate", "24",
    #             "-i", os.path.join(render_path, "%05d.png"), 
    #             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    #             "-c:v", "libx264", 
    #             "-crf", "23", 
    #             # "-pix_fmt", "yuv420p",  # Set pixel format for compatibility
    #             renders_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    #             )

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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    lpips_fn = lpips.LPIPS(net='vgg').cuda()
    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.is_renderrr = True

    postfix = '_den'

    # sometimes we only want to render the images, and do not want to evaluate the metrics
    if args.is_eval:
        with torch.no_grad():
            evaluate([args.model_path])
        exit()

        # exit()
    bg=None
    # bg = render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_all, extra_opts = args)

    if args.render_path:
        render_path(model.extract(args), args.iteration, pipeline.extract(args), extra_opts = args, bg=bg)
