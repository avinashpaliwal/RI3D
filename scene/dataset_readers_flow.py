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

import sys
import os
import os.path as osp
from typing import NamedTuple, Optional
import json

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import yaml
from kornia.geometry.depth import depth_to_3d

from scene.colmap_loader import (qvec2rotmat, read_extrinsics_binary,
                                 read_extrinsics_text, read_intrinsics_binary,
                                 read_intrinsics_text, read_points3D_binary,
                                 read_points3D_text)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2, transform_pcd
from utils.image_utils import load_meshlab_file
from utils.camera_utils import transform_cams, CameraInfo, generate_ellipse_path_from_camera_infos

from utils.bilateral_filtering import sparse_bilateral_filtering

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    render_cameras: Optional[list[CameraInfo]] = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}



def readMipTransforms(path, resolution=4):
    cam_infos = []

    # # direct load resized images, not the original ones
    # if extra_opts is not None and extra_opts.resolution in [1, 2, 4, 8]:
    #     tmp_images_folder = images_folder + f'_{str(extra_opts.resolution)}' if extra_opts.resolution != 1 else images_folder
    #     if not osp.exists(tmp_images_folder):
    #         print(f"The {tmp_images_folder} is not found, use original resolution images")
    #     else:
    #         print(f"Using resized images in {tmp_images_folder}...")
    #         images_folder = tmp_images_folder
    # else:
    #     print("use original resolution images")

    with open(f'{path}/transforms.json') as f:
        camera_dict = json.load(f)

    height, width = camera_dict['h'], camera_dict['w']
    focal_length_x, focal_length_y = camera_dict['fl_x'], camera_dict['fl_y']

    cam_extrinsics = camera_dict['frames']

    _coord_trans = np.diag([1, -1, -1, 1])

    for idx, cam_extr in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extr)))
        sys.stdout.flush()


        impath = cam_extr['file_path']
        extr   = np.array(cam_extr['transform_matrix'])
        extr   = extr @ _coord_trans
        extr   = np.linalg.inv(extr)
        # extr   = np.array(cam_extr['transform_matrix']) @ _coord_trans

        uid = idx
        R = np.transpose(extr[:3, :3])
        T = extr[:3, 3]


        # fl_x is the horizontal focal length and fl_y the vertical one; every
        # consumer pairs FovX with width and FovY with height (see the K build
        # below, GaussianModel.depth_densify, and Camera.__init__). Identical to
        # the PINHOLE branch of readColmapCameras.
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        image_path = osp.join(path, impath)
        image_name = osp.basename(image_path).split(".")[0]
        image = None#Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              width=resolution*width, height=resolution*height, mask=None, mono_depth=None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos



def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, extra_opts=None):
    cam_infos = []

    # direct load resized images, not the original ones
    if extra_opts is not None and extra_opts.resolution in [1, 2, 4, 8]:
        tmp_images_folder = images_folder + f'_{str(extra_opts.resolution)}' if extra_opts.resolution != 1 else images_folder
        if not osp.exists(tmp_images_folder):
            print(f"The {tmp_images_folder} is not found, use original resolution images")
        else:
            print(f"Using resized images in {tmp_images_folder}...")
            images_folder = tmp_images_folder
    else:
        print("use original resolution images")

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE": 
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = osp.join(images_folder, osp.basename(extr.name))
        # print(images_folder, extr.name,  osp.basename(extr.name), image_path)
        # exit()
        image_name = osp.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        ### load masks
        mask_path_png = osp.join(osp.dirname(images_folder), "masks", osp.basename(
            image_path).replace(osp.splitext(osp.basename(image_path))[-1], '.png'))

        if osp.exists(mask_path_png) and hasattr(extra_opts, "use_mask") and extra_opts.use_mask:
            mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = None
        
        mask = np.ones_like(mask)

        mono_depth = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, mask=mask, mono_depth=mono_depth)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)



def _check_metric_depth(depth, source):
    """Guard against disparity being loaded where metric depth is expected.

    `depth_rel/*.npy` must hold metric z-depth in SfM world units. Feeding it
    raw Depth-Anything disparity instead inverts near/far and is typically an
    order of magnitude off -- and because everything downstream still runs, the
    mistake is invisible until reconstruction quality silently collapses.
    Heuristic: Depth-Anything emits unnormalised disparity whose median lands
    in the tens-to-hundreds, while SfM world units for these scenes put median
    depth in the single digits. Measured on sceneA the two populations are
    medians 37-76 (disparity) against 2.5-3.9 (metric), so 20 separates them
    with margin on both sides. A genuinely large metric scene could trip this;
    it only warns.
    """
    finite = depth[np.isfinite(depth) & (depth > 0)]
    if finite.size and np.median(finite) > 20.0:
        print(
            f"[WARNING] {source}: median depth {np.median(finite):.1f} looks like "
            "disparity, not metric depth. Regenerate it with the aligned depth stage "
            "(utils/depth_align.py); training on this will invert the geometry."
        )
    return depth


def readColmapSceneInfo(path, images, eval, llffhold=8, extra_opts=None, ply_init=None):
 
    cam_infos_unsorted = readMipTransforms(path=path, resolution=extra_opts.resolution)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)


    with open(f'{path}/train_test_split_{extra_opts.sparse_view_num}.json') as json_data:
        data = json.load(json_data)
        train_ids = data['train_ids']
        test_ids  = data['test_ids']

    for train_id in train_ids:

        cam_infos[train_id] = cam_infos[train_id]._replace(image=Image.open(cam_infos[train_id].image_path))


    for test_id in test_ids:

        cam_infos[test_id] = cam_infos[test_id]._replace(image=Image.open(cam_infos[test_id].image_path))
    
    cam_infos[0] = cam_infos[0]._replace(image=Image.open(cam_infos[train_ids[0]].image_path))
    
    train_cam_infos = [cam_infos[i] for i in train_ids]
    test_cam_infos = [cam_infos[i] for i in test_ids]
    # print(train_cam_infos)

    render_cam_infos = generate_ellipse_path_from_camera_infos(cam_infos)

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if hasattr(extra_opts, 'sparse_view_num') and extra_opts.sparse_view_num > 0: # means sparse setting
        assert eval == False
        # assert osp.exists(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt")), "sparse_id.txt not found!"
        # ids = np.loadtxt(osp.join(path, f"sparse_{str(extra_opts.sparse_view_num)}.txt"), dtype=np.int32)
        # ids_test = np.loadtxt(osp.join(path, f"sparse_test.txt"), dtype=np.int32)
        # test_cam_infos = [train_cam_infos[i] for i in ids_test]
        # train_cam_infos = [train_cam_infos[i] for i in ids]

        idx_sub = [round(i) for i in np.linspace(0, len(train_cam_infos)-1, extra_opts.sparse_view_num)]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
        assert len(train_cam_infos) == extra_opts.sparse_view_num

        print("Sparse view, only {} images are used for training, others are used for eval.".format(len(idx_sub)))

    flows = np.load(f'{path}/depths{extra_opts.sparse_view_num}.npy')
    masks = np.load(f'{path}/confs{extra_opts.sparse_view_num}.npy')
    # print(train_cam_infos)
    # exit()
    xyz_arr, rgb_arr, radii2_arr, sparse_dep_arr = [], [], [], []


    def _find_and_load_depth(cam_info, prefix):
        img_dir = os.path.dirname(cam_info.image_path)
        base_name = os.path.splitext(cam_info.image_name)[0]
        num_v = extra_opts.sparse_view_num
        candidates = [
            os.path.join(img_dir, "depth_rel", f"{prefix}{base_name}_{num_v}.npy"),
            os.path.join(os.path.dirname(img_dir), "depth_rel", f"{prefix}{base_name}_{num_v}.npy"),
            os.path.join(path, "depth_rel", f"{prefix}{base_name}_{num_v}.npy"),
            os.path.join(img_dir, f"{prefix}{base_name}_{num_v}.npy"),
            os.path.join(img_dir, "depth_rel", f"inp_dust3r{base_name}_{num_v}.npy"),
            os.path.join(path, "depth_rel", f"inp_dust3r{base_name}_{num_v}.npy"),
            os.path.join(img_dir, "depth_rel", f"inpv2{base_name}_{num_v}.npy"),
            os.path.join(path, "depth_rel", f"inpv2{base_name}_{num_v}.npy"),
        ]
        for c in candidates:
            if os.path.exists(c):
                d = np.load(c)
                if d.ndim == 3 and d.shape[-1] == 3:
                    d = d[..., 0]
                return _check_metric_depth(d, c)
        raise FileNotFoundError(
            f"No depth prior for {base_name} (prefix '{prefix}', {num_v} views). Looked in:\n  "
            + "\n  ".join(candidates)
            + "\nRun the SfM/depth stage so depth aligned to the MASt3R pointmaps is written."
        )

    def _find_watermark_mask(cam_info):
        """Photometric-loss mask, if the SfM stage wrote one.

        One mask per scene, at image resolution -- watermarks are composited at
        fixed pixel coordinates, which is the assumption tools/detect_watermark.py
        relies on. Returns a float array, 1 = supervise, 0 = ignore, or None.

        Without this, a burned-in watermark is supervised as ground truth by L1 and
        SSIM in every view, and since it sits at the same image coordinates in all
        of them the Gaussians reproduce it as a consistent floating object.
        """
        img_dir = os.path.dirname(cam_info.image_path)
        sfm_dir = os.path.dirname(img_dir)
        candidates = [
            os.path.join(sfm_dir, "masks", "watermark_mask.png"),
            os.path.join(path, "masks", "watermark_mask.png"),
            os.path.join(img_dir, "masks", "watermark_mask.png"),
            os.path.join(path, "watermark_mask.png"),
            # inference.py's `wm` stage writes here, one level above <backend>_sfm/
            os.path.join(os.path.dirname(sfm_dir), "watermark_mask.png"),
            os.path.join(os.path.dirname(path), "watermark_mask.png"),
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
            # Compare against the *image on disk*, not cam_info.width/height --
            # readMipTransforms stores those pre-divided by `resolution` (4096 for a
            # 1024-wide photo) and the K build divides them back out. loadCam then
            # resizes this mask alongside the image, so full image resolution is the
            # right grid to hand it here.
            img_w, img_h = cam_info.image.size
            if (m.shape[0], m.shape[1]) != (img_h, img_w):
                # Skip, do not abort: a stale pointmap-resolution mask sitting in
                # masks/ must not shadow a correct full-resolution one later in the
                # chain. Resizing it here is not an option either -- that would
                # silently slide the mask off the watermark.
                print(f"[!] {c} is {m.shape[1]}x{m.shape[0]} but images are "
                      f"{img_w}x{img_h}; skipping it.")
                continue
            print(f"[i] Watermark loss mask: {c}")
            return (m <= 127).astype(np.float32)
        return None

    watermark_mask = _find_watermark_mask(train_cam_infos[0]) if train_cam_infos else None
    if watermark_mask is not None:
        print(f"[i] Watermark loss mask: ignoring {(1 - watermark_mask).mean() * 100:.2f}% "
              "of every training pixel (photometric losses only).")

    for idx, cam_info in enumerate(train_cam_infos):
        # depth_rel holds metric z-depth in SfM world units (see utils/depth_align.py).
        depth = torch.Tensor(_find_and_load_depth(cam_info, "inpv2"))[None, None]
        train_cam_infos[idx] = cam_info._replace(mono_depth=depth[0], loss_mask=watermark_mask)

    if not extra_opts.is_renderrr:
        config = yaml.safe_load(open('configs/argument.yaml', 'r'))

        for idx, cam_info in enumerate(train_cam_infos):
            im_data = np.array(cam_info.image.convert('RGB'), dtype=np.float32)
            depth_rel = _find_and_load_depth(cam_info, "inp_dust3r")
            if depth_rel.shape[-1] == 3:
                depth_rel = depth_rel[..., 0]

            # Metric depth in SfM world units -- back-projects straight into the
            # frame the MASt3R poses live in, no rescaling needed.
            depth = torch.Tensor(depth_rel)[None, None]
            print(depth.max(), depth.min(), cam_info.image_path)

            # Init radius equal to shorter length of the rectangle. Default: Height
            # Radii per frame
            radii = np.tan(0.5 * float(cam_info.FovY))  * depth / (cam_info.height / extra_opts.resolution)
            radii2 = radii**2

            K = torch.eye(3)[None]
            K[:, 0, 0] = fov2focal(cam_info.FovX, cam_info.width)
            K[:, 0, 2] = cam_info.width / 2.0
            K[:, 1, 1] = fov2focal(cam_info.FovY, cam_info.height)
            K[:, 1, 2] = cam_info.height / 2.0
            K[:, :2]   = K[:, :2] / extra_opts.resolution

            # print(K, cam_info.width, cam_info.height, im_data.shape)
            # exit()
            height, width, _ = im_data.shape
            # print(depth.max(), depth.min(), K)
            camera3d = depth_to_3d(depth, K)
            
            xyz_cam = camera3d[0].permute(1, 2, 0).reshape(-1, 3).numpy()
            rgb = torch.Tensor(im_data).reshape(-1, 3).numpy()
            radii2 = radii2[0].permute(1, 2, 0).reshape(-1).numpy()
            # print(height, width, depth.shape, xyz_cam.shape, radii2.shape, K, extra_opts.resolution, cam_info.width, cam_info.height)
            # exit()

            # Optional MASt3R-confidence gate on the initial points. Off by
            # default: it makes the per-view point counts unequal, which breaks
            # the `fused_point_cloud.shape[0] // num_cameras` reshape that
            # GaussianModel.create_from_pcd performs when mono_d_so_enable=True
            # (scripts/train_gs_init.py). Safe for train_gs.py / leave_one_out_*.
            conf_thr = getattr(extra_opts, "depth_conf_thr", 0.0)
            if conf_thr > 0:
                conf = masks[idx].reshape(-1)
                if conf.shape[0] != xyz_cam.shape[0]:
                    raise ValueError(
                        f"confs{extra_opts.sparse_view_num}.npy view {idx} has {conf.shape[0]} "
                        f"entries but the depth map back-projects to {xyz_cam.shape[0]} points; "
                        "both must be written at the transforms.json resolution."
                    )
                keep = conf > conf_thr
                print(f"  conf > {conf_thr}: keeping {keep.sum()}/{keep.size} points for {cam_info.image_name}")
                xyz_cam, rgb, radii2 = xyz_cam[keep], rgb[keep], radii2[keep]


            # w2c = np.zeros((4, 4))
            # w2c[:3, :3] = cam_info.R.transpose()
            # w2c[:3, 3] = cam_info.T
            # w2c[3, 3] = 1.0
            # print(w2c.inverse())
            # sparse_dep = np.matmul(K, np.matmul(w2c, sparse_positions)[:3]).T # N, (x, y, z)
            # sparse_dep_hom = (sparse_dep / sparse_dep[:, 2:]).round().int()[:, :2, 0][:, [1, 0 ]]
            # # print(sparse_positions[:5], sparse_dep[:5]/sparse_dep[:5, 2:], sparse_dep[:5, 2:])
            # masked_sparse_dep = np.zeros_like(depth[0, 0]) # H, W
            # u = sparse_dep_hom[:, 0]
            # v = sparse_dep_hom[:, 1]
            # # if n_input_views in [2, 4]:
            # u_filt = np.where(u >= height)[0].tolist() + np.where(u <= 0)[0].tolist()
            # v_filt = np.where(v >= width)[0].tolist() + np.where(v <= 0)[0].tolist()
            # # print(u.max(), v.max(), u.min(), v.min(), depth.shape, u.shape, v.shape, u_filt, v_filt)
            # u = np.delete(u, u_filt + v_filt, 0)
            # v = np.delete(v, u_filt + v_filt, 0)
            # sparse_dep = np.delete(sparse_dep, u_filt + v_filt, 0)
            # #     print(u.max(), v.max(), u.min(), v.min(), depth.shape, u.shape, v.shape)
            # # exit()
            # # print(masked_sparse_dep[(u, v)].shape, sparse_dep[:, 2].shape, sparse_dep.shape)
            # masked_sparse_dep[(u, v)] = sparse_dep[:, 2, 0]
            # masked_sparse_dep = torch.Tensor(masked_sparse_dep).cuda()



            xyz_arr.append(xyz_cam)
            rgb_arr.append(rgb)
            radii2_arr.append(radii2)
            # sparse_dep_arr.append(masked_sparse_dep)

        if ply_init is None:
            xyz = np.concatenate(xyz_arr, axis=0)
            rgb = np.concatenate(rgb_arr, axis=0)
        else:
            # print(np.concatenate(xyz_arr, axis=0).shape, np.concatenate(rgb_arr, axis=0).shape)
            xyz = ply_init[0]
            rgb = ply_init[1][..., 0] * 255
            # print(xyz.shape, rgb.shape)
            # exit()
        radii2 = np.concatenate(radii2_arr, axis=0)
        
        ply_path = os.path.join(path, "point_cloud.ply")

        if os.path.exists(ply_path):
            os.remove(ply_path)

        # storePly(ply_path, xyz, rgb, radii2)
        storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
    
    else:
        pcd = None
        ply_path = None
        radii2 = None

    print(f"PCD {pcd == None} {path}")
    


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           render_cameras=render_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info, [flows, masks, radii2]

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
}
