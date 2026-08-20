#!/usr/bin/env python3
"""
RI3D Rerun Visualization Tool (viz.py)
======================================
Interactive 3D/2D visualizer for the RI3D 3D Gaussian Splatting pipeline
and MASt3R-SfM intermediate and final estimation artifacts using Rerun.

Usage examples:
    # Visualize an output directory with Rerun viewer
    python viz.py -i output/sceneA

    # Visualize a specific subfolder or PLY
    python viz.py -i output/sceneA/mast3r_sfm
    python viz.py -i output/gs_init/sceneA_3

    # Save to a .rrd recording file
    python viz.py -i output/sceneA --save sceneA_viz.rrd

    # Host a web viewer server for remote browser viewing
    python viz.py -i output/sceneA --serve --port 9876

    # Scrub through reconstruction progression along pipeline timeline
    python viz.py -i output/sceneA --timeline
"""

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import rerun as rr
except ImportError:
    print("[!] Error: 'rerun-sdk' is not installed in the active environment.")
    print("    Install with: pip install rerun-sdk")
    sys.exit(1)

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None

# Spherical Harmonics constant for DC term
SH_C0 = 0.28209479177387814
VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG', '.bmp', '.BMP', '.webp', '.WEBP')


# =============================================================================
# Helper Utilities
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def safe_load_torch(path: str):
    """Load torch or pickle files with PyTorch 2.6+ compatibility."""
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_ply_points_and_colors(
    ply_path: str, max_points: int = 500000, min_opacity: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads 3D points and RGB colors from a standard or 3D Gaussian Splatting PLY file.
    Returns: (positions [N, 3], colors [N, 3] uint8, opacities [N] or None, scales [N, 3] or None)
    """
    if not os.path.isfile(ply_path):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), None, None

    if PlyData is None:
        # Fallback simple reader if plyfile is missing
        print(f"[!] plyfile not installed, skipping detailed PLY parse for {ply_path}")
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None, None

    try:
        plydata = PlyData.read(ply_path)
        v = plydata['vertex']
        
        # Position
        xyz = np.stack([v['x'], v['y'], v['z']], axis=-1).astype(np.float32)
        n_pts = len(xyz)

        # Check for 3D Gaussian Splatting attributes
        names = v.data.dtype.names
        is_3dgs = 'f_dc_0' in names and 'opacity' in names

        opacities = None
        scales = None

        if is_3dgs:
            # Color from SH DC component
            f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1).astype(np.float32)
            rgb = np.clip((0.5 + SH_C0 * f_dc), 0.0, 1.0)
            colors = (rgb * 255.0).astype(np.uint8)

            # Opacity
            raw_opacity = np.asarray(v['opacity'], dtype=np.float32)
            opacities = sigmoid(raw_opacity)

            # Scales
            if 'scale_0' in names:
                scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1).astype(np.float32))

            # Filter low opacity if threshold is specified
            if min_opacity > 0.0:
                mask = opacities >= min_opacity
                xyz = xyz[mask]
                colors = colors[mask]
                opacities = opacities[mask]
                if scales is not None:
                    scales = scales[mask]
        elif 'red' in names and 'green' in names and 'blue' in names:
            colors = np.stack([v['red'], v['green'], v['blue']], axis=-1).astype(np.uint8)
        elif 'r' in names and 'g' in names and 'b' in names:
            colors = np.stack([v['r'], v['g'], v['b']], axis=-1).astype(np.uint8)
        else:
            # Default uniform light blue / grey
            colors = np.full((n_pts, 3), 180, dtype=np.uint8)

        # Subsample if exceeding max_points
        if 0 < max_points < len(xyz):
            sub_idx = np.random.choice(len(xyz), size=max_points, replace=False)
            xyz = xyz[sub_idx]
            colors = colors[sub_idx]
            if opacities is not None:
                opacities = opacities[sub_idx]
            if scales is not None:
                scales = scales[sub_idx]

        return xyz, colors, opacities, scales
    except Exception as e:
        print(f"[!] Error reading PLY '{ply_path}': {e}")
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8), None, None


# =============================================================================
# Camera & Pose Parsing
# =============================================================================

def parse_transforms_json(transforms_path: str) -> List[Dict]:
    """Parses transforms.json format camera parameters."""
    with open(transforms_path, 'r') as f:
        data = json.load(f)

    frames = []
    w = data.get('w', 512)
    h = data.get('h', 512)
    fl_x = data.get('fl_x', data.get('camera_angle_x', None))
    fl_y = data.get('fl_y', fl_x)

    if fl_x is not None and isinstance(fl_x, float) and fl_x < 10.0:
        # Camera angle x in radians -> focal length in pixels
        fl_x = 0.5 * w / math.tan(0.5 * fl_x)
        fl_y = fl_x

    cx = data.get('cx', w / 2.0)
    cy = data.get('cy', h / 2.0)

    for idx, frame in enumerate(data.get('frames', [])):
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        # transforms.json stores c2w in the OpenGL convention (+y up, -z
        # forward), matching what scene/dataset_readers_flow.py undoes on load.
        # rr.Pinhole defaults to RDF (OpenCV: +y down, +z forward), so without
        # this flip every camera faces 180 degrees away from the scene and any
        # depth logged under it unprojects behind the pointmap.
        c2w = c2w @ np.diag([1, -1, -1, 1]).astype(np.float32)
        frame_fl_x = frame.get('fl_x', fl_x)
        frame_fl_y = frame.get('fl_y', fl_y)
        frame_w = frame.get('w', w)
        frame_h = frame.get('h', h)
        frame_cx = frame.get('cx', cx)
        frame_cy = frame.get('cy', cy)

        img_path = frame.get('file_path', f"frame_{idx}")
        frames.append({
            'name': Path(img_path).stem,
            'file_path': img_path,
            'c2w': c2w,
            'intrinsics': {
                'width': int(frame_w),
                'height': int(frame_h),
                'fl_x': float(frame_fl_x),
                'fl_y': float(frame_fl_y),
                'cx': float(frame_cx),
                'cy': float(frame_cy),
            }
        })
    return frames


def parse_cameras_json(cameras_json_path: str) -> List[Dict]:
    """Parses cameras.json exported by 3D Gaussian Splatting training."""
    with open(cameras_json_path, 'r') as f:
        data = json.load(f)

    frames = []
    for item in data:
        img_name = item.get('img_name', f"cam_{item.get('id', 0)}")
        width = item.get('width', 512)
        height = item.get('height', 512)
        fx = item.get('fx', 500.0)
        fy = item.get('fy', fx)

        # Rotation and Position
        R = np.array(item.get('rotation', np.eye(3)), dtype=np.float32)
        T = np.array(item.get('position', [0, 0, 0]), dtype=np.float32)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = T

        frames.append({
            'name': img_name,
            'file_path': img_name,
            'c2w': c2w,
            'intrinsics': {
                'width': int(width),
                'height': int(height),
                'fl_x': float(fx),
                'fl_y': float(fy),
                'cx': float(width / 2.0),
                'cy': float(height / 2.0),
            }
        })
    return frames


# =============================================================================
# Artifact Discovery
# =============================================================================

def discover_pipeline_artifacts(root_path: str) -> Dict[str, any]:
    """
    Scans the given folder (and its parent/children) to locate all available
    intermediate and final estimation artifacts.
    """
    root = os.path.abspath(root_path)
    artifacts = {
        'root': root,
        'sfm_dir': None,
        'sfm_plys': [],
        'pointmaps': [],
        'depth_rel': [],
        'flow_arrays': {},
        'transforms_json': None,
        'cameras_json': None,
        'images': [],
        'gs_init_ply': None,
        'gs_base_ply': None,
        'loo_dirs': [],
        'diffs_pkl': None,
        'repair_ply': None,
        'inpainting_ply': None,
        'final_ply': None,
    }

    # Search candidates across root and surrounding standard layout
    search_dirs = [root]
    if os.path.isdir(os.path.join(root, "mast3r_sfm")):
        search_dirs.append(os.path.join(root, "mast3r_sfm"))
    
    # Also check parent if root was a subfolder
    parent = os.path.dirname(root)
    if os.path.basename(parent) == "output":
        search_dirs.append(parent)

    # 1. SfM directory & PLY point clouds
    for s_dir in search_dirs:
        candidate_sfm = s_dir if "mast3r_sfm" in s_dir else os.path.join(s_dir, "mast3r_sfm")
        if os.path.isdir(candidate_sfm):
            artifacts['sfm_dir'] = candidate_sfm
            break

    # Look for point clouds in SfM.
    # Deliberately excludes "point_cloud.ply": scene/dataset_readers_flow.py
    # drops that file into the scene directory as an intermediate, and its points
    # are in each view's CAMERA frame, not world -- the reader hands camera-frame
    # points to GaussianModel, which applies the per-camera c2w itself inside
    # get_xyz (the mono_d_so_enable path). Logging it here as a world-frame cloud
    # stacks every view on top of the origin and looks badly misaligned against
    # points.ply. The world-frame init cloud is the one saved by scene.save(),
    # under <model_path>/point_cloud/iteration_*/point_cloud.ply.
    if artifacts['sfm_dir']:
        for ply_name in ["points.ply", "chart_pcd.ply"]:
            p = os.path.join(artifacts['sfm_dir'], ply_name)
            if os.path.isfile(p):
                artifacts['sfm_plys'].append(p)

        # Pointmaps
        pm_dir = os.path.join(artifacts['sfm_dir'], "pointmaps")
        if os.path.isdir(pm_dir):
            for f in sorted(os.listdir(pm_dir)):
                if f.endswith('.json'):
                    artifacts['pointmaps'].append(os.path.join(pm_dir, f))

        # Relative Depth maps
        d_rel = os.path.join(artifacts['sfm_dir'], "depth_rel")
        if os.path.isdir(d_rel):
            for f in sorted(os.listdir(d_rel)):
                if f.endswith('.npy'):
                    artifacts['depth_rel'].append(os.path.join(d_rel, f))

        # Flow arrays (confs*.npy, depths*.npy)
        for f in os.listdir(artifacts['sfm_dir']):
            if f.startswith('confs') and f.endswith('.npy'):
                artifacts['flow_arrays']['confs'] = os.path.join(artifacts['sfm_dir'], f)
            elif f.startswith('depths') and f.endswith('.npy'):
                artifacts['flow_arrays']['depths'] = os.path.join(artifacts['sfm_dir'], f)

        # Poses
        t_json = os.path.join(artifacts['sfm_dir'], "transforms.json")
        if os.path.isfile(t_json):
            artifacts['transforms_json'] = t_json

        # Images
        img_dir = os.path.join(artifacts['sfm_dir'], "images")
        if os.path.isdir(img_dir):
            for f in sorted(os.listdir(img_dir)):
                if f.endswith(VALID_IMAGE_EXTS) and not os.path.isdir(os.path.join(img_dir, f)):
                    artifacts['images'].append(os.path.join(img_dir, f))

    # 2. Gaussian Initialization (Stage 1a)
    init_candidates = [
        os.path.join(root, "debug", "gs_init"),
        "debug/gs_init",
        os.path.join(root, "debug_gs_init"),
    ]
    for d in init_candidates:
        if os.path.isdir(d):
            plys = glob.glob(f"{d}/**/point_cloud.ply", recursive=True)
            if plys:
                artifacts['gs_init_ply'] = plys[0]
                break

    # 3. Base 3DGS Training (Stage 1b)
    gs_candidates = [
        os.path.join(root, "output", "gs_init"),
        "output/gs_init",
        os.path.join(root, "gs_init"),
    ]
    for d in gs_candidates:
        if os.path.isdir(d):
            # Check for non-loo models
            for scene_sub in os.listdir(d):
                if "_loo_" not in scene_sub and os.path.isdir(os.path.join(d, scene_sub)):
                    ply_path = os.path.join(d, scene_sub, "point_cloud", "iteration_30000", "point_cloud.ply")
                    if os.path.isfile(ply_path):
                        artifacts['gs_base_ply'] = ply_path
                    cam_json = os.path.join(d, scene_sub, "cameras.json")
                    if os.path.isfile(cam_json):
                        artifacts['cameras_json'] = cam_json

    # 4. Leave-One-Out (Stage 2)
    for d in gs_candidates:
        if os.path.isdir(d):
            for scene_sub in os.listdir(d):
                if "_loo_" in scene_sub:
                    loo_root = os.path.join(d, scene_sub)
                    diffs_p = os.path.join(loo_root, "diffs.pkl")
                    if os.path.isfile(diffs_p):
                        artifacts['diffs_pkl'] = diffs_p
                    for leave_dir in sorted(os.listdir(loo_root)):
                        full_leave = os.path.join(loo_root, leave_dir)
                        if os.path.isdir(full_leave) and leave_dir.startswith("leave_"):
                            artifacts['loo_dirs'].append(full_leave)

    # 5. Stage 5a Repair & 5b Inpainting
    for den_dir in glob.glob("output_den*") + glob.glob(os.path.join(root, "output_den*")):
        last_ply = glob.glob(f"{den_dir}/**/last.ply", recursive=True)
        if last_ply:
            artifacts['repair_ply'] = last_ply[0]
            break

    for inp_dir in glob.glob("output_inp*") + glob.glob(os.path.join(root, "output_inp*")):
        last_ply = glob.glob(f"{inp_dir}/**/last.ply", recursive=True)
        if last_ply:
            artifacts['inpainting_ply'] = last_ply[0]
            break

    # 6. Final reconstructed PLYs in root
    final_plys = glob.glob(f"{root}/*_reconstructed_3dgs.ply") + glob.glob(f"{root}/*.ply")
    if final_plys:
        artifacts['final_ply'] = final_plys[0]

    return artifacts


# =============================================================================
# Rerun Logging Routines
# =============================================================================

def log_camera_and_views(artifacts: Dict, timeline: bool = False):
    """Logs cameras, images, and depth priors to Rerun."""
    # Find camera frames from transforms.json or cameras.json
    frames = []
    if artifacts['transforms_json']:
        frames = parse_transforms_json(artifacts['transforms_json'])
    elif artifacts['cameras_json']:
        frames = parse_cameras_json(artifacts['cameras_json'])

    if not frames:
        # Fallback: create placeholder cameras for raw image files
        for idx, img_p in enumerate(artifacts['images']):
            frames.append({
                'name': Path(img_p).stem,
                'file_path': img_p,
                'c2w': None,
                'intrinsics': {'width': 512, 'height': 512, 'fl_x': 500, 'fl_y': 500, 'cx': 256, 'cy': 256}
            })

    print(f"[+] Logging {len(frames)} camera view(s)...")

    # Group depth maps by base image stem
    depth_map_dict = {}
    for d_path in artifacts['depth_rel']:
        fname = Path(d_path).stem
        for f in frames:
            if f['name'] in fname:
                depth_map_dict[f['name']] = d_path
                break

    for frame in frames:
        cam_name = frame['name']
        cam_entity = f"world/cameras/{cam_name}"
        intr = frame['intrinsics']

        # 1. Pinhole intrinsics
        rr.log(
            cam_entity,
            rr.Pinhole(
                resolution=[intr['width'], intr['height']],
                focal_length=[intr['fl_x'], intr['fl_y']],
                principal_point=[intr['cx'], intr['cy']],
            )
        )

        # 2. Camera pose transform (if available)
        if frame['c2w'] is not None:
            c2w = frame['c2w']
            R = c2w[:3, :3]
            t = c2w[:3, 3]
            rr.log(
                cam_entity,
                rr.Transform3D(
                    mat3x3=R,
                    translation=t
                )
            )

        # 3. RGB Image
        if os.path.isfile(frame['file_path']):
            img = Image.open(frame['file_path']).convert("RGB")
            rr.log(f"{cam_entity}/image", rr.Image(np.array(img)))
        else:
            # Check in artifacts['images']
            for img_p in artifacts['images']:
                if Path(img_p).stem == cam_name:
                    img = Image.open(img_p).convert("RGB")
                    rr.log(f"{cam_entity}/image", rr.Image(np.array(img)))
                    break

        # 4. Relative Depth Image
        if cam_name in depth_map_dict:
            d_file = depth_map_dict[cam_name]
            try:
                depth_arr = np.load(d_file)
                if depth_arr.ndim == 2:
                    rr.log(f"{cam_entity}/depth_rel", rr.DepthImage(depth_arr.astype(np.float32)))
            except Exception as e:
                print(f"[!] Could not load depth map '{d_file}': {e}")


def log_point_clouds(artifacts: Dict, max_points: int = 500000, timeline: bool = False):
    """Logs all available 3D point clouds across estimation stages."""
    
    stages = [
        ("stage0_mast3r_sfm", artifacts['sfm_plys'], 0),
        ("stage1a_gs_init", [artifacts['gs_init_ply']] if artifacts['gs_init_ply'] else [], 1),
        ("stage1b_gs_base", [artifacts['gs_base_ply']] if artifacts['gs_base_ply'] else [], 2),
        ("stage5a_repair", [artifacts['repair_ply']] if artifacts['repair_ply'] else [], 5),
        ("stage5b_inpainting", [artifacts['inpainting_ply']] if artifacts['inpainting_ply'] else [], 6),
        ("final_reconstruction", [artifacts['final_ply']] if artifacts['final_ply'] else [], 7),
    ]

    for stage_tag, ply_list, stage_step in stages:
        for ply_file in ply_list:
            if not ply_file or not os.path.isfile(ply_file):
                continue

            name = Path(ply_file).stem
            entity_path = f"world/point_clouds/{stage_tag}/{name}"
            print(f"[+] Loading {stage_tag} ({name}) from {ply_file}...")

            if timeline:
                rr.set_time("pipeline_stage", stage_step)

            xyz, rgb, opacities, scales = load_ply_points_and_colors(ply_file, max_points=max_points)
            if len(xyz) == 0:
                continue

            # Calculate radii based on scales if available, or default
            radii = None
            if scales is not None:
                radii = np.mean(scales, axis=-1) * 0.5
                radii = np.clip(radii, 0.001, 0.05)

            rr.log(
                entity_path,
                rr.Points3D(
                    positions=xyz,
                    colors=rgb,
                    radii=radii
                )
            )
            print(f"    -> Logged {len(xyz)} points to '{entity_path}'")

    # Log Leave-One-Out Models
    for l_idx, loo_dir in enumerate(artifacts['loo_dirs']):
        loo_plys = glob.glob(f"{loo_dir}/**/point_cloud.ply", recursive=True)
        if loo_plys:
            leave_name = Path(loo_dir).name
            entity_path = f"world/leave_one_out/{leave_name}"
            if timeline:
                rr.set_time("pipeline_stage", 3)
                rr.set_time("loo_view", l_idx)

            xyz, rgb, _, _ = load_ply_points_and_colors(loo_plys[0], max_points=max_points)
            if len(xyz) > 0:
                rr.log(entity_path, rr.Points3D(positions=xyz, colors=rgb))
                print(f"    -> Logged LOO {leave_name} ({len(xyz)} points)")


def log_mast3r_pointmaps(artifacts: Dict, min_conf: float = 1.0, max_points: int = 200000):
    """Logs per-camera MASt3R dense pointmaps with confidence filtering."""
    if not artifacts['pointmaps']:
        return

    print(f"[+] Logging {len(artifacts['pointmaps'])} MASt3R pointmap(s)...")
    for pm_path in artifacts['pointmaps']:
        name = Path(pm_path).stem
        entity_path = f"world/mast3r_pointmaps/{name}"
        try:
            with open(pm_path, 'r') as f:
                data = json.load(f)

            pts = np.array(data['points'], dtype=np.float32)  # [H, W, 3]
            rgb = np.array(data['rgb'], dtype=np.float32) if data.get('rgb') is not None else None  # [H, W, 3]
            confs = np.array(data['confs'], dtype=np.float32) if data.get('confs') is not None else None  # [H, W]

            pts_flat = pts.reshape(-1, 3)
            mask = np.ones(len(pts_flat), dtype=bool)

            if confs is not None:
                mask = (confs.reshape(-1) >= min_conf)
                pts_flat = pts_flat[mask]

            if rgb is not None:
                rgb_flat = rgb.reshape(-1, 3)[mask]
                if rgb_flat.max() <= 1.0:
                    rgb_flat = (rgb_flat * 255.0).astype(np.uint8)
                else:
                    rgb_flat = rgb_flat.astype(np.uint8)
            else:
                rgb_flat = np.full((len(pts_flat), 3), 200, dtype=np.uint8)

            # Subsample
            if 0 < max_points < len(pts_flat):
                sub = np.random.choice(len(pts_flat), size=max_points, replace=False)
                pts_flat = pts_flat[sub]
                rgb_flat = rgb_flat[sub]

            if len(pts_flat) > 0:
                rr.log(entity_path, rr.Points3D(positions=pts_flat, colors=rgb_flat))
                print(f"    -> Logged pointmap for {name} ({len(pts_flat)} points, min_conf={min_conf})")
        except Exception as e:
            print(f"[!] Error loading pointmap '{pm_path}': {e}")


def log_diff_statistics(artifacts: Dict):
    """Logs Leave-One-Out parameter noise distribution statistics (diffs.pkl)."""
    if not artifacts['diffs_pkl']:
        return

    print(f"[+] Logging LOO distribution stats from {artifacts['diffs_pkl']}...")
    try:
        import pickle
        with open(artifacts['diffs_pkl'], 'rb') as f:
            diffs = pickle.load(f)

        for param_name, (mean_val, std_val) in diffs.items():
            scalar_path = f"metrics/loo_diffs/{param_name}"
            mean_norm = float(np.linalg.norm(mean_val))
            std_norm = float(np.linalg.norm(std_val))
            rr.log(f"{scalar_path}/mean_norm", rr.Scalar(mean_norm))
            rr.log(f"{scalar_path}/std_norm", rr.Scalar(std_norm))
            print(f"    -> Metric {param_name}: mean_norm={mean_norm:.4f}, std_norm={std_norm:.4f}")
    except Exception as e:
        print(f"[!] Could not parse diffs.pkl: {e}")


# =============================================================================
# Main CLI & Execution Flow
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RI3D Rerun Intermediate & Reconstruction Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", "--output_dir", "-o",
        dest="input_path",
        type=str,
        default="output/sceneA",
        help="Path to output root, scene output directory, or intermediate artifact folder.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=500000,
        help="Maximum number of 3D points per cloud to render in Rerun (0 for unlimited).",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=1.0,
        help="Minimum confidence threshold for filtering MASt3R dense pointmaps.",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Structure artifacts along a timeline sequence for progressive scrubber replay.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save Rerun recording directly to a .rrd file (e.g. --save scene.rrd).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind when running with --serve (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Host a web viewer server preloaded with the recording for remote browser inspection.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9876,
        help="Port number when running with --serve (default: 9876).",
    )
    parser.add_argument(
        "--spawn",
        action="store_true",
        default=True,
        help="Spawn native desktop Rerun viewer if graphical display is available.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and list discovered artifacts without sending them to Rerun.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_path)

    if not os.path.exists(input_path):
        print(f"[!] Error: Specified input path does not exist: {input_path}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  RI3D Rerun Visualizer: {input_path}")
    print(f"{'='*70}\n")

    # 1. Discover artifacts
    artifacts = discover_pipeline_artifacts(input_path)

    print("[*] Discovered Pipeline Artifacts:")
    print(f"  • SfM Directory:        {artifacts['sfm_dir'] or 'None'}")
    print(f"  • SfM Point Clouds:     {len(artifacts['sfm_plys'])} file(s)")
    print(f"  • Pointmaps:            {len(artifacts['pointmaps'])} file(s)")
    print(f"  • Relative Depths:      {len(artifacts['depth_rel'])} file(s)")
    print(f"  • Camera Poses/Images:  {len(artifacts['images'])} view(s)")
    print(f"  • Stage 1a (GS Init):   {artifacts['gs_init_ply'] or 'None'}")
    print(f"  • Stage 1b (Base 3DGS): {artifacts['gs_base_ply'] or 'None'}")
    print(f"  • Stage 2 (LOO Dirs):   {len(artifacts['loo_dirs'])} folder(s)")
    print(f"  • LOO Diff Stats:       {artifacts['diffs_pkl'] or 'None'}")
    print(f"  • Stage 5a (Repair):    {artifacts['repair_ply'] or 'None'}")
    print(f"  • Stage 5b (Inpaint):   {artifacts['inpainting_ply'] or 'None'}")
    print(f"  • Final Reconstruction: {artifacts['final_ply'] or 'None'}\n")

    if args.dry_run:
        print("[✓] Dry run complete. No viewer launched.")
        return

    # 2. Determine output mode (save .rrd, serve web viewer, or native GUI)
    is_headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET"))
    should_serve = args.serve or is_headless

    app_id = f"RI3D_{Path(input_path).name}"
    rr.init(app_id, spawn=False)

    # If saving to disk or serving via web server, save to .rrd file
    rrd_target = None
    if args.save:
        rrd_target = os.path.abspath(args.save)
    elif should_serve:
        rrd_target = os.path.abspath(os.path.join(input_path, f"{Path(input_path).stem}_viz.rrd"))

    if rrd_target:
        print(f"[+] Recording Rerun stream to: {rrd_target}")
        rr.save(rrd_target)
    else:
        print("[+] Spawning native desktop Rerun viewer...")
        try:
            rr.spawn()
        except Exception as e:
            print(f"[!] Could not spawn desktop GUI ({e}). Falling back to recording .rrd...")
            rrd_target = os.path.abspath(os.path.join(input_path, f"{Path(input_path).stem}_viz.rrd"))
            rr.save(rrd_target)
            should_serve = True

    # 3. Log cameras and 2D views
    log_camera_and_views(artifacts, timeline=args.timeline)

    # 4. Log MASt3R dense pointmaps
    log_mast3r_pointmaps(artifacts, min_conf=args.min_conf, max_points=args.max_points // 2)

    # 5. Log 3D point clouds & Gaussian Splats
    log_point_clouds(artifacts, max_points=args.max_points, timeline=args.timeline)

    # 6. Log parameter perturbation stats
    log_diff_statistics(artifacts)

    print(f"\n[✓] Visualization data successfully generated.")

    # 7. If serving, launch the Rerun Web Viewer hosting the .rrd over HTTP/WebSocket
    if should_serve and rrd_target and os.path.isfile(rrd_target):
        print(f"\n{'='*70}")
        print(f"  [+] Starting Rerun Web Viewer Server on {args.host}:{args.port}")
        print(f"  [+] Serving recording: {rrd_target}")
        print(f"  [+] Open in browser:  http://localhost:{args.port}  (or http://<remote-ip>:{args.port})")
        print(f"{'='*70}\n")

        # Use python -m rerun to host both the wasm web app and the streaming source
        import subprocess
        cmd = [
            sys.executable,
            "-m",
            "rerun",
            rrd_target,
            "--web-viewer",
            "--web-viewer-port",
            str(args.port),
            "--port",
            str(args.port + 1),
            "--bind",
            str(args.host),
        ]
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n[+] Rerun web server stopped.")
    elif not should_serve:
        print(f"    Explore 3D point clouds, cameras, depth maps, and stages in the viewer.")


if __name__ == "__main__":
    main()
