#!/usr/bin/env python3
"""
RI3D Rerun Visualization Tool (viz.py)
======================================
Interactive 3D/2D visualizer for the RI3D 3D Gaussian Splatting pipeline
and SfM (GGPT-SfM & MASt3R-SfM) intermediate and final estimation artifacts using Rerun.

Usage examples:
    # Visualize an output directory with Rerun viewer (auto-detects GGPT/MASt3R)
    python viz.py -i output/sceneA
    python viz.py -i output/sceneB

    # Visualize a specific subfolder or PLY
    python viz.py -i output/sceneB/ggpt_sfm
    python viz.py -i output/sceneA/mast3r_sfm
    python viz.py -i output/gs_init/sceneB_9

    # Save to a .rrd recording file
    python viz.py -i output/sceneB --save sceneB_viz.rrd

    # Host a web viewer server for remote browser viewing
    python viz.py -i output/sceneB --serve --port 9876

    # Scrub through reconstruction progression along pipeline timeline
    python viz.py -i output/sceneB --timeline
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


def log_scalar(entity_path: str, value: float):
    """Log scalar metric compatible across rerun-sdk versions."""
    try:
        if hasattr(rr, "TimeSeriesScalar"):
            rr.log(entity_path, rr.TimeSeriesScalar(value))
        elif hasattr(rr, "Scalar"):
            rr.log(entity_path, rr.Scalar(value))
        elif hasattr(rr, "log_scalar"):
            rr.log_scalar(entity_path, value)
    except Exception as e:
        print(f"[!] Could not log scalar to '{entity_path}': {e}")


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

    transforms_dir = os.path.dirname(os.path.abspath(transforms_path))
    frames = []
    w = data.get('w', 512)
    h = data.get('h', 512)
    fl_x = data.get('fl_x', data.get('camera_angle_x', None))
    fl_y = data.get('fl_y', fl_x)

    if fl_x is not None and isinstance(fl_x, float) and fl_x < 10.0:
        fl_x = 0.5 * w / math.tan(0.5 * fl_x)
        fl_y = fl_x

    cx = data.get('cx', w / 2.0)
    cy = data.get('cy', h / 2.0)

    for idx, frame in enumerate(data.get('frames', [])):
        c2w = np.array(frame['transform_matrix'], dtype=np.float32)
        # transforms.json stores c2w in OpenGL (+y up, -z forward) convention.
        # rr.Pinhole defaults to RDF (+y down, +z forward), so flip y and z.
        c2w = c2w @ np.diag([1, -1, -1, 1]).astype(np.float32)
        frame_fl_x = frame.get('fl_x', fl_x)
        frame_fl_y = frame.get('fl_y', fl_y)
        frame_w = frame.get('w', w)
        frame_h = frame.get('h', h)
        frame_cx = frame.get('cx', cx)
        frame_cy = frame.get('cy', cy)

        img_path = frame.get('file_path', f"frame_{idx}")
        if not os.path.isabs(img_path):
            resolved = os.path.join(transforms_dir, img_path)
            if os.path.exists(resolved):
                img_path = resolved

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
    """
    Parses cameras.json from either:
    1. SfM export (MASt3R or GGPT): dict with 'filepaths', 'focals' / 'focals_xy', 'cams2world' (OpenCV c2w)
    2. 3D Gaussian Splatting training export: list of dicts with 'id', 'img_name', 'width', 'height', 'fx', 'fy', 'rotation', 'position'
    """
    with open(cameras_json_path, 'r') as f:
        data = json.load(f)

    frames = []
    if isinstance(data, list):
        # 3DGS training format
        for item in data:
            img_name = item.get('img_name', f"cam_{item.get('id', 0)}")
            width = item.get('width', 512)
            height = item.get('height', 512)
            fx = item.get('fx', 500.0)
            fy = item.get('fy', fx)

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
    elif isinstance(data, dict):
        # SfM export format (MASt3R / GGPT)
        filepaths = data.get('filepaths', [])
        cams2world = data.get('cams2world', [])
        focals = data.get('focals', [])
        focals_xy = data.get('focals_xy', None)
        principal_point = data.get('principal_point', None)
        img_size = data.get('image_size', None)

        n_views = len(filepaths) if filepaths else len(cams2world)
        for i in range(n_views):
            fp = filepaths[i] if i < len(filepaths) else f"frame_{i}"
            img_name = Path(fp).stem
            c2w = np.array(cams2world[i], dtype=np.float32) if i < len(cams2world) else np.eye(4, dtype=np.float32)

            if focals_xy and i < len(focals_xy):
                fx, fy = float(focals_xy[i][0]), float(focals_xy[i][1])
            elif focals and i < len(focals):
                fx = fy = float(focals[i])
            else:
                fx = fy = 500.0

            if img_size and len(img_size) >= 2:
                w, h = int(img_size[0]), int(img_size[1])
            else:
                w, h = 512, 512
                if os.path.isfile(fp):
                    try:
                        with Image.open(fp) as im:
                            w, h = im.size
                    except Exception:
                        pass

            if principal_point and len(principal_point) >= 2:
                cx, cy = float(principal_point[0]), float(principal_point[1])
            else:
                cx, cy = w / 2.0, h / 2.0

            frames.append({
                'name': img_name,
                'file_path': fp,
                'c2w': c2w,
                'intrinsics': {
                    'width': int(w),
                    'height': int(h),
                    'fl_x': float(fx),
                    'fl_y': float(fy),
                    'cx': float(cx),
                    'cy': float(cy),
                }
            })
    return frames


# =============================================================================
# Artifact Discovery
# =============================================================================

def discover_pipeline_artifacts(root_path: str, sfm_backend: str = "auto") -> Dict[str, any]:
    """
    Scans the given folder (and its parent/children) to locate all available
    intermediate and final estimation artifacts from both GGPT-SfM and MASt3R-SfM.
    """
    root = os.path.abspath(root_path)

    # Determine scene directory and scene name
    if os.path.basename(root) in ("ggpt_sfm", "mast3r_sfm") or os.path.basename(root).endswith("_sfm"):
        scene_dir = os.path.dirname(root)
        scene_name = os.path.basename(scene_dir)
    else:
        scene_dir = root
        scene_name = os.path.basename(root)

    base_scene_name = scene_name.replace("_ggpt", "").replace("_mast3r", "")

    artifacts = {
        'root': root,
        'scene_name': scene_name,
        'sfm_dir': None,
        'sfm_dirs': [],
        'sfm_plys': [],
        'pointmaps': [],
        'depth_rel': [],
        'flow_arrays': {},
        'transforms_json': None,
        'cameras_json': None,
        'images': [],
        'sfm_stats': [],
        'ba_matches_imgs': [],
        'gs_init_ply': None,
        'gs_base_ply': None,
        'loo_dirs': [],
        'diffs_pkl': None,
        'repair_ply': None,
        'inpainting_ply': None,
        'final_ply': None,
    }

    # 1. Discover SfM directories (GGPT, MASt3R, generic)
    candidate_sfm_dirs = []

    if os.path.basename(root) in ("ggpt_sfm", "mast3r_sfm") or os.path.basename(root).endswith("_sfm") or os.path.isdir(os.path.join(root, "pointmaps")):
        backend = "ggpt_sfm" if "ggpt" in os.path.basename(root) else ("mast3r_sfm" if "mast3r" in os.path.basename(root) else "sfm")
        candidate_sfm_dirs.append((root, backend))
    else:
        possible_subs = []
        if sfm_backend in ("ggpt", "auto"):
            possible_subs.append(("ggpt_sfm", "ggpt_sfm"))
        if sfm_backend in ("mast3r", "auto"):
            possible_subs.append(("mast3r_sfm", "mast3r_sfm"))

        for sub_name, tag in possible_subs:
            p = os.path.join(scene_dir, sub_name)
            if os.path.isdir(p):
                candidate_sfm_dirs.append((p, tag))

        # Check for any other *_sfm directories
        if os.path.isdir(scene_dir):
            for item in sorted(os.listdir(scene_dir)):
                full_p = os.path.join(scene_dir, item)
                if os.path.isdir(full_p) and item.endswith("_sfm") and (full_p, item) not in candidate_sfm_dirs:
                    candidate_sfm_dirs.append((full_p, item))

    artifacts['sfm_dirs'] = candidate_sfm_dirs
    if candidate_sfm_dirs:
        artifacts['sfm_dir'] = candidate_sfm_dirs[0][0]

    # Collect artifacts across all discovered SfM backends
    for sfm_p, backend_tag in candidate_sfm_dirs:
        # PLYs in SfM: points.ply, chart_pcd.ply (excluding camera-frame point_cloud.ply)
        for ply_name in ["points.ply", "chart_pcd.ply"]:
            p = os.path.join(sfm_p, ply_name)
            if os.path.isfile(p):
                artifacts['sfm_plys'].append((p, f"stage0_{backend_tag}"))

        # Pointmaps
        pm_dir = os.path.join(sfm_p, "pointmaps")
        if os.path.isdir(pm_dir):
            backend_label = "ggpt" if "ggpt" in backend_tag else ("mast3r" if "mast3r" in backend_tag else "sfm")
            for f in sorted(os.listdir(pm_dir)):
                if f.endswith('.json') and not os.path.isdir(os.path.join(pm_dir, f)):
                    artifacts['pointmaps'].append((os.path.join(pm_dir, f), backend_label))

        # Relative Depth maps
        d_rel = os.path.join(sfm_p, "depth_rel")
        if os.path.isdir(d_rel):
            for f in sorted(os.listdir(d_rel)):
                if f.endswith('.npy') and not os.path.isdir(os.path.join(d_rel, f)):
                    artifacts['depth_rel'].append(os.path.join(d_rel, f))

        # Flow arrays (confs*.npy, depths*.npy)
        for f in os.listdir(sfm_p):
            if f.startswith('confs') and f.endswith('.npy'):
                artifacts['flow_arrays']['confs'] = os.path.join(sfm_p, f)
            elif f.startswith('depths') and f.endswith('.npy'):
                artifacts['flow_arrays']['depths'] = os.path.join(sfm_p, f)

        # Poses
        t_json = os.path.join(sfm_p, "transforms.json")
        if os.path.isfile(t_json) and not artifacts['transforms_json']:
            artifacts['transforms_json'] = t_json

        c_json = os.path.join(sfm_p, "cameras.json")
        if os.path.isfile(c_json) and not artifacts['cameras_json']:
            artifacts['cameras_json'] = c_json

        # Images
        img_dir = os.path.join(sfm_p, "images")
        if os.path.isdir(img_dir) and not artifacts['images']:
            for f in sorted(os.listdir(img_dir)):
                if f.endswith(VALID_IMAGE_EXTS) and not os.path.isdir(os.path.join(img_dir, f)):
                    artifacts['images'].append(os.path.join(img_dir, f))

        # GGPT Stats & Matching Diagnostics
        stats_p = os.path.join(sfm_p, "ggpt_sfm_stats.json")
        if os.path.isfile(stats_p):
            artifacts['sfm_stats'].append(stats_p)

        matches_img = os.path.join(sfm_p, "matches_for_ba.png")
        if os.path.isfile(matches_img):
            artifacts['ba_matches_imgs'].append(matches_img)

    # Fallback to scene_dir for transforms / cameras / images if not inside SfM dir
    if not artifacts['transforms_json'] and os.path.isfile(os.path.join(scene_dir, "transforms.json")):
        artifacts['transforms_json'] = os.path.join(scene_dir, "transforms.json")
    if not artifacts['cameras_json'] and os.path.isfile(os.path.join(scene_dir, "cameras.json")):
        artifacts['cameras_json'] = os.path.join(scene_dir, "cameras.json")
    if not artifacts['images'] and os.path.isdir(os.path.join(scene_dir, "images")):
        for f in sorted(os.listdir(os.path.join(scene_dir, "images"))):
            if f.endswith(VALID_IMAGE_EXTS) and not os.path.isdir(os.path.join(scene_dir, "images", f)):
                artifacts['images'].append(os.path.join(scene_dir, "images", f))

    # 2. Gaussian Initialization (Stage 1a)
    init_candidates = [
        os.path.join(scene_dir, "debug", "gs_init"),
        "debug/gs_init",
        os.path.join(scene_dir, "debug_gs_init"),
    ]
    for d in init_candidates:
        if os.path.isdir(d):
            # Scene-scoped only. The shared `debug/gs_init` holds every scene ever
            # trained, so an unscoped `{d}/**/point_cloud.ply` fallback would drop a
            # different scene's 2.4M-point cloud into this world -- at a different
            # SfM gauge, with no warning and nothing on screen to say so.
            plys = glob.glob(f"{d}/{base_scene_name}_*/**/point_cloud.ply", recursive=True)
            if plys:
                artifacts['gs_init_ply'] = plys[0]
                break
            if glob.glob(f"{d}/**/point_cloud.ply", recursive=True):
                print(f"[!] {d} has stage-1a clouds, but none for scene '{base_scene_name}'. "
                      "Skipping rather than showing another scene's geometry.")

    # 3. Base 3DGS Training (Stage 1b)
    gs_candidates = [
        os.path.join(scene_dir, "output", "gs_init"),
        "output/gs_init",
        os.path.join(scene_dir, "gs_init"),
    ]
    for d in gs_candidates:
        if os.path.isdir(d):
            # Scene-scoped only -- see the stage-1a note above.
            matched_subs = [s for s in os.listdir(d) if "_loo_" not in s and base_scene_name in s and os.path.isdir(os.path.join(d, s))]
            if not matched_subs and any(
                "_loo_" not in s and os.path.isdir(os.path.join(d, s)) for s in os.listdir(d)
            ):
                print(f"[!] {d} has stage-1b models, but none for scene '{base_scene_name}'. "
                      "Skipping rather than showing another scene's geometry.")
            for scene_sub in matched_subs:
                iter_plys = glob.glob(os.path.join(d, scene_sub, "point_cloud", "iteration_*", "point_cloud.ply"))
                if iter_plys:
                    def _iter_num(p):
                        try:
                            return int(Path(p).parent.name.replace("iteration_", ""))
                        except Exception:
                            return 0
                    iter_plys.sort(key=_iter_num, reverse=True)
                    artifacts['gs_base_ply'] = iter_plys[0]
                cam_json = os.path.join(d, scene_sub, "cameras.json")
                if os.path.isfile(cam_json) and not artifacts['cameras_json']:
                    artifacts['cameras_json'] = cam_json

    # 4. Leave-One-Out (Stage 2)
    for d in gs_candidates:
        if os.path.isdir(d):
            for scene_sub in os.listdir(d):
                if "_loo_" in scene_sub and base_scene_name in scene_sub and os.path.isdir(os.path.join(d, scene_sub)):
                    loo_root = os.path.join(d, scene_sub)
                    diffs_p = os.path.join(loo_root, "diffs.pkl")
                    if os.path.isfile(diffs_p):
                        artifacts['diffs_pkl'] = diffs_p
                    for leave_dir in sorted(os.listdir(loo_root)):
                        full_leave = os.path.join(loo_root, leave_dir)
                        if os.path.isdir(full_leave) and leave_dir.startswith("leave_"):
                            artifacts['loo_dirs'].append(full_leave)

    # 5. Stage 5a Repair & 5b Inpainting
    for den_dir in glob.glob(f"output_den*/**/{base_scene_name}_*") + glob.glob(f"{scene_dir}/output_den*/**/{base_scene_name}_*"):
        last_ply = glob.glob(f"{den_dir}/**/last.ply", recursive=True)
        if last_ply:
            artifacts['repair_ply'] = last_ply[0]
            break

    for inp_dir in glob.glob(f"output_inp*/**/{base_scene_name}_*") + glob.glob(f"{scene_dir}/output_inp*/**/{base_scene_name}_*"):
        last_ply = glob.glob(f"{inp_dir}/**/last.ply", recursive=True)
        if last_ply:
            artifacts['inpainting_ply'] = last_ply[0]
            break

    # 6. Final reconstructed PLYs in scene_dir
    final_plys = glob.glob(f"{scene_dir}/*_reconstructed_3dgs.ply") + glob.glob(f"{scene_dir}/*.ply")
    final_plys = [p for p in final_plys if not (os.path.basename(p) in ("points.ply", "chart_pcd.ply", "point_cloud.ply"))]
    if final_plys:
        artifacts['final_ply'] = final_plys[0]

    return artifacts


# =============================================================================
# Rerun Logging Routines
# =============================================================================

def log_camera_and_views(artifacts: Dict, timeline: bool = False):
    """Logs cameras, images, and depth priors to Rerun."""
    frames = []
    if artifacts['transforms_json']:
        frames = parse_transforms_json(artifacts['transforms_json'])
    elif artifacts['cameras_json']:
        frames = parse_cameras_json(artifacts['cameras_json'])

    if not frames:
        for idx, img_p in enumerate(artifacts['images']):
            frames.append({
                'name': Path(img_p).stem,
                'file_path': img_p,
                'c2w': None,
                'intrinsics': {'width': 512, 'height': 512, 'fl_x': 500, 'fl_y': 500, 'cx': 256, 'cy': 256}
            })

    print(f"[+] Logging {len(frames)} camera view(s)...")

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
    stages = []
    for sfm_ply_item in artifacts.get('sfm_plys', []):
        if isinstance(sfm_ply_item, tuple):
            ply_file, backend_tag = sfm_ply_item
        else:
            ply_file, backend_tag = sfm_ply_item, "stage0_sfm"
        stages.append((backend_tag, [ply_file], 0))

    stages.extend([
        ("stage1a_gs_init", [artifacts['gs_init_ply']] if artifacts['gs_init_ply'] else [], 1),
        ("stage1b_gs_base", [artifacts['gs_base_ply']] if artifacts['gs_base_ply'] else [], 2),
        ("stage5a_repair", [artifacts['repair_ply']] if artifacts['repair_ply'] else [], 5),
        ("stage5b_inpainting", [artifacts['inpainting_ply']] if artifacts['inpainting_ply'] else [], 6),
        ("final_reconstruction", [artifacts['final_ply']] if artifacts['final_ply'] else [], 7),
    ])

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


def log_sfm_pointmaps(artifacts: Dict, min_conf: Optional[float] = None, max_points: int = 200000):
    """
    Logs per-camera dense pointmaps (from GGPT-SfM or MASt3R-SfM) with confidence filtering
    and DLT multi-view triangulated anchor points.
    """
    if not artifacts.get('pointmaps'):
        return

    print(f"[+] Logging {len(artifacts['pointmaps'])} SfM pointmap(s)...")

    stats_conf_thr = None
    if artifacts.get('sfm_stats'):
        try:
            with open(artifacts['sfm_stats'][0], 'r') as f:
                sdata = json.load(f)
                stats_conf_thr = sdata.get('conf_thr')
        except Exception:
            pass

    for item in artifacts['pointmaps']:
        if isinstance(item, tuple):
            pm_path, backend = item
        else:
            pm_path, backend = item, "sfm"

        name = Path(pm_path).stem
        entity_path = f"world/pointmaps/{backend}/{name}"
        anchor_entity_path = f"world/pointmaps/{backend}/dlt_anchors/{name}"
        try:
            with open(pm_path, 'r') as f:
                data = json.load(f)

            pts = np.array(data['points'], dtype=np.float32)
            pts_flat = pts.reshape(-1, 3)
            n_total = len(pts_flat)

            confs = np.array(data['confs'], dtype=np.float32) if data.get('confs') is not None else None
            rgb = np.array(data['rgb'], dtype=np.float32) if data.get('rgb') is not None else None
            dlt_mask = np.array(data['dlt_mask'], dtype=np.uint8) if data.get('dlt_mask') is not None else None

            # Determine confidence threshold to apply
            thr = min_conf
            if thr is None:
                if "ggpt" in backend:
                    thr = stats_conf_thr if stats_conf_thr is not None else 1.0
                else:
                    thr = 1.0

            mask = np.ones(n_total, dtype=bool)
            if confs is not None and thr is not None and thr > 0.0:
                confs_flat = confs.reshape(-1)
                if len(confs_flat) == n_total:
                    mask = (confs_flat >= thr)

            filtered_pts = pts_flat[mask]

            if rgb is not None:
                rgb_flat = rgb.reshape(-1, 3)
                if len(rgb_flat) == n_total:
                    rgb_flat = rgb_flat[mask]
                else:
                    rgb_flat = np.full((len(filtered_pts), 3), 200, dtype=np.uint8)

                if len(rgb_flat) > 0 and rgb_flat.max() <= 1.0:
                    rgb_flat = (rgb_flat * 255.0).astype(np.uint8)
                else:
                    rgb_flat = rgb_flat.astype(np.uint8)
            else:
                rgb_flat = np.full((len(filtered_pts), 3), 200, dtype=np.uint8)

            if 0 < max_points < len(filtered_pts):
                sub = np.random.choice(len(filtered_pts), size=max_points, replace=False)
                filtered_pts = filtered_pts[sub]
                rgb_flat = rgb_flat[sub]

            if len(filtered_pts) > 0:
                rr.log(entity_path, rr.Points3D(positions=filtered_pts, colors=rgb_flat))
                print(f"    -> Logged pointmap for {backend}/{name} ({len(filtered_pts)} points, thr={thr:.2f})")

            # If dlt_mask is present (GGPT SfM), log triangulated DLT anchors
            if dlt_mask is not None:
                dlt_flat = dlt_mask.reshape(-1)
                if len(dlt_flat) == n_total:
                    dlt_pts = pts_flat[dlt_flat > 0]
                    if len(dlt_pts) > 0:
                        if 0 < max_points < len(dlt_pts):
                            sub_d = np.random.choice(len(dlt_pts), size=max_points, replace=False)
                            dlt_pts = dlt_pts[sub_d]
                        dlt_colors = np.full((len(dlt_pts), 3), [255, 215, 0], dtype=np.uint8)
                        rr.log(anchor_entity_path, rr.Points3D(positions=dlt_pts, colors=dlt_colors, radii=0.008))
                        print(f"       DLT anchors: logged {len(dlt_pts)} triangulated points to '{anchor_entity_path}'")
        except Exception as e:
            print(f"[!] Error loading pointmap '{pm_path}': {e}")


def log_mast3r_pointmaps(artifacts: Dict, min_conf: Optional[float] = 1.0, max_points: int = 200000):
    """Backward-compatible alias for log_sfm_pointmaps."""
    return log_sfm_pointmaps(artifacts, min_conf=min_conf, max_points=max_points)


def log_sfm_stats(artifacts: Dict):
    """Logs bundle adjustment, reprojection, and scale fusion statistics from GGPT-SfM (ggpt_sfm_stats.json)."""
    if not artifacts.get('sfm_stats'):
        return

    for stats_path in artifacts['sfm_stats']:
        print(f"[+] Logging GGPT-SfM solve statistics from {stats_path}...")
        try:
            with open(stats_path, 'r') as f:
                data = json.load(f)

            prefix = "metrics/ggpt_sfm"
            if "backend" in data:
                print(f"    • Backend: {data['backend']}")
            if "conf_thr" in data:
                log_scalar(f"{prefix}/conf_thr", float(data['conf_thr']))
                print(f"    • Conf Floor: {data['conf_thr']:.4f}")

            # Bundle Adjustment metrics
            ba = data.get('bundle_adjustment', {})
            if 'reproj_p50_px' in ba:
                log_scalar(f"{prefix}/ba/reproj_p50_px", float(ba['reproj_p50_px']))
                print(f"    • BA Reprojection p50: {ba['reproj_p50_px']:.4f} px")
            if 'reproj_p90_px' in ba:
                log_scalar(f"{prefix}/ba/reproj_p90_px", float(ba['reproj_p90_px']))
                print(f"    • BA Reprojection p90: {ba['reproj_p90_px']:.4f} px")

            # Per-view BA metrics
            for pv in ba.get('per_view', []):
                v_idx = pv.get('view', 0)
                v_name = data.get('views', [])[v_idx] if v_idx < len(data.get('views', [])) else f"view_{v_idx}"
                if 'p50_px' in pv:
                    log_scalar(f"{prefix}/per_view/{v_name}/reproj_p50_px", float(pv['p50_px']))
                if 'p90_px' in pv:
                    log_scalar(f"{prefix}/per_view/{v_name}/reproj_p90_px", float(pv['p90_px']))

            # Fusion metrics
            fusion = data.get('fusion', {})
            if 'global_scale' in fusion:
                log_scalar(f"{prefix}/fusion/global_scale", float(fusion['global_scale']))
                print(f"    • Fusion Global Scale: {fusion['global_scale']:.4f}")
            if 'anchor_pixels' in fusion:
                log_scalar(f"{prefix}/fusion/anchor_pixels", float(fusion['anchor_pixels']))
                print(f"    • Triangulated Anchor Pixels: {fusion['anchor_pixels']}")

            # Per-view DLT coverage and scale
            for f_pv in fusion.get('per_view', []):
                v_idx = f_pv.get('view', 0)
                v_name = data.get('views', [])[v_idx] if v_idx < len(data.get('views', [])) else f"view_{v_idx}"
                if 'dlt_coverage' in f_pv:
                    log_scalar(f"{prefix}/per_view/{v_name}/dlt_coverage", float(f_pv['dlt_coverage']))
                if 'view_scale' in f_pv:
                    log_scalar(f"{prefix}/per_view/{v_name}/view_scale", float(f_pv['view_scale']))
        except Exception as e:
            print(f"[!] Could not parse stats '{stats_path}': {e}")


def log_sfm_diagnostics(artifacts: Dict):
    """Logs 2D matching diagnostics such as RoMaV2 matches_for_ba.png."""
    if not artifacts.get('ba_matches_imgs'):
        return

    for img_path in artifacts['ba_matches_imgs']:
        if os.path.isfile(img_path):
            print(f"[+] Logging SfM match diagnostic image from {img_path}...")
            try:
                img = Image.open(img_path).convert("RGB")
                rr.log("world/sfm/matches_for_ba", rr.Image(np.array(img)))
            except Exception as e:
                print(f"[!] Could not load diagnostic image '{img_path}': {e}")


def log_diff_statistics(artifacts: Dict):
    """Logs Leave-One-Out parameter noise distribution statistics (diffs.pkl)."""
    if not artifacts.get('diffs_pkl'):
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
            log_scalar(f"{scalar_path}/mean_norm", mean_norm)
            log_scalar(f"{scalar_path}/std_norm", std_norm)
            print(f"    -> Metric {param_name}: mean_norm={mean_norm:.4f}, std_norm={std_norm:.4f}")
    except Exception as e:
        print(f"[!] Could not parse diffs.pkl: {e}")


# =============================================================================
# Main CLI & Execution Flow
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RI3D Rerun Intermediate & Reconstruction Visualizer (supports GGPT-SfM & MASt3R-SfM)",
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
        "--sfm_backend",
        type=str,
        default="auto",
        choices=["auto", "ggpt", "mast3r"],
        help="SfM backend to visualize: 'auto' (detect all), 'ggpt', or 'mast3r'.",
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
        default=None,
        help="Confidence threshold for filtering pointmaps (defaults to auto/conf_thr from stats).",
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
    artifacts = discover_pipeline_artifacts(input_path, sfm_backend=args.sfm_backend)

    print("[*] Discovered Pipeline Artifacts:")
    print(f"  • Scene Name:           {artifacts['scene_name']}")
    print(f"  • SfM Directory:        {artifacts['sfm_dir'] or 'None'} ({len(artifacts['sfm_dirs'])} backend(s))")
    print(f"  • SfM Point Clouds:     {len(artifacts['sfm_plys'])} file(s)")
    print(f"  • Pointmaps:            {len(artifacts['pointmaps'])} file(s)")
    print(f"  • Relative Depths:      {len(artifacts['depth_rel'])} file(s)")
    print(f"  • Camera Poses/Images:  {len(artifacts['images'])} view(s)")
    print(f"  • SfM Solve Stats:      {len(artifacts['sfm_stats'])} file(s)")
    print(f"  • BA Diagnostics Img:   {len(artifacts['ba_matches_imgs'])} file(s)")
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

    # 2. Determine output mode
    is_headless = not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.environ.get("WAYLAND_SOCKET"))
    should_serve = args.serve or (is_headless and not args.save)

    app_id = f"RI3D_{Path(input_path).name}"
    rr.init(app_id, spawn=False)

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

    # 4. Log SfM dense pointmaps & triangulated DLT anchors
    log_sfm_pointmaps(artifacts, min_conf=args.min_conf, max_points=args.max_points // 2)

    # 5. Log 3D point clouds & Gaussian Splats across pipeline stages
    log_point_clouds(artifacts, max_points=args.max_points, timeline=args.timeline)

    # 6. Log SfM solve statistics & BA diagnostics
    log_sfm_stats(artifacts)
    log_sfm_diagnostics(artifacts)

    # 7. Log LOO parameter perturbation stats
    log_diff_statistics(artifacts)

    print(f"\n[✓] Visualization data successfully generated.")

    # 8. If serving, launch the Rerun Web Viewer hosting the .rrd over HTTP/WebSocket
    if should_serve and rrd_target and os.path.isfile(rrd_target):
        print(f"\n{'='*70}")
        print(f"  [+] Starting Rerun Web Viewer Server on {args.host}:{args.port}")
        print(f"  [+] Serving recording: {rrd_target}")
        print(f"  [+] Open in browser:  http://localhost:{args.port}  (or http://<remote-ip>:{args.port})")
        print(f"{'='*70}\n")

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
