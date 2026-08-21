#!/usr/bin/env python3
"""
RI3D End-to-End Sparse-View 3DGS Inference Pipeline
===================================================
Reconstructs 3D Gaussian Splatting (3DGS) scenes from sparse RGB photos
using SfM for camera pose & pointmap extraction, followed by the RI3D
5-stage Repair & Inpainting Diffusion Priors pipeline.

Two SfM backends write the same contract (--sfm_backend):
  ggpt    (default) VGGT dense pointmap -> RoMaV2 matching -> BA -> DLT
  mast3r  MASt3R-SfM; the only one that can hold a known calibration fixed

Usage Examples:
---------------
1. Unposed sparse photos (extract poses & pointmaps, then run 5-stage RI3D):
   python inference.py -i /path/to/photos -o output/my_scene --sfm_config unposed --num_views 3

2. Pre-calibrated scene (transforms.json already available):
   python inference.py -i data/mipnerf360/bicycle -o output/bicycle_3 --sfm_config posed --num_views 3

3. Fast Stage 1 Gaussian Splatting baseline only:
   python inference.py -i data/mipnerf360/bicycle --stages 1 --num_views 3

4. Specific stages (e.g. Stage 1, 2, 3, 5a, 5b):
   python inference.py -i data/mipnerf360/bicycle --stages 1,2,3,5a,5b --num_views 3

5. Dry-run inspection (print commands without executing):
   python inference.py -i data/mipnerf360/bicycle --dry_run
"""

import os
import sys
import argparse
import time
import json
import shutil
import glob
from pathlib import Path
import numpy as np
from PIL import Image

# Color styling for CLI output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def log_banner(stage_num, total_stages, title):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*75}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}[Stage {stage_num}/{total_stages}] {title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*75}{Colors.ENDC}\n")


def log_success(msg):
    print(f"{Colors.GREEN}[✓] {msg}{Colors.ENDC}")


def log_warn(msg):
    print(f"{Colors.WARNING}[!] {msg}{Colors.ENDC}")


def log_error(msg):
    print(f"{Colors.FAIL}[✗] {msg}{Colors.ENDC}")


def run_command(command, dry_run=False):
    print(f"{Colors.BLUE}$ {command}{Colors.ENDC}")
    if dry_run:
        return True
    
    # Ensure current project directory, mast3r, and third_party submodules are on PYTHONPATH
    env = os.environ.copy()
    ri3d_root = os.path.dirname(os.path.abspath(__file__))
    mast3r_path = os.path.join(ri3d_root, "mast3r")
    minlora_path = os.path.join(ri3d_root, "third_party", "minLoRA")
    clip_path = os.path.join(ri3d_root, "third_party", "CLIP")
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{ri3d_root}:{mast3r_path}:{minlora_path}:{clip_path}:{current_pythonpath}"
    
    import subprocess
    proc = subprocess.run(command, shell=True, env=env)
    if proc.returncode != 0:
        log_error(f"Command failed with return code {proc.returncode}:")
        log_error(f"  {command}")
        sys.exit(proc.returncode)
    return True


def check_checkpoints(args):
    """Verify that required pretrained weights exist, or display download guidance."""
    missing = []

    sd15_ckpt = Path("models/v1-5-pruned.ckpt")
    controlnet_ckpt = Path("models/control_v11f1e_sd15_tile.pth")
    sd2_inp_dir = Path("models/stable-diffusion-2-inpainting")

    if not sd15_ckpt.exists():
        missing.append(("SD 1.5 Pruned Checkpoint", sd15_ckpt, "python scripts/download_hf_models.py"))
    if not controlnet_ckpt.exists():
        missing.append(("ControlNet Tile v1.1", controlnet_ckpt, "python scripts/download_hf_models.py"))
    if not sd2_inp_dir.exists():
        missing.append(("SD2 Inpainting Model", sd2_inp_dir, "python scripts/download_hf_models.py"))

    if args.sfm_config == "unposed" and args.sfm_backend == "mast3r":
        mast3r_ckpt = Path("third_party/MASt3R/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        if not mast3r_ckpt.exists() and not Path("/home/nguyen/projects/G4Splat/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").exists():
            missing.append(("MASt3R Checkpoint", mast3r_ckpt, "wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P ./third_party/MASt3R/checkpoints/"))

    if args.sfm_config == "unposed" and args.sfm_backend == "ggpt":
        ggpt_root = Path(args.ggpt_root)
        if not ggpt_root.is_dir():
            missing.append(("GGPT checkout", ggpt_root, "git clone --recursive https://github.com/chenyutongthu/GGPT.git"))
        if not Path(args.ggpt_python).exists():
            missing.append(("GGPT environment interpreter", Path(args.ggpt_python),
                            "create the env from GGPT/requirements_sfm.txt, then pass --ggpt_python"))
        # VGGT streams from the HF cache; a missing snapshot means a download,
        # not a failure, so this is informational.
        vggt_cache = Path.home() / ".cache/huggingface/hub/models--facebook--VGGT-1B"
        if not vggt_cache.exists():
            log_warn("VGGT-1B is not in the HF cache; the SfM stage will download it on first run.")
        # Only the PTv3 refiner needs the GGPT weights.
        if args.ggpt_refine and not (ggpt_root / "ckpts/model.step228000.pth").exists():
            missing.append(("GGPT PTv3 checkpoint", ggpt_root / "ckpts/model.step228000.pth",
                            "download from https://huggingface.co/YutongGoose/GGPT into GGPT/ckpts/"))

    if missing:
        log_warn("The following required checkpoint(s) were not found:")
        for name, path, dl_cmd in missing:
            print(f"  - {Colors.BOLD}{name}{Colors.ENDC} expected at: {path}")
            print(f"    Download with: {Colors.CYAN}{dl_cmd}{Colors.ENDC}")
        log_warn("Attempting to proceed anyway (if using custom paths or online loading)...")


def validate_and_standardize_images(input_path, auto_orient=True, dry_run=False, num_views=3):
    """
    Check that all input photos have consistent orientations (all landscape or all portrait).
    If mixed orientations are detected (e.g. 1 portrait amongst landscape photos):
    Automatically rotate the outlier image so all images share the dominant aspect ratio.
    """
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    p = Path(input_path)
    if not p.exists() and not dry_run:
        log_error(f"Input directory does not exist: {input_path}")
        sys.exit(1)

    for img_subdir in ["images_4", "images_2", "images_8", "images"]:
        if (p / img_subdir).exists() and (p / img_subdir).is_dir():
            image_dir = p / img_subdir
            break
    else:
        image_dir = p

    image_files = sorted([f for f in image_dir.iterdir() if f.is_file() and f.suffix in valid_exts]) if image_dir.exists() else []

    # If no images directly found, check if transforms.json exists
    transforms_file = p / "transforms.json"
    if len(image_files) == 0 and transforms_file.exists():
        try:
            with open(transforms_file, 'r') as f:
                tf_data = json.load(f)
                frame_paths = [Path(f["file_path"]) for f in tf_data.get("frames", [])]
                image_files = [p / fp if not fp.is_absolute() else fp for fp in frame_paths]
                # Filter to existing if any
                existing = [f for f in image_files if f.exists()]
                if existing:
                    image_files = existing
        except Exception:
            pass

    if len(image_files) == 0:
        if dry_run:
            log_warn("No images found on disk, but continuing in --dry_run mode with simulated views.")
            image_files = [Path(f"image_{i:03d}.png") for i in range(max(num_views, 3))]
            return len(image_files), str(p), str(image_dir), image_files
        return 0, str(p), str(image_dir), []

    sizes = []
    for f in image_files:
        if f.exists():
            with Image.open(f) as img:
                sizes.append((f, img.width, img.height, img.width >= img.height))

    if len(sizes) > 0:
        is_landscape = [s[3] for s in sizes]
        num_landscape = sum(is_landscape)
        num_portrait = len(is_landscape) - num_landscape

        if 0 < num_landscape < len(sizes):
            dominant_is_landscape = num_landscape >= num_portrait
            dominant_type = "landscape (width >= height)" if dominant_is_landscape else "portrait (height > width)"
            log_warn(f"Mixed image orientations detected in {image_dir}: {num_landscape} landscape, {num_portrait} portrait.")

            if auto_orient:
                log_warn(f"Auto-orienting outlier photos to match dominant orientation ({dominant_type})...")
                for f, w, h, is_land in sizes:
                    if is_land != dominant_is_landscape:
                        print(f"  - Auto-rotating {f.name} ({w}x{h}) by 90°...")
                        with Image.open(f) as img:
                            rotated = img.rotate(270, expand=True)
                            rotated.save(f)
                log_success("All input photos standardized to consistent orientation.")
            else:
                log_error("All images in a scene must have consistent orientation.")
                sys.exit(1)

    return len(image_files), str(p), str(image_dir), image_files


def convert_colmap_to_transforms_json(sparse_dir, images_dir, output_transforms_path):
    """Convert COLMAP sparse model (cameras.txt, images.txt) to NeRF transforms.json format."""
    cams_file = os.path.join(sparse_dir, "cameras.txt")
    imgs_file = os.path.join(sparse_dir, "images.txt")
    if not os.path.exists(cams_file) or not os.path.exists(imgs_file):
        return False

    cameras = {}
    with open(cams_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]
            if model == "SIMPLE_PINHOLE" or model == "SIMPLE_RADIAL":
                fl_x = fl_y = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE" or model == "OPENCV":
                fl_x, fl_y = params[0], params[1]
                cx, cy = params[2], params[3]
            else:
                fl_x = fl_y = params[0]
                cx, cy = width / 2.0, height / 2.0
            cameras[cam_id] = {"w": width, "h": height, "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy}

    frames = []
    with open(imgs_file, "r") as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
        for i in range(0, len(lines), 2):
            parts = lines[i].split()
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            img_name = parts[9]

            # Quaternion to Rotation Matrix
            R = np.array([
                [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
            ])
            T = np.array([tx, ty, tz])

            # World to Camera -> Camera to World
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = T
            c2w = np.linalg.inv(w2c)

            # Convert to OpenGL camera coordinates (flip Y and Z)
            _coord_trans = np.diag([1, -1, -1, 1])
            c2w_gl = c2w @ _coord_trans

            frames.append({
                "file_path": f"./images/{img_name}",
                "transform_matrix": c2w_gl.tolist()
            })

    first_cam = list(cameras.values())[0] if cameras else {"w": 512, "h": 512, "fl_x": 512.0, "fl_y": 512.0, "cx": 256.0, "cy": 256.0}
    transforms_data = {
        "w": first_cam["w"],
        "h": first_cam["h"],
        "fl_x": first_cam["fl_x"],
        "fl_y": first_cam["fl_y"],
        "cx": first_cam["cx"],
        "cy": first_cam["cy"],
        "frames": frames
    }

    with open(output_transforms_path, "w") as f:
        json.dump(transforms_data, f, indent=4)
    log_success(f"Exported transforms.json from COLMAP sparse reconstruction: {output_transforms_path}")
    return True


def run_sfm_and_extract_pointmap(source_path, output_scene_dir, args, dry_run=False):
    """
    Recover camera intrinsics, extrinsics, transforms.json and dense pointmaps
    from unposed sparse images.

    Two interchangeable engines write the same output contract (`cameras.json`,
    `sparse/0`, `pointmaps/*.json`, `points.ply`, `images/`), so everything
    downstream is backend-agnostic:

      ggpt    VGGT dense pointmap -> RoMaV2 dense matching -> pycolmap BA ->
              DLT re-triangulation. Runs out-of-process in the GGPT
              environment, which carries romav2.
      mast3r  the original MASt3R-SfM.

    Output lands in `<output_scene_dir>/<backend>_sfm/`, so switching backends
    does not overwrite the other one's solve.
    """
    backend = args.sfm_backend
    label = "GGPT-SfM (VGGT + RoMaV2)" if backend == "ggpt" else "MASt3R-SfM"
    log_success(f"Running {label} camera pose estimation & pointmap recovery...")

    sfm_out_dir = os.path.join(output_scene_dir, f"{backend}_sfm")
    os.makedirs(sfm_out_dir, exist_ok=True)

    local_sfm = "scripts/run_sfm.py"
    g4splat_sfm = "/home/nguyen/projects/G4Splat/scripts/run_sfm.py"

    if os.path.exists(local_sfm):
        cmd = (f"python {local_sfm} --backend {backend} --source_path {source_path} "
               f"--output_path {sfm_out_dir} --config {args.sfm_config} --n_images {args.num_views}")
        if backend == "ggpt":
            cmd += (f" --ggpt_root {args.ggpt_root} --ggpt_python {args.ggpt_python}"
                    f" --fuse_mode {args.fuse_mode}")
            if args.ggpt_refine:
                cmd += " --ggpt_refine"
            if args.watermark_mask:
                cmd += f" --watermark_mask {args.watermark_mask}"
        run_command(cmd, dry_run=dry_run)
    elif backend == "mast3r" and os.path.exists(g4splat_sfm):
        cmd = (f"cd /home/nguyen/projects/G4Splat && python scripts/run_sfm.py "
               f"--source_path {source_path} --output_path {sfm_out_dir} "
               f"--config {args.sfm_config} --n_images {args.num_views}")
        run_command(cmd, dry_run=dry_run)
    else:
        log_warn("No run_sfm.py found; checking local tools...")
        sparse_pc_tool = "tools/sparse_pc.py"
        if os.path.exists(sparse_pc_tool):
            cmd = f"python {sparse_pc_tool} --source_path {source_path} --output_path {output_scene_dir}"
            run_command(cmd, dry_run=dry_run)

    # Convert resulting COLMAP sparse model to transforms.json
    if not dry_run:
        sparse_0 = os.path.join(sfm_out_dir, "sparse/0")
        images_dir = os.path.join(sfm_out_dir, "images")
        tf_out = os.path.join(sfm_out_dir, "transforms.json")
        convert_colmap_to_transforms_json(sparse_0, images_dir, tf_out)

    return sfm_out_dir


def resolve_train_view_names(scene_path, num_views):
    """Reproduce the train-view selection performed by the dataset reader.

    `readColmapSceneInfo` sorts every camera by image_name *before* indexing
    with train_ids, then subsamples with linspace. `depths<N>.npy` /
    `confs<N>.npy` are indexed positionally against that final list, so this
    ordering has to match exactly or the per-view arrays are silently
    mismatched.

    Returns [(image_name, image_path), ...] in reader order.
    """
    tf_path = os.path.join(scene_path, "transforms.json")
    if not os.path.exists(tf_path):
        log_error(f"Missing '{tf_path}'. For unposed data, include 'sfm' in --stages (e.g. '--stages sfm,1').")
        sys.exit(1)

    with open(tf_path, "r") as f:
        transforms = json.load(f)

    entries = []
    for frame in transforms["frames"]:
        rel = frame["file_path"]
        path = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(scene_path, rel))
        entries.append((os.path.basename(path).split(".")[0], path))
    entries.sort(key=lambda e: e[0])

    with open(os.path.join(scene_path, f"train_test_split_{num_views}.json"), "r") as f:
        train_ids = json.load(f)["train_ids"]

    train_entries = [entries[i] for i in train_ids]
    idx_sub = [round(i) for i in np.linspace(0, len(train_entries) - 1, num_views)]
    return [e for idx, e in enumerate(train_entries) if idx in idx_sub]


def _depth_from_pointmap(scene_path, name, K, width, height, c2w):
    """Dense metric depth taken straight from the SfM pointmap.

    The pointmap is already a dense depth image on its own grid, so resampling
    it to the target resolution keeps every value MASt3R produced. Projecting
    the points individually instead scatters them into a finer grid and leaves
    most target pixels empty (~20% coverage at 2x upsampling), which is what
    forces the monocular prior to invent the remaining 80%.
    """
    from utils.pointmap_utils import resample_view_depth, render_view_depth, fill_holes_nearest

    resampled = resample_view_depth(scene_path, name, K, width, height, c2w=c2w)
    if resampled is not None:
        depth, conf, valid = resampled
        # MASt3R occasionally places a pixel behind the camera (z <= 0). Those
        # are rare and low-confidence, but writing one into depth_rel would
        # back-project a point behind the view, and the disparity guard in the
        # reader filters to z > 0 before taking its median so it would not warn.
        if not valid.all():
            depth = fill_holes_nearest(depth, valid)
        return depth, conf, valid, "resampled"

    # MASt3R centre-cropped this view, so the grid no longer maps linearly onto
    # the image; fall back to per-point projection plus a nearest-neighbour fill.
    depth, conf, valid = render_view_depth(scene_path, name, K, width, height, c2w=c2w)
    if not valid.any():
        raise ValueError(f"{name}: pointmap projects to no valid pixel in the target view")
    return fill_holes_nearest(depth, valid), conf, valid, "projected+filled"


def generate_aligned_depth(scene_path, image_dir, num_views, depth_source="pointmap"):
    """Write metric depth priors in SfM world units for the sparse train views.

    The pipeline (point-cloud init, Gaussian radii, every monocular depth loss)
    reads `depth_rel/*.npy` as metric z-depth in MASt3R world units. Two ways to
    produce it:

    * "pointmap" -- resample MASt3R's own dense pointmap depth. Metric by
      construction and multi-view consistent, since it is the same geometry the
      poses came from.
    * "align" -- run Depth-Anything and fit its affine-invariant disparity onto
      the pointmap's metric anchors (`utils.depth_align`). Keeps the monocular
      network's fine detail, at the cost of a global affine that cannot track
      the true disparity relation: the residual carries a depth-dependent bias
      and heavy tails, which bends the far field.

    Returns True when depth was written, False when there is no pointmap to
    work from (e.g. a pre-processed mip-NeRF 360 scene shipping its own
    depth_rel).
    """
    from utils.pointmap_utils import load_target_intrinsics, load_sfm_poses

    pointmaps_dir = os.path.join(scene_path, "pointmaps")
    if not os.path.isdir(pointmaps_dir):
        return False

    K, width, height = load_target_intrinsics(scene_path)
    poses = load_sfm_poses(scene_path)
    train_views = resolve_train_view_names(scene_path, num_views)
    log_success(
        f"Writing depth for {len(train_views)} train view(s) at {width}x{height} "
        f"from '{depth_source}' (target grid from transforms.json)"
    )

    # Both locations are searched downstream: the reader walks a candidate list
    # while threestudio/data/loo_mip.py historically hard-coded the image_dir one.
    depth_dirs = [os.path.join(scene_path, "depth_rel"), os.path.join(image_dir, "depth_rel")]
    for d in depth_dirs:
        os.makedirs(d, exist_ok=True)

    if depth_source == "align":
        import cv2
        import torch
        from torchvision.transforms import ToTensor

        from utils.depth_align import align_mono_to_pointmap
        from utils.depth_utils import estimate_depth

    depths, confs = [], []
    for name, img_path in train_views:
        if name not in poses:
            raise KeyError(f"{name} has no pose in {scene_path}/cameras.json")

        depth, conf, valid, how = _depth_from_pointmap(
            scene_path, name, K, width, height, poses[name]
        )

        if depth_source == "align":
            img = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                disparity = estimate_depth(ToTensor()(img).cuda()).cpu().numpy()
            if disparity.shape != (height, width):
                interp = cv2.INTER_AREA if disparity.shape[0] > height else cv2.INTER_LINEAR
                disparity = cv2.resize(disparity, (width, height), interpolation=interp)

            fitted, info = align_mono_to_pointmap(disparity, depth, valid, conf=conf)
            if fitted is None:
                log_warn(f"{name}: monocular fit rejected ({info['fallback']}); keeping pointmap depth")
            else:
                depth = fitted
                log_success(
                    f"{name}: anchors {valid.mean()*100:.1f}%  inliers {info['inlier_ratio']*100:.0f}%"
                    f"  median err {info['median_rel_err']*100:.2f}%"
                    f"{'  (piecewise)' if info['refined'] else ''}"
                    f"  depth [{info['depth_range'][0]:.2f}, {info['depth_range'][1]:.2f}]"
                )
        else:
            finite = depth[np.isfinite(depth) & (depth > 0)]
            log_success(
                f"{name}: {how}  coverage {valid.mean()*100:.1f}%"
                f"  conf>1.5 {(conf > 1.5).mean()*100:.1f}%"
                f"  depth [{finite.min():.2f}, {finite.max():.2f}]"
            )

        depth = depth.astype(np.float32)
        for d in depth_dirs:
            np.save(os.path.join(d, f"inp_dust3r{name}_{num_views}.npy"), depth)
            np.save(os.path.join(d, f"inpv2{name}_{num_views}.npy"), depth)
        depths.append(depth)
        confs.append(conf.astype(np.float32))

    # Stacked at the depth maps' own resolution so confs[idx].reshape(-1) lines
    # up 1:1 with the per-view back-projected points in the reader.
    np.save(os.path.join(scene_path, f"depths{num_views}.npy"), np.stack(depths, axis=0))
    np.save(os.path.join(scene_path, f"confs{num_views}.npy"), np.stack(confs, axis=0))
    log_success(f"Wrote metric depth + per-view confidence for {len(depths)} view(s)")
    return True


def setup_ri3d_scene_data(scene_path, image_dir, image_files, num_views=3, resolution=4,
                          dry_run=False, depth_source="pointmap"):
    """
    Ensure all required RI3D data files are populated:
    - transforms.json
    - train_test_split_<N>.json
    - depth_rel/inp_dust3r<image_name>_<N>.npy (and inpv2)
    - depths<N>.npy & confs<N>.npy
    """
    os.makedirs(scene_path, exist_ok=True)

    # 1. Check or generate train_test_split_<N>.json
    split_file = os.path.join(scene_path, f"train_test_split_{num_views}.json")
    total_imgs = len(image_files)
    if not os.path.exists(split_file) and not dry_run:
        train_indices = [int(round(i)) for i in np.linspace(0, max(0, total_imgs - 1), num_views)]
        test_indices = [i for i in range(total_imgs) if i not in train_indices]
        if len(test_indices) == 0:
            test_indices = train_indices.copy()
        split_data = {
            "train_ids": train_indices,
            "test_ids": test_indices
        }
        with open(split_file, "w") as f:
            json.dump(split_data, f, indent=2)
        log_success(f"Generated sparse train/test split: {split_file} (train: {train_indices})")
    
    if dry_run:
        return

    # 2. Metric depth priors + per-view confidence, aligned to the SfM pointmaps.
    if generate_aligned_depth(scene_path, image_dir, num_views, depth_source=depth_source):
        return

    # 3. No pointmaps: the scene must already ship metric depth_rel from its own
    # preprocessing (this is how the mip-NeRF 360 scenes are distributed).
    # Never synthesise a stand-in here -- writing raw disparity or a constant
    # into depth_rel is exactly the mismatch this pipeline is meant to avoid,
    # and it fails silently all the way through training.
    depth_rel_dirs = [
        os.path.join(scene_path, "depth_rel"),
        os.path.join(image_dir, "depth_rel"),
    ]
    train_views = resolve_train_view_names(scene_path, num_views)
    missing = [
        name for name, _ in train_views
        if not any(
            os.path.exists(os.path.join(d, f"{prefix}{name}_{num_views}.npy"))
            for d in depth_rel_dirs
            for prefix in ("inpv2", "inp_dust3r")
        )
    ]
    if missing:
        log_error(f"No SfM pointmaps in {scene_path}/pointmaps and no depth_rel for: {', '.join(missing)}")
        log_error("Run the SfM stage (--sfm_config unposed) so depth can be aligned to the pointmaps.")
        sys.exit(1)

    for fname in (f"depths{num_views}.npy", f"confs{num_views}.npy"):
        if not os.path.exists(os.path.join(scene_path, fname)):
            log_error(f"Missing {os.path.join(scene_path, fname)} and no pointmaps available to rebuild it.")
            sys.exit(1)
    log_success("Using the scene's existing metric depth_rel priors.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RI3D: Few-Shot Gaussian Splatting Reconstruction with Repair and Inpainting Diffusion Priors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input / Output Arguments
    parser.add_argument("-i", "-s", "--input_dir", "--source_path", dest="source_path", type=str, required=True,
                        help="Path to input sparse RGB photos folder or scene directory")
    parser.add_argument("-o", "--output_dir", dest="output_dir", type=str, default=None,
                        help="Output directory root for all checkpoints, logs, and renders (default: output/<scene_name>)")
    parser.add_argument("--save_ply", type=str, default=None,
                        help="Custom path to save the final reconstructed 3DGS point cloud PLY")

    # Pipeline & SfM Settings
    parser.add_argument("--sfm_config", type=str, default="posed", choices=["posed", "unposed"],
                        help="'unposed' to run SfM pose/pointmap extraction, or 'posed' if transforms.json exists")
    parser.add_argument("--sfm_backend", type=str, default="ggpt", choices=["ggpt", "mast3r"],
                        help="Which SfM engine recovers poses and pointmaps. 'ggpt' runs VGGT + "
                             "RoMaV2 + BA + DLT triangulation; 'mast3r' runs MASt3R-SfM. Both write "
                             "the same contract, into <output>/<backend>_sfm/. Only 'mast3r' can "
                             "hold a known calibration fixed, so --sfm_config posed requires it.")
    parser.add_argument("--ggpt_root", type=str, default="/home/nguyen/projects/GGPT",
                        help="Path to the GGPT checkout (--sfm_backend ggpt)")
    parser.add_argument("--ggpt_python", type=str, default="/media/nguyen/conda/envs/ggpt/bin/python",
                        help="Interpreter used for the GGPT SfM worker. GGPT needs romav2 and RI3D "
                             "needs open3d; they live in separate environments, so the worker runs "
                             "out-of-process.")
    parser.add_argument("--fuse_mode", type=str, default="global_scale",
                        choices=["global_scale", "per_view_scale", "per_view_affine"],
                        help="How VGGT's dense depth is reconciled with the RoMaV2/DLT anchors. "
                             "'global_scale' corrects only the bundle-adjustment gauge and preserves "
                             "VGGT's multi-view consistency; 'per_view_affine' also removes each "
                             "view's own depth bias. (--sfm_backend ggpt)")
    parser.add_argument("--ggpt_refine", action="store_true",
                        help="Additionally run GGPT's PTv3 point transformer over the fused pointmap "
                             "(needs GGPT/ckpts/model.step228000.pth and the ptv3 extras).")
    parser.add_argument("--watermark_mask", type=str, default=None,
                        help="PNG mask (255 = ignore) of burned-in watermarks. Supplying one skips "
                             "the 'wm' detection stage. Masked pixels are excluded from matching, "
                             "bundle adjustment and triangulation; the photos are never modified. "
                             "Requires --sfm_backend ggpt.")
    parser.add_argument("-n", "--num_views", "--sparse_num", dest="num_views", type=int, default=3,
                        help="Number of sparse training views (e.g. 3, 6, 9)")
    parser.add_argument("-r", "--resolution", type=int, default=4,
                        help="Downsampling resolution factor for 3DGS training (1, 2, 4, 8)")
    parser.add_argument("--sh_degree", type=int, default=2,
                        help="Maximum Spherical Harmonics degree")
    parser.add_argument("--prompt", type=str, default="xxy5syt00",
                        help="Rare token / text prompt identifier for LoRA personalization")
    parser.add_argument("--depth_source", type=str, default="pointmap", choices=["pointmap", "align"],
                        help="Where the metric depth prior comes from. 'pointmap' resamples "
                             "the SfM backend's dense pointmap depth directly (metric and multi-view "
                             "consistent by construction). 'align' instead runs Depth-Anything "
                             "and fits its disparity onto the pointmap anchors, keeping monocular "
                             "detail but introducing a depth-dependent bias in the far field.")

    # Stage Controls
    parser.add_argument("--stages", type=str, default="all",
                        help="Stages to execute: comma-separated list (e.g. '1,2,3,5a,5b' or '1' or 'all')")
    parser.add_argument("--auto_orient", action="store_true", default=True,
                        help="Automatically rotate outlier photos to achieve consistent landscape/portrait orientation")
    parser.add_argument("--no_auto_orient", dest="auto_orient", action="store_false",
                        help="Disable automatic orientation standardization")
    parser.add_argument("--render_video", action="store_true", default=True,
                        help="Render a smooth 360-degree novel-view trajectory video upon completion")
    parser.add_argument("--no_render_video", dest="render_video", action="store_false",
                        help="Skip novel-view video rendering")

    # Hardware & Debug
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU device index")
    parser.add_argument("--dry_run", action="store_true", help="Print all stage execution commands without executing")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Resolve paths
    source_path = os.path.abspath(args.source_path)
    scene_name = Path(source_path).name or Path(source_path).parent.name
    if args.output_dir is None:
        args.output_dir = os.path.abspath(os.path.join("output", f"{scene_name}_{args.num_views}views"))
    else:
        args.output_dir = os.path.abspath(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{Colors.BOLD}{Colors.HEADER}=== RI3D 3D Gaussian Splatting Reconstruction Pipeline ==={Colors.ENDC}")
    print(f"  • Source Directory: {Colors.CYAN}{source_path}{Colors.ENDC}")
    print(f"  • Output Root:      {Colors.CYAN}{args.output_dir}{Colors.ENDC}")
    print(f"  • Scene Name:       {Colors.CYAN}{scene_name}{Colors.ENDC}")
    print(f"  • Sparse Views:     {Colors.CYAN}{args.num_views}{Colors.ENDC}")
    print(f"  • Resolution Scale: 1/{Colors.CYAN}{args.resolution}{Colors.ENDC}")
    print(f"  • SfM Mode:         {Colors.CYAN}{args.sfm_config}{Colors.ENDC}")
    print(f"  • SfM Backend:      {Colors.CYAN}{args.sfm_backend}{Colors.ENDC}"
          + (f" ({args.fuse_mode}{', +ptv3' if args.ggpt_refine else ''})" if args.sfm_backend == "ggpt" else ""))
    print(f"  • Stages to Run:    {Colors.CYAN}{args.stages}{Colors.ENDC}")
    print(f"  • GPU Index:        {Colors.CYAN}cuda:{args.gpu}{Colors.ENDC}\n")

    # Check checkpoints
    check_checkpoints(args)

    # Validate image orientation
    n_images, valid_source, image_dir, image_files = validate_and_standardize_images(
        source_path, auto_orient=args.auto_orient, dry_run=args.dry_run, num_views=args.num_views
    )
    if n_images == 0:
        log_error(f"No valid image files found in {source_path}")
        sys.exit(1)

    start_time = time.time()

    # Determine which stages to run.
    # 'wm' is opt-in rather than part of 'all': most scenes carry no watermark,
    # and the detector is only meaningful on a set of photos sharing one overlay.
    all_stages = ["wm", "sfm", "1a", "1b", "2a", "2b", "3", "4", "5a", "5b", "render"]
    if args.stages.strip().lower() == "all":
        stages_to_run = set(all_stages)
        stages_to_run.remove("wm")
        if args.sfm_config == "posed":
            stages_to_run.remove("sfm")
    else:
        selected = [s.strip().lower() for s in args.stages.split(",")]
        stages_to_run = set()
        for s in selected:
            if s == "1":
                stages_to_run.update(["1a", "1b"])
            elif s == "2":
                stages_to_run.update(["2a", "2b"])
            elif s == "5":
                stages_to_run.update(["5a", "5b"])
            elif s in all_stages:
                stages_to_run.add(s)
            else:
                log_warn(f"Unrecognized stage name: '{s}'")

    # GGPT's bundle adjustment always re-solves extrinsics (upstream
    # sfm/sfm_func.py carries "TODO: support known camera poses"; ba_config's
    # `calibrated` flag only substitutes known intrinsics). It therefore cannot
    # honour a fixed calibration the way configs/mast3r/posed.yaml does, and
    # running it anyway would silently discard the poses the user asked to keep.
    # MASt3R-SfM has its own matching path with no mask plumbing, so a mask there
    # would be silently ignored while the user believed watermarks were excluded.
    if (args.watermark_mask or "wm" in stages_to_run) and args.sfm_backend == "mast3r":
        log_error("Watermark masking is only supported by the ggpt backend; MASt3R-SfM has a "
                  "separate matching path with no mask plumbing.")
        sys.exit(1)

    if "sfm" in stages_to_run and args.sfm_config == "posed" and args.sfm_backend == "ggpt":
        log_error("--sfm_config posed is not supported by the ggpt backend: its bundle "
                  "adjustment always re-solves camera extrinsics.")
        log_error("Use --sfm_backend mast3r for pre-calibrated scenes, or --sfm_config unposed "
                  "to accept a fresh solve.")
        sys.exit(1)

    total_stages = len(stages_to_run)
    curr_stage_idx = 1

    # Directory constants
    debug_gs_init_dir = f"debug/gs_init/{scene_name}_{args.num_views}"
    output_gs_init_dir = f"output/gs_init/{scene_name}_{args.num_views}"
    loo_dir = f"output/gs_init/{scene_name}_loo_{args.num_views}"
    lora_exp_dir = f"controlnet_finetune/{scene_name}_{args.num_views}"
    output_lora_dir = f"output/{lora_exp_dir}"
    output_den_dir = f"output_den{args.num_views}"
    output_inp_dir = f"output_inp{args.num_views}"
    final_stage1_ply = f"{output_den_dir}/gaussian_object/{scene_name}_{args.num_views}/save/last.ply"
    final_stage2_ply = f"{output_inp_dir}/gaussian_object/{scene_name}_{args.num_views}/save/last.ply"

    # =========================================================================
    # Stage 0: SfM Pose & Pointmap Extraction (if unposed)
    # =========================================================================
    sfm_label = "GGPT-SfM (VGGT + RoMaV2)" if args.sfm_backend == "ggpt" else "MASt3R-SfM"
    effective_source_path = source_path

    # =========================================================================
    # Stage wm: watermark detection (opt-in, before SfM)
    # =========================================================================
    # Burned-in watermarks sit at identical pixel coordinates in every photo, so
    # they match themselves at zero disparity and -- because run_sfm ranks BA
    # tracks by SuperPoint saliency -- are preferentially promoted into the
    # bundle adjustment. Detect them once here; the SfM stage excludes them.
    if "wm" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Watermark Detection")
        detected = os.path.join(args.output_dir, "watermark_mask.png")
        cmd = (f"python tools/detect_watermark.py -i {source_path} "
               f"--output_dir {args.output_dir}")
        run_command(cmd, dry_run=args.dry_run)
        if args.dry_run:
            pass
        elif os.path.exists(detected):
            args.watermark_mask = detected
            log_success(f"Watermark mask: {detected} (preview alongside it)")
        else:
            log_success("No watermark detected; SfM will use every pixel.")
        curr_stage_idx += 1

    if "sfm" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, f"{sfm_label} Camera Pose & Pointmap Extraction")
        sfm_out_dir = run_sfm_and_extract_pointmap(source_path, args.output_dir, args, dry_run=args.dry_run)
        effective_source_path = sfm_out_dir
        # Re-scan images from the SfM output images directory: those are the
        # frames the exported intrinsics refer to (MASt3R centre-crops, GGPT
        # transposes portrait input), not the originals.
        _, _, image_dir, image_files = validate_and_standardize_images(
            effective_source_path, auto_orient=False, dry_run=args.dry_run, num_views=args.num_views
        )
        curr_stage_idx += 1
    elif args.sfm_config == "unposed":
        # Reuse a previous solve. Prefer the selected backend's directory, but
        # fall back to the other one so an existing scene stays usable after the
        # default flipped to ggpt.
        candidates = [os.path.join(args.output_dir, f"{args.sfm_backend}_sfm")]
        candidates += [os.path.join(args.output_dir, f"{b}_sfm")
                       for b in ("ggpt", "mast3r") if b != args.sfm_backend]
        sfm_dir = next((d for d in candidates if os.path.isdir(d)), None)
        if sfm_dir is not None:
            if not sfm_dir.endswith(f"{args.sfm_backend}_sfm"):
                log_warn(f"No {args.sfm_backend}_sfm/ solve found; reusing {os.path.basename(sfm_dir)}/ instead.")
            effective_source_path = sfm_dir
            _, _, image_dir, image_files = validate_and_standardize_images(
                effective_source_path, auto_orient=False, dry_run=args.dry_run, num_views=args.num_views
            )
        else:
            log_error(f"Unposed scene '{source_path}' requires camera poses and pointmaps from SfM, "
                      f"but none of {[os.path.basename(c) for c in candidates]} was found under {args.output_dir}.")
            log_error(f"Please include 'sfm' in --stages (e.g. '--stages sfm,1' or '--stages all') to run {sfm_label} first.")
            sys.exit(1)

    # Prepare scene dataset structures & depth priors
    setup_ri3d_scene_data(effective_source_path, image_dir, image_files, num_views=args.num_views,
                          resolution=args.resolution, dry_run=args.dry_run,
                          depth_source=args.depth_source)

    # =========================================================================
    # Stage 1a: Gaussian Point Cloud Initialization
    # =========================================================================
    if "1a" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Stage 1a: Gaussian Point Cloud Initialization")
        cmd = (
            f"python scripts/train_gs_init.py -s {effective_source_path} "
            f"-m {debug_gs_init_dir} "
            f"-r {args.resolution} --sparse_view_num {args.num_views} --sh_degree {args.sh_degree} "
            f"--white_background --random_background"
        )
        run_command(cmd, dry_run=args.dry_run)
        log_success(f"Point cloud initialized at {debug_gs_init_dir}/point_cloud/iteration_1/point_cloud.ply")
        curr_stage_idx += 1

    # =========================================================================
    # Stage 1b: Base 3D Gaussian Splatting Training
    # =========================================================================
    if "1b" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Stage 1b: Base 3DGS Optimization on Sparse Views")
        init_ply = f"{debug_gs_init_dir}/point_cloud/iteration_1/point_cloud.ply"
        cmd = (
            f"python -W ignore scripts/train_gs.py -s {effective_source_path} "
            f"-m {output_gs_init_dir} "
            f"-r {args.resolution} --sparse_view_num {args.num_views} --sh_degree {args.sh_degree} "
            f"--white_background --random_background "
            f"--ply_path {init_ply}"
        )
        run_command(cmd, dry_run=args.dry_run)
        log_success(f"Base 3DGS model saved to {output_gs_init_dir}")
        curr_stage_idx += 1

    # =========================================================================
    # Stage 2: Leave-One-Out (LOO) Data Generation
    # =========================================================================
    if "2a" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Stage 2a: Leave-One-Out Data Generation (Pass 1)")
        init_ply = f"{debug_gs_init_dir}/point_cloud/iteration_1/point_cloud.ply"
        cmd = (
            f"python -W ignore scripts/leave_one_out_stage1.py -s {effective_source_path} "
            f"-m {loo_dir} "
            f"-r {args.resolution} --sparse_view_num {args.num_views} --sh_degree {args.sh_degree} "
            f"--white_background --random_background "
            f"--ply_path {init_ply}"
        )
        run_command(cmd, dry_run=args.dry_run)
        curr_stage_idx += 1

    if "2b" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Stage 2b: Leave-One-Out Data Generation (Pass 2 & Diff Stats)")
        init_ply = f"{debug_gs_init_dir}/point_cloud/iteration_1/point_cloud.ply"
        cmd = (
            f"python -W ignore scripts/leave_one_out_stage2.py -s {effective_source_path} "
            f"-m {loo_dir} "
            f"-r {args.resolution} --sparse_view_num {args.num_views} --sh_degree {args.sh_degree} "
            f"--white_background --random_background "
            f"--ply_path {init_ply}"
        )
        run_command(cmd, dry_run=args.dry_run)
        log_success(f"Leave-one-out data and parameter noise distributions saved to {loo_dir}")
        curr_stage_idx += 1

    # =========================================================================
    # Stage 3: ControlNet LoRA Fine-Tuning (Repair Model)
    # =========================================================================
    if "3" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Stage 3: LoRA Fine-Tuning for ControlNet Repair Prior")
        cmd = (
            f"python scripts/train_lora.py --exp_name {lora_exp_dir} "
            f"--prompt {args.prompt} --sh_degree {args.sh_degree} --resolution {args.resolution} "
            f"--sparse_num {args.num_views} "
            f"--data_dir {effective_source_path} "
            f"--gs_dir {output_gs_init_dir} "
            f"--loo_dir {loo_dir} "
            f"--bg_white --sd_locked --train_lora "
            f"--add_diffusion_lora --add_control_lora --add_clip_lora"
        )
        run_command(cmd, dry_run=args.dry_run)
        log_success(f"Trained ControlNet LoRA checkpoint saved to {output_lora_dir}")
        curr_stage_idx += 1

    # =========================================================================
    # Stage 4: Inpainting Prior Setup / Fine-Tuning
    # =========================================================================
    if "4" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Stage 4: SD2 Inpainting Prior Setup (RealFill)")
        inpainting_target_dir = f"inpainting/{scene_name}_{args.num_views}"
        if os.path.exists("train_realfill.py"):
            cmd = (
                f"python train_realfill.py "
                f"--pretrained_model_name_or_path=models/stable-diffusion-2-inpainting "
                f"--train_data_dir={effective_source_path}/{args.num_views} "
                f"--output_dir={inpainting_target_dir} "
                f"--resolution=512 --train_batch_size=16 --max_train_steps=2000"
            )
            run_command(cmd, dry_run=args.dry_run)
        else:
            log_warn("External train_realfill.py not found; using base SD2 Inpainting checkpoint.")
            os.makedirs(inpainting_target_dir, exist_ok=True)
        curr_stage_idx += 1

    # =========================================================================
    # Stage 5a: Dual-Prior Repair (Densification) Optimization
    # =========================================================================
    if "5a" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Stage 5a: Diffusion Repair Densification Optimization")
        os.makedirs(f"{output_den_dir}/gaussian_object/{scene_name}_{args.num_views}", exist_ok=True)
        cmd = (
            f"python -W ignore scripts/train_repair.py "
            f"--config configs/gaussian-object.yaml "
            f"--train --gpu 0 "
            f"tag=\"{scene_name}_{args.num_views}\" "
            f"exp_root_dir=\"{output_den_dir}\" "
            f"system.init_dreamer=\"{debug_gs_init_dir}\" "
            f"system.exp_name=\"{output_lora_dir}\" "
            f"system.refresh_size=8 "
            f"data.data_dir=\"{effective_source_path}\" "
            f"data.resolution={args.resolution} "
            f"data.sparse_num={args.num_views} "
            f"data.prompt=\"a photo of a {args.prompt}\" "
            f"data.refresh_size=8 "
            f"system.sh_degree={args.sh_degree}"
        )
        run_command(cmd, dry_run=args.dry_run)
        log_success(f"Stage 5a (Repair) completed. Saved to {output_den_dir}")
        curr_stage_idx += 1

    # =========================================================================
    # Stage 5b: Dual-Prior Inpainting Refinement Optimization
    # =========================================================================
    if "5b" in stages_to_run:
        log_banner(curr_stage_idx, total_stages, "Stage 5b: Diffusion Inpainting Refinement Optimization")
        os.makedirs(f"{output_inp_dir}/gaussian_object/{scene_name}_{args.num_views}", exist_ok=True)
        cmd = (
            f"python -W ignore scripts/train_repair.py "
            f"--config configs/gaussian-object_inp.yaml "
            f"--train --gpu 0 "
            f"tag=\"{scene_name}_{args.num_views}\" "
            f"exp_root_dir=\"{output_inp_dir}\" "
            f"system.init_dreamer=\"{output_den_dir}/gaussian_object/{scene_name}_{args.num_views}\" "
            f"system.exp_name=\"{output_lora_dir}\" "
            f"system.inpainting_dir=\"inpainting\" "
            f"system.refresh_size=10 "
            f"data.data_dir=\"{effective_source_path}\" "
            f"data.resolution={args.resolution} "
            f"data.sparse_num={args.num_views} "
            f"data.prompt=\"a photo of a {args.prompt}\" "
            f"data.refresh_size=10 "
            f"system.sh_degree={args.sh_degree}"
        )
        run_command(cmd, dry_run=args.dry_run)
        log_success(f"Stage 5b (Inpainting Refinement) completed. Saved to {output_inp_dir}")
        curr_stage_idx += 1

    # =========================================================================
    # Rendering & Final Model Export
    # =========================================================================
    def _latest_gs_ply(model_dir):
        """Highest-iteration checkpoint under <model_dir>/point_cloud/iteration_*.

        The iteration count is not fixed: OptimizationParams.iterations is 10_000
        here (upstream 3DGS defaults to 30_000) and train_gs.py saves at whatever
        --iterations resolves to, so the directory name cannot be hardcoded.
        """
        pc_dir = os.path.join(model_dir, "point_cloud")
        if not os.path.isdir(pc_dir):
            return None
        iters = []
        for d in os.listdir(pc_dir):
            ply = os.path.join(pc_dir, d, "point_cloud.ply")
            if d.startswith("iteration_") and os.path.isfile(ply):
                try:
                    iters.append((int(d.split("_")[-1]), ply))
                except ValueError:
                    continue
        return max(iters)[1] if iters else None

    best_ply = final_stage2_ply if os.path.exists(final_stage2_ply) else (
        final_stage1_ply if os.path.exists(final_stage1_ply) else _latest_gs_ply(output_gs_init_dir)
    )

    if args.render_video and ("render" in stages_to_run or "5b" in stages_to_run or "5a" in stages_to_run or "1b" in stages_to_run):
        log_banner(curr_stage_idx, total_stages, "Rendering Novel-View Orbit Video & Metrics")
        if best_ply is not None or args.dry_run:
            ply_to_render = best_ply if best_ply else final_stage2_ply
            cmd = (
                f"python scripts/render.py "
                f"-m {output_gs_init_dir} "
                f"--sparse_view_num {args.num_views} --sh_degree {args.sh_degree} "
                f"--white_background --render_path "
                f"--postfix _stage2 "
                f"--load_ply {ply_to_render}"
            )
            run_command(cmd, dry_run=args.dry_run)
            log_success("Novel-view video rendered.")
        else:
            log_warn("No trained PLY model found to render video.")

    # Copy PLY to destination
    if best_ply and os.path.exists(best_ply) and not args.dry_run:
        dest_ply = args.save_ply or os.path.join(args.output_dir, f"{scene_name}_reconstructed_3dgs.ply")
        shutil.copyfile(best_ply, dest_ply)
        log_success(f"Exported final 3DGS scene to: {dest_ply}")

    total_time = time.time() - start_time
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*75}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}   🎉 RI3D Reconstruction Finished Successfully! ({total_time:.1f}s){Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*75}{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
