"""
Utilities for turning MASt3R-SfM dense pointmaps into per-view metric depth.

MASt3R writes one JSON per view under ``<sfm_dir>/pointmaps/<image_name>.json``
holding world-frame XYZ (flattened, row-major over the pointmap grid) plus a
per-pixel confidence map. The grid is the *resized* image MASt3R ran on
(e.g. 384x512), which is generally smaller than -- and possibly cropped
relative to -- the images RI3D trains on.

Rather than assuming the pointmap grid aligns with the target image grid, we
project the 3D points into the target camera with its own intrinsics and
z-buffer them. That is exact regardless of MASt3R's internal resize/crop, at
the cost of producing a *sparse* depth map (~20% coverage when the target is
4x the pointmap resolution). That is enough: these pixels serve as metric
anchors for the monocular depth fit in `utils.depth_align`, which supplies the
dense structure.
"""

import json
import os.path as osp

import numpy as np


def load_pointmap(pointmaps_dir, image_name, conf_thr=0.0):
    """Load one MASt3R pointmap.

    Returns (points_world, confs) with shapes (H_pm*W_pm, 3) and (H_pm, W_pm).
    `points` is stored flat in the JSON; `confs` carries the grid shape.
    """
    path = osp.join(pointmaps_dir, f"{image_name}.json")
    if not osp.exists(path):
        raise FileNotFoundError(f"MASt3R pointmap not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    confs = np.asarray(data["confs"], dtype=np.float32)
    points = np.asarray(data["points"], dtype=np.float64)

    if points.shape[0] != confs.size:
        raise ValueError(
            f"{path}: {points.shape[0]} points do not match conf grid {confs.shape}"
        )
    if conf_thr > 0:
        points = points[confs.reshape(-1) > conf_thr]
        # keep confs full-size; the caller masks with the same threshold
    return points, confs


def load_sfm_poses(sfm_dir):
    """Read camera-to-world matrices from ``<sfm_dir>/cameras.json``.

    This is the array MASt3R's COLMAP export is derived from, so using it
    avoids a rotation -> quaternion -> rotation round-trip. Convention is
    OpenCV (x right, y down, z forward), matching what `run_mast3r.py` writes
    into ``sparse/0``.

    Returns {image_name (no extension): (4, 4) c2w}.
    """
    path = osp.join(sfm_dir, "cameras.json")
    if not osp.exists(path):
        raise FileNotFoundError(f"MASt3R cameras.json not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    c2ws = np.asarray(data["cams2world"], dtype=np.float64)
    names = [osp.splitext(osp.basename(p))[0] for p in data["filepaths"]]
    return {name: c2ws[i] for i, name in enumerate(names)}


def load_target_intrinsics(scene_dir):
    """Read the target image grid and intrinsics from ``transforms.json``.

    The depth maps RI3D consumes must match ``transforms.json``'s w/h exactly:
    `readMipTransforms` scales cam_info.width by `resolution` and the K build in
    `dataset_readers_flow.py` divides it back out, so the two cancel only at
    this resolution. Never key off the image file's own size -- for mip-NeRF 360
    `transforms.json` stores the /4 dimensions, while `inference.py`'s COLMAP
    converter stores full-resolution ones.

    Returns (K, width, height).
    """
    path = osp.join(scene_dir, "transforms.json")
    if not osp.exists(path):
        raise FileNotFoundError(f"transforms.json not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    width, height = int(data["w"]), int(data["h"])
    K = np.array(
        [
            [data["fl_x"], 0.0, data["cx"]],
            [0.0, data["fl_y"], data["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K, width, height


def project_pointmap(points_world, confs, c2w, K, width, height, conf_thr=0.1):
    """Z-buffer a world-frame pointmap into a target view.

    Returns (depth, conf, valid), each (height, width). `depth` is metric
    z-depth in MASt3R world units at covered pixels and 0 elsewhere; `valid`
    marks the covered pixels.
    """
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ points_world.T + w2c[:3, 3:4]).T
    z = cam[:, 2]

    keep = z > 1e-6
    safe_z = np.where(keep, z, 1.0)[:, None]
    uv = (K[:2, :2] @ (cam[:, :2] / safe_z).T).T + K[:2, 2]
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)

    conf_flat = confs.reshape(-1)
    keep &= (u >= 0) & (u < width) & (v >= 0) & (v < height) & (conf_flat > conf_thr)

    depth = np.zeros((height, width), dtype=np.float32)
    conf = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)

    # Sort far-to-near so the nearest surface wins each pixel.
    order = np.argsort(-z[keep])
    vi, ui = v[keep][order], u[keep][order]
    depth[vi, ui] = z[keep][order]
    conf[vi, ui] = conf_flat[keep][order]
    valid[vi, ui] = True

    return depth, conf, valid


def render_view_depth(sfm_dir, image_name, K, width, height, c2w=None, conf_thr=0.1):
    """Convenience wrapper: pointmap JSON -> sparse metric depth for one view."""
    if c2w is None:
        c2w = load_sfm_poses(sfm_dir)[image_name]
    points, confs = load_pointmap(osp.join(sfm_dir, "pointmaps"), image_name)
    return project_pointmap(points, confs, c2w, K, width, height, conf_thr=conf_thr)


def pointmap_grid_is_aligned(points_world, confs, c2w, K, width, height, tol=1.5):
    """Check the pointmap grid maps linearly onto the target image grid.

    MASt3R runs on a resized (and possibly centre-cropped) copy of the image.
    When no crop was applied, pointmap pixel (i, j) corresponds to target pixel
    ((i + .5) * width / W_pm - .5, ...), and the depth image can simply be
    resampled. When a crop *was* applied that mapping breaks and the caller must
    fall back to projecting points individually.
    """
    Hp, Wp = confs.shape
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ points_world.T + w2c[:3, 3:4]).T
    z = cam[:, 2]
    ok = z > 1e-6
    if ok.sum() < 100:
        return False

    uv = (K[:2, :2] @ (cam[:, :2] / np.where(ok, z, 1.0)[:, None]).T).T + K[:2, 2]
    jj, ii = np.mgrid[:Hp, :Wp]
    exp_u = (ii.reshape(-1) + 0.5) * (width / Wp) - 0.5
    exp_v = (jj.reshape(-1) + 0.5) * (height / Hp) - 0.5
    du = np.percentile(np.abs(uv[ok, 0] - exp_u[ok]), 99)
    dv = np.percentile(np.abs(uv[ok, 1] - exp_v[ok]), 99)
    return bool(du < tol and dv < tol)


def resample_view_depth(sfm_dir, image_name, K, width, height, c2w=None):
    """Dense metric depth for one view by resampling the pointmap grid.

    The pointmap is a dense depth image on its own grid, so upsampling it to the
    target resolution keeps every value MASt3R produced. Projecting the points
    individually instead (see `project_pointmap`) scatters them into a finer
    grid and leaves most target pixels empty -- at 2x upsampling that discards
    ~80% of the coverage and hands those pixels to the monocular prior.

    Nearest-neighbour is deliberate: this only ever upsamples, so it loses
    nothing, and it will not smear depth across occlusion boundaries the way
    bilinear does.

    Returns (depth, conf, valid), or None if the grid is not linearly aligned.
    """
    import cv2

    if c2w is None:
        c2w = load_sfm_poses(sfm_dir)[image_name]
    points, confs = load_pointmap(osp.join(sfm_dir, "pointmaps"), image_name)
    if not pointmap_grid_is_aligned(points, confs, c2w, K, width, height):
        return None

    Hp, Wp = confs.shape
    w2c = np.linalg.inv(c2w)
    cam = (w2c[:3, :3] @ points.T + w2c[:3, 3:4]).T
    z = cam[:, 2].reshape(Hp, Wp).astype(np.float32)

    depth = cv2.resize(z, (width, height), interpolation=cv2.INTER_NEAREST)
    conf = cv2.resize(confs.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return depth, conf, depth > 1e-6


def fill_holes_nearest(depth, valid):
    """Nearest-neighbour fill of uncovered pixels.

    Only used for the fallback path where a monocular fit was rejected and the
    sparse MASt3R depth has to stand on its own.
    """
    from scipy.ndimage import distance_transform_edt

    if valid.all():
        return depth.copy()
    if not valid.any():
        raise ValueError("cannot fill a depth map with no valid pixels")

    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    return depth[iy, ix].astype(np.float32)
