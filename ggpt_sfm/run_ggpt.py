"""GGPT-SfM front-end for RI3D -- a drop-in replacement for ``mast3r/run_mast3r.py``.

Geometry comes from the GGPT stack instead of MASt3R-SfM:

    VGGT           dense per-view pointmap + initial poses/intrinsics
    RoMaV2         dense two-view matching over every image pair
    pycolmap BA    refines poses and focal on the selected tracks
    DLT            re-triangulates the surviving matches in the BA frame
    GGPT (opt.)    PTv3 point transformer refines the dense pointmap

and the result is written in exactly the layout the rest of RI3D reads:

    <output_dir>/images/<name>.<ext>     the images the intrinsics refer to
    <output_dir>/cameras.json            filepaths / focals / cams2world (OpenCV c2w)
    <output_dir>/sparse/0/*.bin|.txt     COLMAP model at full image resolution
    <output_dir>/pointmaps/<name>.json   dense world XYZ (flat, row-major) + confs (2-D)
    <output_dir>/points.ply              confidence-filtered fused cloud

``inference.py`` then converts ``sparse/0`` to ``transforms.json`` and
``utils/pointmap_utils.py`` resamples the pointmaps into the metric depth prior,
both unchanged.

Two things make this contract non-trivial and are handled explicitly below.

**Gauge.** VGGT's pointmap lives in VGGT's own world; bundle adjustment produces
a second world that is close to it but free to drift in scale. Rather than
aligning the two clouds, we work in per-view depth -- which is what the pointmap
grid actually stores -- and rescale VGGT's depth onto the DLT depth measured at
the same pixels. Nothing is ever unprojected with mismatched poses.

**Grid alignment.** ``utils/pointmap_utils.pointmap_grid_is_aligned`` requires
pointmap pixel (i, j) to land on full-image pixel ((i+.5)*W/Wp-.5, ...) to within
1.5 px, or RI3D silently falls back to the sparse projection path. GGPT resizes
to a width of 518 and rounds the height to a multiple of 14, so the ff grid is a
*mildly anisotropic* stretch of the full image (1.3214 vs 1.3333 on a 4:3 photo).
Scaling fx and fy by W/Wp and H/Hp independently, with the principal point at
(W/2-0.5, H/2-0.5), makes that mapping exact rather than approximate. The check
is re-run at the end of this script and reported.
"""

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np

VALID_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

RI3D_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------


def bootstrap(ggpt_root):
    """Put the GGPT repo on sys.path and make it the cwd.

    GGPT resolves `vggt` and `ckpts/` relative to the working directory
    (`feedforward/__init__.py` literally does ``sys.path.append('vggt')``), so it
    has to be entered rather than merely imported.

    GGPT and RI3D both ship a top-level ``utils`` package, and GGPT's must win.
    Ordering sys.path is *not* sufficient: GGPT's ``utils/`` has no
    ``__init__.py`` so it is only a namespace portion, while RI3D's is a regular
    package -- and a regular package found anywhere on the path beats a
    namespace portion found earlier. `inference.py:run_command` puts the RI3D
    root on PYTHONPATH, which is enough to break every ``from utils.x import y``
    inside GGPT. So the RI3D root is removed from sys.path outright.

    The only RI3D module this script needs is ``colmap.read_write_model``, which
    lives under ``mast3r/`` and is a standalone COLMAP I/O file with no MASt3R
    dependencies; `verify_grid_alignment` loads `utils/pointmap_utils.py` by
    file path for the same reason.
    """
    ggpt_root = osp.abspath(ggpt_root)
    if not osp.isdir(ggpt_root):
        sys.exit(f"[ERROR] GGPT checkout not found: {ggpt_root}")

    sys.path[:] = [p for p in sys.path if p and osp.abspath(p) != RI3D_ROOT]
    sys.path.insert(0, ggpt_root)
    sys.path.insert(1, osp.join(ggpt_root, "vggt"))
    sys.path.append(osp.join(RI3D_ROOT, "mast3r"))
    os.chdir(ggpt_root)


# --------------------------------------------------------------------------
# image selection / preprocessing
# --------------------------------------------------------------------------


def resolve_image_dir(scene_path):
    """Mirror the directory probing `inference.py` performs on the input."""
    for sub in ("images", "images_4", "images_2", "images_8"):
        cand = osp.join(scene_path, sub)
        if osp.isdir(cand):
            return cand
    return scene_path


def select_images(image_dir, n_images, image_idx, randomize, use_all, seed):
    """Pick the subset of views to reconstruct, matching `scripts/run_sfm.py`."""
    names = sorted(f for f in os.listdir(image_dir) if f.endswith(VALID_EXTS))
    if not names:
        sys.exit(f"[ERROR] No images found in {image_dir}")

    if use_all or n_images is None or n_images < 0 or n_images >= len(names):
        chosen = names
    elif image_idx:
        chosen = [names[i] for i in image_idx]
    else:
        order = list(range(len(names)))
        if randomize:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        picked = [order[int(round(i))] for i in np.linspace(0, len(names) - 1, n_images)]
        chosen = [names[i] for i in sorted(picked)]

    return [osp.join(image_dir, n) for n in chosen]


def load_images_ggpt(filelist, max_width):
    """Reproduce GGPT's `run_demo.py` image preparation.

    Returns (match_images, full_images, transposed) where `match_images` is the
    (N, H, W, 3) float tensor RoMaV2 matches on and `full_images` are the
    original-resolution uint8 arrays *after* the same portrait handling, which
    is what `<output_dir>/images/` must contain for the intrinsics to be valid.
    """
    import cv2
    import torch

    full = []
    h = w = None
    for i, path in enumerate(filelist):
        img = cv2.imread(path)[:, :, ::-1]
        if i == 0:
            h, w = img.shape[:2]
        elif (h, w) != img.shape[:2]:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        full.append(np.ascontiguousarray(img))

    # GGPT turns portrait input landscape by *transposing* (not rotating) the
    # array. Geometry is solved consistently either way, but the images RI3D
    # trains on must be transposed the same way or every intrinsic is a
    # transpose of the truth.
    transposed = h > w
    if transposed:
        full = [np.ascontiguousarray(img.transpose(1, 0, 2)) for img in full]
        h, w = w, h

    images = torch.from_numpy(np.stack(full, axis=0)).float() / 255.0

    H, W = h, w
    if W > max_width:
        H = int(round(H * max_width / W))
        W = max_width
    newH, newW = int(round(H / 14) * 14), int(round(W / 14) * 14)
    images = torch.nn.functional.interpolate(
        images.permute(0, 3, 1, 2), size=(newH, newW), mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)

    return images, full, transposed


# --------------------------------------------------------------------------
# fusion: VGGT dense depth x DLT sparse depth
# --------------------------------------------------------------------------


def load_watermark_mask(path, full_shape, transposed, lr_h, lr_w):
    """Load a watermark mask and put it on the matcher's low-res grid.

    The mask is authored against the *original* photo, so it has to follow those
    photos through exactly the preprocessing `load_images_ggpt` applied -- GGPT
    transposes portrait input to landscape, and a mask that misses that lands on
    the wrong pixels *silently*, since a misplaced mask still yields a
    plausible-looking solve.

    Nearest-neighbour throughout: a bilinear resize would produce fractional mask
    values whose threshold is arbitrary, and would bleed the mask outward by half
    a source pixel at every boundary.

    Returns (full_res_mask, lr_mask), both bool with True = ignore this pixel. The
    full-resolution one is what the 3DGS training losses need; the low-res one is
    what the matcher is indexed on.
    """
    import cv2

    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        sys.exit(f"[ERROR] Could not read watermark mask: {path}")

    H, W = full_shape
    if transposed:
        # `load_images_ggpt` transposed the images; the mask must match.
        m = np.ascontiguousarray(m.T)
    if m.shape != (H, W):
        sys.exit(
            f"[ERROR] Watermark mask is {m.shape[1]}x{m.shape[0]} but the images are {W}x{H}"
            + (" (after the portrait->landscape transpose)" if transposed else "")
            + ". A mask drawn for a different photo set would silently mask the wrong pixels; "
              "re-run tools/detect_watermark.py on these images."
        )

    full = m > 127
    lr = cv2.resize(m, (lr_w, lr_h), interpolation=cv2.INTER_NEAREST) > 127
    return full, lr


def install_match_mask(mask_lr):
    """Make GGPT's matcher ignore watermark pixels, without editing GGPT.

    A watermark sits at identical image coordinates in every view, so it matches
    itself at zero disparity -- geometry that contradicts real camera motion. It
    is not merely noise either: `run_sfm` ranks BA track candidates by SuperPoint
    saliency, and a crisp watermark edge scores *higher* than real scene texture,
    so those pixels are preferentially promoted into the bundle adjustment.

    `sfm/sfm_func.py` binds the matcher at module scope (`from matching import
    match_images`), so the name to replace lives in `sfm_func`, not `matching`.
    The wrapper edits the returned scores only; every threshold, filter and
    reshape downstream is GGPT's own, untouched.

    Tracks are keyed `(source_view, pixel)` flattened source-major to Nsrc*H*W --
    which is what lets one (H*W,) mask broadcast across both tensors below.
    """
    import torch

    import sfm.sfm_func as sfm_func

    original = sfm_func.match_images
    flat = torch.from_numpy(mask_lr.reshape(-1))
    lr_h, lr_w = mask_lr.shape

    def masked_match_images(*a, **kw):
        r = original(*a, **kw)
        m = flat.to(r["pred_scores"].device)

        # Source side: retire every track that *originates* on the watermark.
        # A track needs >=2 visible views, so zeroing its score in every target
        # drops it from both M_ba and M_dlt outright.
        r["sp_scores"][:, m] = 0.0
        r["pred_scores"][:, :, m] = 0.0

        # Target side: a match that *lands* on a watermark pixel is unreliable
        # even when it starts from real scene content -- the watermark occludes
        # whatever it was supposed to correspond to.
        xy = r["pred_matches_lr"].round().long()
        u = xy[..., 0].clamp_(0, lr_w - 1)
        v = xy[..., 1].clamp_(0, lr_h - 1)
        lands_on_wm = m.view(lr_h, lr_w)[v, u]
        r["pred_scores"][lands_on_wm] = 0.0

        n_src = int(m.sum())
        print(f"[INFO] watermark mask: retired {n_src} of {m.numel()} source pixels "
              f"({n_src / m.numel() * 100:.2f}%), plus "
              f"{int(lands_on_wm.sum())} matches landing on masked pixels")
        return r

    sfm_func.match_images = masked_match_images


def _depth_from_world(points, extrinsics):
    """Camera-frame z of world points under (N, 3, 4) w2c extrinsics."""
    R, t = extrinsics[:, :3, :3], extrinsics[:, :3, 3]
    return np.einsum("nj,nhwj->nhw", R[:, 2, :], points) + t[:, 2, None, None]


def ba_diagnostics(dlt_points, dlt_mask, ba_extr, ba_intr):
    """Reprojection error of the triangulated anchors in the views that saw them.

    GGPT's `run_sfm` builds the bundle adjuster locally and drops it, so Ceres's
    own convergence report reaches the console but not the caller -- and on
    sparse captures it routinely prints "No convergence" after hitting the
    iteration cap while still lowering the cost, which makes it a poor pass/fail
    signal anyway. This measures the thing that actually matters instead: with
    the final poses and intrinsics, does a DLT point land back on the pixel it
    was triangulated from?
    """
    N, H, W, _ = dlt_points.shape
    jj, ii = np.mgrid[:H, :W].astype(np.float64)
    per_view, pooled = [], []

    for n in range(N):
        m = dlt_mask[n]
        if not m.any():
            per_view.append({"view": n, "n": 0})
            continue
        R, t = ba_extr[n, :3, :3], ba_extr[n, :3, 3]
        cam = dlt_points[n][m] @ R.T + t
        z = cam[:, 2]
        ok = z > 1e-6
        fx, fy = ba_intr[n, 0, 0], ba_intr[n, 1, 1]
        cx, cy = ba_intr[n, 0, 2], ba_intr[n, 1, 2]
        u = fx * cam[ok, 0] / z[ok] + cx
        v = fy * cam[ok, 1] / z[ok] + cy
        err = np.hypot(u - ii[m][ok], v - jj[m][ok])
        per_view.append(
            {
                "view": n,
                "n": int(err.size),
                "behind_camera": int((~ok).sum()),
                "p50_px": float(np.median(err)),
                "p90_px": float(np.percentile(err, 90)),
            }
        )
        pooled.append(err)

    err = np.concatenate(pooled) if pooled else np.zeros(0)
    return {
        "reproj_p50_px": float(np.median(err)) if err.size else None,
        "reproj_p90_px": float(np.percentile(err, 90)) if err.size else None,
        "per_view": per_view,
    }


def _robust_scale(z_ref, z_src):
    """Median ratio, the scale that maps `z_src` onto `z_ref`."""
    ok = np.isfinite(z_ref) & np.isfinite(z_src) & (z_ref > 1e-6) & (z_src > 1e-6)
    if ok.sum() < 32:
        return None, 0
    return float(np.median(z_ref[ok] / z_src[ok])), int(ok.sum())


def _affine_in_disparity(z_dense, z_anchor, mask):
    """Robust IRLS-Huber affine fit of 1/z_dense onto 1/z_anchor.

    Same idea as `utils/depth_align.py`: a per-view scale alone cannot absorb a
    depth-dependent bias, and fitting in disparity space keeps the far field from
    dominating the residual. Returns the corrected dense depth, or None if the
    fit does not beat a plain scale.
    """
    ok = mask & (z_dense > 1e-6) & (z_anchor > 1e-6)
    if ok.sum() < 128:
        return None, None

    x = 1.0 / z_dense[ok]
    y = 1.0 / z_anchor[ok]
    wts = np.ones_like(x)
    a, b = 1.0, 0.0
    for _ in range(12):
        A = np.stack([x, np.ones_like(x)], axis=1) * wts[:, None]
        sol, *_ = np.linalg.lstsq(A, y * wts, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        r = np.abs(a * x + b - y)
        s = 1.4826 * np.median(r) + 1e-12
        wts = np.clip(s * 1.345 / np.maximum(r, 1e-12), None, 1.0)

    disp = a / np.maximum(z_dense, 1e-6) + b
    if not np.all(np.isfinite(disp)) or np.median(disp) <= 0:
        return None, None

    scale, _ = _robust_scale(z_anchor[ok], z_dense[ok])
    resid_affine = float(np.median(np.abs(a * x + b - y) / y))
    resid_scale = float(np.median(np.abs(1.0 / (scale * z_dense[ok]) - y) / y)) if scale else np.inf
    if resid_affine >= resid_scale:
        return None, {"rejected": True, "affine": resid_affine, "scale": resid_scale}

    return 1.0 / np.maximum(disp, 1e-8), {
        "a": a,
        "b": b,
        "affine": resid_affine,
        "scale": resid_scale,
    }


def fuse_pointmaps(ff_points, ff_extr, dlt_points, dlt_mask, ba_extr, ba_intr, mode):
    """Bring VGGT's dense pointmap into the bundle-adjusted frame.

    VGGT's pointmap is dense but lives in VGGT's gauge; the DLT points are exact
    in the BA gauge but cover only the pixels that survived matching, BA and the
    epipolar/reprojection/parallax filters. Both are indexed by the same ff grid,
    so the two can be reconciled as depth maps without ever aligning the clouds.

    Returns (world_points (N,H,W,3), depth (N,H,W), stats).
    """
    N, H, W, _ = ff_points.shape

    z_ff = _depth_from_world(ff_points, ff_extr)
    z_dlt = _depth_from_world(dlt_points, ba_extr)
    z_dlt = np.where(dlt_mask, z_dlt, 0.0)

    stats = {"mode": mode, "per_view": []}

    # One global scale first: BA is a local refinement of VGGT's poses, so the
    # dominant discrepancy is the gauge scale, and correcting it globally leaves
    # VGGT's multi-view consistency intact.
    gscale, n_used = _robust_scale(z_dlt[dlt_mask], z_ff[dlt_mask])
    if gscale is None:
        raise SystemExit(
            "[ERROR] Fewer than 32 pixels carry both a VGGT depth and a DLT depth. "
            "The matching stage effectively failed -- loosen dlt_config thresholds "
            "or check that the views actually overlap."
        )
    stats["global_scale"] = gscale
    stats["anchor_pixels"] = n_used
    depth = gscale * z_ff

    for n in range(N):
        m = dlt_mask[n]
        cov = float(m.mean())
        s_n, _ = _robust_scale(z_dlt[n][m], z_ff[n][m]) if m.any() else (None, 0)
        entry = {"view": n, "dlt_coverage": cov, "view_scale": s_n}

        if mode == "per_view_affine" and m.any():
            fitted, info = _affine_in_disparity(z_ff[n], z_dlt[n], m)
            if fitted is not None:
                depth[n] = fitted
                entry["affine"] = info
            else:
                if s_n:
                    depth[n] = s_n * z_ff[n]
                entry["affine"] = info or {"rejected": True}
        elif mode == "per_view_scale" and s_n:
            depth[n] = s_n * z_ff[n]

        # Wherever a triangulated depth exists it is the better measurement:
        # it comes from RoMaV2 correspondences resolved against the BA cameras,
        # not from a monocular-ish prediction.
        depth[n] = np.where(m, z_dlt[n], depth[n])

        after = depth[n][m]
        if m.any():
            entry["resid_after"] = float(
                np.median(np.abs(after - z_dlt[n][m]) / np.maximum(z_dlt[n][m], 1e-6))
            )
        stats["per_view"].append(entry)

    # Depth <= 0 back-projects behind the camera; point_map.md flags this as a
    # silent corruptor of the init cloud, so clamp to the view's own median.
    bad = ~np.isfinite(depth) | (depth <= 1e-6)
    stats["nonpositive_pixels"] = int(bad.sum())
    if bad.any():
        for n in range(N):
            good = depth[n][~bad[n]]
            depth[n][bad[n]] = float(np.median(good)) if good.size else 1.0

    # Unproject with the *bundle-adjusted* intrinsics and poses.
    jj, ii = np.mgrid[:H, :W].astype(np.float64)
    world = np.empty((N, H, W, 3), dtype=np.float64)
    for n in range(N):
        fx, fy = ba_intr[n, 0, 0], ba_intr[n, 1, 1]
        cx, cy = ba_intr[n, 0, 2], ba_intr[n, 1, 2]
        z = depth[n]
        cam = np.stack([(ii - cx) / fx * z, (jj - cy) / fy * z, z], axis=-1)
        R, t = ba_extr[n, :3, :3], ba_extr[n, :3, 3]
        world[n] = (cam - t) @ R  # R^T (X_cam - t)

    return world, depth, stats


# --------------------------------------------------------------------------
# writing the RI3D contract
# --------------------------------------------------------------------------


def write_outputs(
    output_dir,
    names,
    exts,
    full_images,
    world_points,
    confs,
    dlt_mask,
    ba_extr,
    ba_intr,
    ff_rgb,
    conf_thr,
    save_rgb,
):
    from colmap.read_write_model import (
        Camera,
        Image,
        Point3D,
        rotmat2qvec,
        write_cameras_binary,
        write_cameras_text,
        write_images_binary,
        write_images_text,
        write_points3D_binary,
        write_points3D_text,
    )

    N, Hp, Wp, _ = world_points.shape
    H_full, W_full = full_images[0].shape[:2]

    # images/ -- the grid every intrinsic below refers to.
    img_dir = osp.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    import cv2

    filepaths = []
    for name, ext, img in zip(names, exts, full_images):
        dst = osp.join(img_dir, f"{name}{ext}")
        cv2.imwrite(dst, img[:, :, ::-1])
        filepaths.append(dst)

    # cameras.json -- OpenCV c2w, the array sparse/0 is derived from.
    c2ws = []
    for n in range(N):
        w2c = np.eye(4)
        w2c[:3, :4] = ba_extr[n]
        c2ws.append(np.linalg.inv(w2c))
    c2ws = np.stack(c2ws)

    # fx/fy scale independently: the ff grid is an anisotropic stretch of the
    # full image (518-wide, height rounded to a multiple of 14).
    scale_w, scale_h = W_full / Wp, H_full / Hp
    fx_full = ba_intr[:, 0, 0] * scale_w
    fy_full = ba_intr[:, 1, 1] * scale_h
    cx_full, cy_full = W_full / 2.0 - 0.5, H_full / 2.0 - 0.5

    with open(osp.join(output_dir, "cameras.json"), "w") as f:
        json.dump(
            {
                "filepaths": filepaths,
                "focals": ((fx_full + fy_full) / 2.0).tolist(),
                "cams2world": c2ws.tolist(),
                # extras, ignored by load_sfm_poses but useful when debugging
                "focals_xy": np.stack([fx_full, fy_full], axis=1).tolist(),
                "principal_point": [cx_full, cy_full],
                "pointmap_size": [Wp, Hp],
                "image_size": [W_full, H_full],
            },
            f,
        )

    # sparse/0 -- one COLMAP camera per image, at full resolution.
    cameras, images_colmap, points3d = {}, {}, {}
    all_xyz, all_rgb = [], []
    pid = 1

    grid_u = ((np.arange(Wp) + 0.5) * scale_w - 0.5)[None, :].repeat(Hp, axis=0)
    grid_v = ((np.arange(Hp) + 0.5) * scale_h - 0.5)[:, None].repeat(Wp, axis=1)

    for n in range(N):
        cam_id = n + 1
        cameras[cam_id] = Camera(
            int(cam_id),
            "PINHOLE",
            width=int(W_full),
            height=int(H_full),
            params=np.array([fx_full[n], fy_full[n], cx_full, cy_full]),
        )

        keep = confs[n] > conf_thr
        xyz = world_points[n][keep]
        uu = np.clip(grid_u[keep].astype(np.int64), 0, W_full - 1)
        vv = np.clip(grid_v[keep].astype(np.int64), 0, H_full - 1)
        rgb = full_images[n][vv, uu].astype(np.int64)
        xys = np.stack([grid_u[keep], grid_v[keep]], axis=1)

        ids = np.arange(pid, pid + xyz.shape[0], dtype=np.int64)
        img_ids = np.array([cam_id], dtype=np.int64)
        for k in range(xyz.shape[0]):
            points3d[int(ids[k])] = Point3D(
                id=int(ids[k]),
                xyz=xyz[k],
                rgb=rgb[k],
                error=np.array(2.0, dtype=np.float64),
                image_ids=img_ids,
                point2D_idxs=np.array([k], dtype=np.int64),
            )
        pid += xyz.shape[0]
        all_xyz.append(xyz)
        all_rgb.append(rgb.astype(np.float64) / 255.0)

        w2c = np.eye(4)
        w2c[:3, :4] = ba_extr[n]
        images_colmap[cam_id] = Image(
            cam_id,
            rotmat2qvec(w2c[:3, :3]).astype(np.float64),
            w2c[:3, 3].astype(np.float64),
            cam_id,
            f"{names[n]}{exts[n]}",
            xys.astype(np.float64),
            ids,
        )

    sparse_dir = osp.join(output_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    write_cameras_binary(cameras, osp.join(sparse_dir, "cameras.bin"))
    write_images_binary(images_colmap, osp.join(sparse_dir, "images.bin"))
    write_points3D_binary(points3d, osp.join(sparse_dir, "points3D.bin"))
    write_cameras_text(cameras, osp.join(sparse_dir, "cameras.txt"))
    write_images_text(images_colmap, osp.join(sparse_dir, "images.txt"))
    write_points3D_text(points3d, osp.join(sparse_dir, "points3D.txt"))

    # points.ply -- written with plyfile rather than open3d, which the GGPT
    # environment does not carry.
    from plyfile import PlyData, PlyElement

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = (np.concatenate(all_rgb, axis=0) * 255.0).astype(np.uint8)
    verts = np.empty(
        xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")]).write(osp.join(output_dir, "points.ply"))

    # pointmaps/ -- points flat row-major (H*W, 3), confs 2-D so the grid shape
    # travels with the file. `dlt_mask` is an extra key; load_pointmap ignores it.
    pm_dir = osp.join(output_dir, "pointmaps")
    os.makedirs(pm_dir, exist_ok=True)
    for n, name in enumerate(names):
        with open(osp.join(pm_dir, f"{name}.json"), "w") as f:
            json.dump(
                {
                    "rgb": ff_rgb[n].tolist() if save_rgb else None,
                    "points": world_points[n].reshape(-1, 3).tolist(),
                    "confs": confs[n].tolist(),
                    "dlt_mask": dlt_mask[n].astype(np.uint8).tolist(),
                },
                f,
            )

    return xyz.shape[0], (fx_full, fy_full, cx_full, cy_full, W_full, H_full)


def verify_grid_alignment(output_dir, names, intr_full):
    """Re-run RI3D's own alignment guard on what we just wrote.

    If this fails, `--depth_source pointmap` silently falls back to the sparse
    projection path (~20% pixel coverage) and the init cloud gets ~9x worse --
    the exact failure point_map.md 4 documents. Better to say so loudly.
    """
    # Loaded by file path, NOT by `import utils.pointmap_utils`. GGPT's own
    # `utils` package is already bound in sys.modules by bootstrap(), and no
    # amount of sys.path juggling displaces an imported package -- the import
    # form raises ModuleNotFoundError here. pointmap_utils only needs json /
    # os.path / numpy at module level, so it loads standalone.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_ri3d_pointmap_utils", osp.join(RI3D_ROOT, "utils", "pointmap_utils.py")
    )
    pmu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pmu)
    load_pointmap = pmu.load_pointmap
    load_sfm_poses = pmu.load_sfm_poses
    pointmap_grid_is_aligned = pmu.pointmap_grid_is_aligned

    fx, fy, cx, cy, W, H = intr_full
    poses = load_sfm_poses(output_dir)
    K = np.array([[fx[0], 0, cx], [0, fy[0], cy], [0, 0, 1]], dtype=np.float64)

    ok_all = True
    for i, name in enumerate(names):
        K[0, 0], K[1, 1] = fx[i], fy[i]
        pts, confs = load_pointmap(osp.join(output_dir, "pointmaps"), name)
        ok = pointmap_grid_is_aligned(pts, confs, poses[name], K, W, H)
        print(f"  {name}: grid aligned = {ok}")
        ok_all &= ok
    return ok_all


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene_path", required=True, help="Directory of images, or a scene dir containing images/")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ggpt_root", default="/home/nguyen/projects/GGPT")
    ap.add_argument("--config", default=osp.join(RI3D_ROOT, "configs/ggpt/unposed.yaml"))

    ap.add_argument("--n_images", type=int, default=-1)
    ap.add_argument("--image_idx", type=int, nargs="*", default=None)
    ap.add_argument("--randomize_images", action="store_true")
    ap.add_argument("--use_all_images", action="store_true")
    ap.add_argument("--seed", type=int, default=47)

    ap.add_argument(
        "--fuse_mode",
        default="global_scale",
        choices=["global_scale", "per_view_scale", "per_view_affine"],
        help="How VGGT's dense depth is reconciled with the DLT anchors. "
        "global_scale corrects only the BA gauge and preserves VGGT's "
        "multi-view consistency; per_view_affine also removes each view's own "
        "depth bias (robust fit in disparity space, kept only if it beats a scale).",
    )
    ap.add_argument("--ggpt_refine", action="store_true",
                    help="Run the GGPT PTv3 point transformer on the fused pointmap.")
    ap.add_argument("--output_conf_thr", type=float, default=None,
                    help="Absolute confidence floor for points.ply / sparse. Overrides --output_conf_quantile.")
    ap.add_argument("--output_conf_quantile", type=float, default=0.2,
                    help="Drop this bottom fraction of confidences instead of using an absolute threshold. "
                         "VGGT confidence is not on MASt3R's scale, so a quantile ports across scenes.")
    ap.add_argument("--watermark_mask", default=None,
                    help="PNG mask (255 = ignore) of burned-in watermarks, from "
                         "tools/detect_watermark.py. Masked pixels are excluded from matching, "
                         "bundle adjustment and triangulation, and their pointmap confidence is "
                         "zeroed. The photos themselves are never modified.")
    ap.add_argument("--no_pointmap_rgb", action="store_true",
                    help="Omit the 'rgb' field from pointmaps/*.json (halves their size; RI3D never reads it).")
    args = ap.parse_args()

    scene_path = osp.abspath(args.scene_path)
    output_dir = osp.abspath(args.output_dir)
    config_path = osp.abspath(args.config)
    os.makedirs(output_dir, exist_ok=True)

    bootstrap(args.ggpt_root)

    import torch
    from omegaconf import OmegaConf

    from feedforward import FeedForward_Model
    from matching import init_match_models
    from sfm.sfm_func import run_sfm
    from utils.basic import set_seed

    torch.set_float32_matmul_precision("highest")  # RoMaV2 refuses to run otherwise
    cfg = OmegaConf.load(config_path)
    set_seed(cfg.common_config.get("seed", args.seed))
    device = "cuda:0"

    image_dir = resolve_image_dir(scene_path)
    filelist = select_images(
        image_dir, args.n_images, args.image_idx, args.randomize_images,
        args.use_all_images, args.seed,
    )
    names = [osp.splitext(osp.basename(p))[0] for p in filelist]
    exts = [osp.splitext(p)[1] for p in filelist]
    print(f"[INFO] {len(filelist)} view(s) from {image_dir}: {', '.join(names)}")

    images, full_images, transposed = load_images_ggpt(filelist, cfg.match_config.max_width)
    if transposed:
        print("[INFO] Portrait input transposed to landscape (GGPT convention); "
              "the transposed copies are what gets written to images/.")
    images = images.to(device)
    print(f"[INFO] full {full_images[0].shape[1]}x{full_images[0].shape[0]}, "
          f"match {images.shape[2]}x{images.shape[1]}")

    # 1. VGGT feed-forward: dense pointmap + initial poses/intrinsics.
    ff_model = FeedForward_Model(cfg.feedforward_config).to(device).eval()
    with torch.no_grad():
        ff_outputs = ff_model(images, preprocessed=False)
    print(f"[INFO] VGGT ({cfg.feedforward_config.model}) pointmap "
          f"{ff_outputs['points'].shape[2]}x{ff_outputs['points'].shape[1]}")
    if cfg.common_config.reduce_memory:
        del ff_model
        torch.cuda.empty_cache()

    # 2. RoMaV2 dense matching -> BA -> DLT triangulation.
    #
    # The mask goes on the ff grid because run_sfm matches at lr_h=ff_h, lr_w=ff_w,
    # so that is the grid every match tensor is indexed on -- and the same grid the
    # pointmaps are exported on, so one resample serves both uses.
    ff_h, ff_w = ff_outputs["points"].shape[1:3]
    wm_mask = wm_mask_full = None
    if args.watermark_mask:
        wm_mask_full, wm_mask = load_watermark_mask(
            args.watermark_mask, full_images[0].shape[:2], transposed, ff_h, ff_w
        )
        print(f"[INFO] watermark mask {args.watermark_mask}: "
              f"{wm_mask.mean() * 100:.2f}% of the frame excluded")
        install_match_mask(wm_mask)

    match_models = init_match_models(cfg.match_config.models, device=device)
    print(f"[INFO] matchers: {list(match_models.keys())}")
    sfm_outputs = run_sfm(images, ff_outputs, match_models, cfg, output_dir=output_dir)
    if not sfm_outputs.get("points_success"):
        sys.exit("[ERROR] GGPT SfM failed to triangulate any point. "
                 "Views likely have too little overlap or parallax.")
    del match_models
    torch.cuda.empty_cache()

    ff_points = ff_outputs["points"].detach().float().cpu().numpy().astype(np.float64)
    ff_conf = ff_outputs["points_conf"].detach().float().cpu().numpy().astype(np.float32)
    ff_rgb = (ff_outputs["images_ff"].detach().float().cpu().numpy() * 255).astype(np.uint8)
    ff_extr = ff_outputs["extrinsics"].detach().float().cpu().numpy()[:, :3, :4].astype(np.float64)
    ba_extr = sfm_outputs["extrinsics"].detach().float().cpu().numpy()[:, :3, :4].astype(np.float64)
    ba_intr = sfm_outputs["intrinsics"].detach().float().cpu().numpy().astype(np.float64)
    dlt_points = sfm_outputs["points"].detach().float().cpu().numpy().astype(np.float64)
    dlt_mask = sfm_outputs["point_masks"].detach().cpu().numpy().astype(bool)

    # 3. How healthy was the solve? Reported before fusion so a bad BA is
    # visible even if everything downstream nominally succeeds.
    ba_stats = ba_diagnostics(dlt_points, dlt_mask, ba_extr, ba_intr)
    max_reproj = float(cfg.dlt_config.max_reproj_error)
    if ba_stats["reproj_p50_px"] is None:
        # fuse_pointmaps raises on this below; say why before it does.
        print("[WARN] no triangulated anchor survived filtering -- nothing to measure.")
    else:
        print(f"[INFO] anchor reprojection error: p50 {ba_stats['reproj_p50_px']:.3f} px  "
              f"p90 {ba_stats['reproj_p90_px']:.3f} px  (dlt max_reproj_error {max_reproj})")
    if ba_stats["reproj_p90_px"] and ba_stats["reproj_p90_px"] > max_reproj:
        print("[WARN] 10%+ of triangulated anchors miss their own pixel by more than the "
              "DLT filter allows. The bundle adjustment likely did not settle -- check the "
              "Ceres report above, and consider raising ba_config.mintrack_per_view or "
              "loosening ba_config.score_thresh.")

    # 4. Reconcile the dense VGGT pointmap with the triangulated anchors.
    world, depth, stats = fuse_pointmaps(
        ff_points, ff_extr, dlt_points, dlt_mask, ba_extr, ba_intr, args.fuse_mode
    )
    print(f"[INFO] fuse mode={stats['mode']} global_scale={stats['global_scale']:.6f} "
          f"anchors={stats['anchor_pixels']} nonpositive={stats['nonpositive_pixels']}")
    for e in stats["per_view"]:
        vs = f"{e['view_scale']:.4f}" if e["view_scale"] else "n/a"
        print(f"  view {e['view']}: dlt coverage {e['dlt_coverage']*100:5.1f}%  view scale {vs}")

    # 5. Optional PTv3 refinement of the dense pointmap.
    confs = ff_conf
    if args.ggpt_refine:
        world, confs = run_ggpt_refine(cfg, ff_outputs, sfm_outputs, world, device)

    # Watermark pixels carry no real geometry -- VGGT predicted depth for a logo
    # pasted onto the lens, not for the surface behind it. Zeroing the confidence
    # puts them below the export threshold, so they never reach points.ply or
    # sparse/0, and marks them in confs<N>.npy for anything downstream that gates
    # on confidence.
    if wm_mask is not None:
        confs = confs.copy()
        confs[:, wm_mask] = 0.0

    # 6. Write the RI3D contract.
    if args.output_conf_thr is not None:
        conf_thr = args.output_conf_thr
    else:
        conf_thr = float(np.quantile(confs, args.output_conf_quantile))
    print(f"[INFO] confidence: min {confs.min():.3f} median {np.median(confs):.3f} "
          f"max {confs.max():.3f} -> threshold {conf_thr:.3f}")

    n_pts, intr_full = write_outputs(
        output_dir, names, exts, full_images, world, confs, dlt_mask,
        ba_extr, ba_intr, ff_rgb, conf_thr, save_rgb=not args.no_pointmap_rgb,
    )
    print(f"[INFO] wrote {n_pts} points to points.ply")

    if wm_mask is not None:
        # Keep the mask beside the solve it shaped, so later stages find it without
        # re-deriving which mask was used. Saved at *image* resolution, matching
        # images/ and depth_rel/, because that is the grid the 3DGS training losses
        # work on; the pointmap-grid copy sits next to it for match-side debugging.
        import cv2

        masks_dir = osp.join(output_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        cv2.imwrite(osp.join(masks_dir, "watermark_mask.png"),
                    wm_mask_full.astype(np.uint8) * 255)
        cv2.imwrite(osp.join(masks_dir, "watermark_mask_pointmap.png"),
                    wm_mask.astype(np.uint8) * 255)
        print(f"[INFO] wrote masks/watermark_mask.png at "
              f"{wm_mask_full.shape[1]}x{wm_mask_full.shape[0]} (image resolution)")

    print("[INFO] verifying pointmap grid alignment against RI3D's own guard:")
    if not verify_grid_alignment(output_dir, names, intr_full):
        print("[WARN] At least one pointmap is NOT linearly aligned to the image grid. "
              "`--depth_source pointmap` will fall back to sparse projection "
              "(~20% coverage). See docs/point_map.md 4.")

    with open(osp.join(output_dir, "ggpt_sfm_stats.json"), "w") as f:
        json.dump(
            {
                "views": names,
                "backend": f"vggt({cfg.feedforward_config.model}) + {list(cfg.match_config.models)}",
                "ggpt_refine": bool(args.ggpt_refine),
                "watermark_mask": None if wm_mask is None else {
                    "source": args.watermark_mask,
                    "coverage": float(wm_mask.mean()),
                    "masked_pixels_per_view": int(wm_mask.sum()),
                },
                "bundle_adjustment": ba_stats,
                "fusion": stats,
                "conf_thr": conf_thr,
            },
            f,
            indent=2,
        )
    print("\n[INFO] GGPT-SfM complete.")


def run_ggpt_refine(cfg, ff_outputs, sfm_outputs, world, device):
    """Refine the fused pointmap with GGPT's PTv3 point transformer."""
    import torch
    from hydra.utils import instantiate
    from tqdm import tqdm

    from ggpt.dataloader.demo_dataset import DemoDataset
    from sfm.run_benchmark_sfm import move_to_device
    from utils.points import aggregate_chunks

    model = instantiate(cfg.ggptmodel_config).eval()
    ckpt = torch.load(cfg.common_config.ggpt_ckpt, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()}, strict=True)
    model = model.to(device)

    # Feed the fused (BA-frame) pointmap in, not VGGT's raw one, so the
    # transformer refines the same geometry the rest of this script exports.
    ff_in = dict(ff_outputs)
    ff_in["points"] = torch.from_numpy(world).float().to(device)
    ds = DemoDataset(name="demo", ff_data=ff_in, geo_data=sfm_outputs)
    chunks, scene = ds[0]

    pts, cfs = [], []
    for chunk in tqdm(([c] for c in chunks), total=len(chunks), desc="GGPT refine"):
        chunk = move_to_device(chunk, device)
        with torch.no_grad():
            out = model(chunk)
        pts.append(ds.unnormalize_pts(chunk[0], out["ff_pts_out"]))
        cfs.append(out["ff_pts_conf_out"])

    msks = torch.stack([c["msks_in_scene"] for c in chunks], dim=0).to(device)
    pred_pts, pred_confs, _ = aggregate_chunks(
        torch.cat(pts, dim=0), torch.cat(cfs, dim=0), msks, scene
    )
    return (
        pred_pts.detach().float().cpu().numpy().astype(np.float64),
        pred_confs.detach().float().cpu().numpy().astype(np.float32),
    )


if __name__ == "__main__":
    main()
