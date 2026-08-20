"""
Align Depth-Anything's affine-invariant disparity to MASt3R's metric depth.

Depth-Anything predicts disparity that is only defined up to an unknown scale
and shift, so it cannot be used as depth directly -- doing so inverts near/far
and lands roughly an order of magnitude off the SfM world scale. This module
recovers the missing affine transform from the metric anchors that MASt3R's
pointmap provides (see `utils.pointmap_utils`), yielding a dense depth map in
MASt3R world units that keeps Depth-Anything's high-resolution detail.

The two-stage recipe mirrors the one already used at inference time in
`threestudio/systems/gaussian_object_system_mip.py` (where the anchors are
rendered depth instead of pointmap depth): a global fit in disparity space,
then a monotone piecewise-linear refinement binned by `get_depth_bins`.
"""

import numpy as np

from utils.depth_layering import get_depth_bins


def robust_affine_disparity(disp, inv_depth, weights=None, iters=10, huber_k=1.345):
    """IRLS Huber fit of ``a * disp + b ~= inv_depth``.

    Working in disparity space keeps the model linear and puts the residual
    weighting where the monocular network is actually well conditioned -- an
    equivalent fit on depth would be dominated by far-field pixels.

    Returns (a, b, inlier_mask, sigma).
    """
    disp = np.asarray(disp, dtype=np.float64).reshape(-1)
    inv_depth = np.asarray(inv_depth, dtype=np.float64).reshape(-1)

    if weights is None:
        w0 = np.ones_like(disp)
    else:
        w0 = np.asarray(weights, dtype=np.float64).reshape(-1)
        w0 = w0 / max(w0.mean(), 1e-12)

    w = w0.copy()
    a, b, sigma = 0.0, 0.0, 1.0
    resid = np.zeros_like(disp)
    for _ in range(iters):
        A = np.stack([disp, np.ones_like(disp)], axis=1) * w[:, None]
        a, b = np.linalg.lstsq(A, inv_depth * w, rcond=None)[0]
        resid = np.abs(a * disp + b - inv_depth)
        sigma = 1.4826 * np.median(resid) + 1e-12
        w = w0 * np.clip(huber_k * sigma / np.maximum(resid, 1e-12), None, 1.0)

    return a, b, resid < 3.0 * sigma, sigma


def piecewise_refine(pred_depth, ref_depth, mask, num_bins=5):
    """Monotone piecewise-linear correction of `pred_depth` towards `ref_depth`.

    Fits in disparity space over the masked anchor pixels, then applies the fit
    to every pixel. Same estimator and bins as the inpainting path in
    `gaussian_object_system_mip.py`, so data prep and inference stay consistent.

    Returns the corrected depth, or None if the fit could not be made.
    """
    try:
        import torch
        from ropwr import RobustPWRegression
    except ImportError:
        return None

    if mask.sum() < 100:
        return None

    clamped = np.clip(pred_depth, 1e-2, None)
    try:
        bins = get_depth_bins(
            depth=torch.tensor(clamped, dtype=torch.float32)[None, None],
            num_bins=num_bins,
            mask=torch.tensor(mask.astype(np.float32))[None, None],
        )
    except Exception:
        return None

    splits = sorted({1.0 / b for b in bins if b > 1e-6})
    if len(splits) < 3:
        return None
    splits = splits[1:-1]

    pw = RobustPWRegression(
        objective="huber",
        degree=1,
        monotonic_trend="ascending",
        extrapolation="continue",
    )
    try:
        pw.fit(
            1.0 / clamped[mask].reshape(-1),
            1.0 / np.clip(ref_depth[mask], 1e-2, None).reshape(-1),
            splits=splits,
        )
        refined_disp = pw.predict(1.0 / clamped.reshape(-1)).reshape(pred_depth.shape)
    except Exception:
        return None

    refined_disp = np.clip(refined_disp, 1e-4, None)
    refined = 1.0 / refined_disp
    if not np.isfinite(refined).all() or refined.min() <= 0:
        return None
    return refined.astype(np.float32)


def align_mono_to_pointmap(
    disp,
    ref_depth,
    valid,
    conf=None,
    min_inlier_ratio=0.3,
    num_bins=5,
    refine=True,
):
    """Turn Depth-Anything disparity into metric depth anchored on MASt3R.

    Args:
        disp: (H, W) Depth-Anything disparity at the target resolution.
        ref_depth: (H, W) sparse metric depth from the pointmap, 0 where absent.
        valid: (H, W) bool mask of pixels where `ref_depth` is meaningful.
        conf: optional (H, W) MASt3R confidence, used to weight the fit.

    Returns (depth, info). `info` carries the fitted coefficients and residual
    diagnostics, plus a `fallback` reason when the monocular fit was rejected
    and the caller is getting hole-filled pointmap depth instead.
    """
    info = {"fallback": None, "n_anchors": int(valid.sum())}

    usable = valid & np.isfinite(disp) & (disp > 1e-6) & (ref_depth > 1e-6)
    if usable.sum() < 100:
        info["fallback"] = f"only {int(usable.sum())} usable anchor pixels"
        return None, info

    weights = np.sqrt(np.clip(conf[usable], 0.0, None)) if conf is not None else None
    a, b, inliers, sigma = robust_affine_disparity(
        disp[usable], 1.0 / ref_depth[usable], weights=weights
    )
    inlier_ratio = float(inliers.mean())
    info.update({"a": float(a), "b": float(b), "inlier_ratio": inlier_ratio,
                 "sigma": float(sigma)})

    # A non-positive slope means the fit found no monotone disparity->depth
    # relationship, i.e. the monocular prediction disagrees with the geometry.
    if not np.isfinite(a) or a <= 0:
        info["fallback"] = f"degenerate affine slope a={a:.3e}"
        return None, info
    if inlier_ratio < min_inlier_ratio:
        info["fallback"] = f"inlier ratio {inlier_ratio:.2f} < {min_inlier_ratio}"
        return None, info

    pred_disp = np.clip(a * disp.astype(np.float64) + b, 1e-4, None)
    depth = (1.0 / pred_disp).astype(np.float32)

    def _rel_err(d):
        return float(np.median(np.abs(d[usable] - ref_depth[usable]) / ref_depth[usable]))

    base_err = _rel_err(depth)
    info["refined"] = False

    # The piecewise fit only pays off when the disparity->depth relationship is
    # genuinely non-affine. On well-conditioned views it over-fits the anchor
    # distribution and compresses the far field, so accept it only when it
    # actually lowers the residual.
    if refine:
        refined = piecewise_refine(depth, ref_depth, usable, num_bins=num_bins)
        if refined is not None and _rel_err(refined) < base_err:
            depth = refined
            info["refined"] = True

    info["median_rel_err"] = _rel_err(depth)
    info["affine_rel_err"] = base_err
    info["depth_range"] = (float(depth.min()), float(depth.max()))
    return depth, info
