"""Derive a watermark mask from a set of photos that share one composited overlay.

Listing-site photos often carry a logo or repeated watermark burned into every
frame at the *same* pixel coordinates. That is worse than noise for SfM: GGPT
ranks bundle-adjustment track candidates by SuperPoint saliency
(``sfm/sfm_func.py``: ``argsort(candidate_sps, descending=True)``), and a
watermark edge is a strong, perfectly repeatable corner -- so watermark pixels
are *preferentially* promoted into BA, where they match themselves at zero
disparity and contradict the real camera motion.

Detection signal: **cross-image edge persistence**. A watermark edge appears at
the same pixel in every photo; a scene edge moves as the camera moves. So the
per-pixel minimum of the Sobel magnitude over the stack is high on the watermark
and near zero everywhere else -- no model, no network, sub-second.

    gmin = min_i |grad I_i|

Measured with the defaults below:

    examples/sceneB  10 views, watermarked   p99(gmin) 135.8   8 components   11.4% masked
    examples/sceneA   3 views, clean         p99(gmin)  24.7   (rejected: no watermark)

Detected components are grouped and boxed by default: a logo is several
disconnected pieces (a ring drawn as two arcs, letters, a tagline), and an
edge detector cannot see a solid interior, so masking exact shapes leaves the
bulk of a filled logo behind. See --no_fill_boxes.

**Assumption: one shared mask per scene.** That is the same condition that makes
the detector work at all -- a watermark at varying positions would be neither
detectable this way nor maskable with a single PNG.

Writes ``watermark_mask.png`` (single channel, 255 = ignore this pixel) and
``watermark_preview.png`` (the mask overlaid on the first frame, for eyeballing).
Both are inputs to the SfM stage, which never modifies the photos themselves.
"""

import argparse
import os
import os.path as osp
import sys

import cv2
import numpy as np
from scipy import ndimage

VALID_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def resolve_image_dir(scene_path):
    """Mirror the directory probing `inference.py` performs on the input."""
    for sub in ("images", "images_4", "images_2", "images_8"):
        cand = osp.join(scene_path, sub)
        if osp.isdir(cand):
            return cand
    return scene_path


def load_stack(image_dir):
    """Grayscale float32 stack, all frames conformed to the first one's shape."""
    names = sorted(f for f in os.listdir(image_dir) if f.endswith(VALID_EXTS))
    if not names:
        sys.exit(f"[ERROR] No images found in {image_dir}")

    gray, shape, kept = [], None, []
    for n in names:
        img = cv2.imread(osp.join(image_dir, n), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        gray.append(img.astype(np.float32))
        kept.append(n)

    if not gray:
        sys.exit(f"[ERROR] No readable images in {image_dir}")
    return np.stack(gray), kept, osp.join(image_dir, kept[0])


def edge_persistence(stack):
    """Per-pixel minimum Sobel magnitude across the stack."""
    out = None
    for img in stack:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.magnitude(gx, gy)
        out = g if out is None else np.minimum(out, g)
    return out


def build_mask(gmin, thr, dilate, min_component, fill_boxes=False, merge_distance=13):
    """Threshold -> close -> fill holes -> dilate -> drop specks.

    Hole filling is not cosmetic: the detector fires on *edges*, so the ring of a
    circular logo mark comes out hollow and its interior would stay unmasked.
    """
    m = gmin > thr
    m = ndimage.binary_closing(m, np.ones((7, 7)))
    m = ndimage.binary_fill_holes(m)
    if dilate > 0:
        m = ndimage.binary_dilation(m, np.ones((dilate, dilate)))

    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        small = np.flatnonzero(sizes < min_component) + 1
        if small.size:
            m = m & ~np.isin(lab, small)
        lab, n = ndimage.label(m)

    if fill_boxes:
        # Mask each component's whole bounding box. Nothing inside a logo's box is
        # worth preserving, and it sweeps up the faint tagline strokes sitting just
        # below the detection threshold.
        #
        # Group first, at `merge_distance`. A logo is usually several disconnected
        # pieces -- a ring drawn as two arcs, letters, a tagline -- and boxing each
        # piece separately leaves the gaps between them uncovered, which is exactly
        # where a solid logo interior survives. Grouping is done on a dilated copy
        # so the boxes come from merged clusters, while the mask itself keeps its
        # original extent.
        if merge_distance > 0:
            grouped = ndimage.binary_dilation(m, np.ones((merge_distance, merge_distance)))
        else:
            grouped = m
        lab, n = ndimage.label(grouped)
        for sl in ndimage.find_objects(lab):
            m[sl] = True
        lab, n = ndimage.label(m)

    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(ys.size)))
    boxes.sort(key=lambda b: -b[4])
    return m, boxes


def write_preview(path, image_path, mask):
    """Mask overlaid in red, with contours, on the first frame."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    if img.shape[:2] != mask.shape:
        img = cv2.resize(img, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = img.copy()
    overlay[mask] = (0, 0, 255)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    contours = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
    cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--scene_path", required=True,
                    help="Directory of images, or a scene dir containing images/")
    ap.add_argument("-o", "--output_dir", required=True,
                    help="Where watermark_mask.png and watermark_preview.png are written")
    ap.add_argument("--threshold", type=float, default=60.0,
                    help="Sobel magnitude a pixel must exceed in EVERY view to count as watermark")
    ap.add_argument("--dilate", type=int, default=5,
                    help="Dilation kernel size; grows the mask past the anti-aliased fringe")
    ap.add_argument("--min_component", type=int, default=300,
                    help="Drop connected components smaller than this many pixels")
    ap.add_argument("--max_coverage", type=float, default=0.25,
                    help="Refuse a mask covering more than this fraction of the frame")
    ap.add_argument("--merge_distance", type=int, default=13,
                    help="With --fill_boxes, group components within this many pixels before "
                         "boxing them. A logo is several disconnected pieces (a ring drawn as two "
                         "arcs, letters, a tagline); boxing each separately leaves the gaps between "
                         "them - and a solid logo interior - unmasked.")
    ap.add_argument("--no_fill_boxes", dest="fill_boxes", action="store_false",
                    help="Mask each component's exact shape instead of its bounding box. Tighter, "
                         "but a solid logo interior has no gradient for the detector to fire on, so "
                         "it survives: on sceneB the exact-shape mask covers only 20%% of the "
                         "circular logo mark against 99%% for the default (frame total 5.5%% vs "
                         "11.4%%). Use it only when masking real content is costlier than leaving "
                         "watermark behind.")
    ap.set_defaults(fill_boxes=True)
    args = ap.parse_args()

    scene_path = osp.abspath(args.scene_path)
    output_dir = osp.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_dir = resolve_image_dir(scene_path)
    stack, names, first_path = load_stack(image_dir)
    n_views, H, W = stack.shape
    print(f"[INFO] {n_views} view(s) from {image_dir} at {W}x{H}")
    if n_views < 5:
        print("[WARN] Fewer than 5 views. This detector separates watermark edges from scene "
              "edges by requiring the scene to *change* between views; with few or near-identical "
              "views, persistent scene edges can be flagged too. Check the preview.")

    gmin = edge_persistence(stack)
    p95, p99 = float(np.percentile(gmin, 95)), float(np.percentile(gmin, 99))
    print(f"[INFO] cross-view edge persistence: p95 {p95:.1f}  p99 {p99:.1f}  "
          f"(threshold {args.threshold:.0f})")

    # A clean scene has no edge surviving every view: sceneA's p99 is 24.7 against
    # sceneB's 135.8. Below the threshold there is nothing to mask, and emitting a
    # mask anyway would silently discard real content.
    if p99 < args.threshold:
        print(f"[INFO] No watermark detected (p99 {p99:.1f} < {args.threshold:.0f}). "
              "No mask written; the SfM stage will use every pixel.")
        return 0

    mask, boxes = build_mask(
        gmin, args.threshold, args.dilate, args.min_component, args.fill_boxes,
        args.merge_distance,
    )
    coverage = float(mask.mean())

    if not boxes:
        print("[INFO] Nothing survived component filtering; no mask written.")
        return 0

    print(f"[INFO] {len(boxes)} component(s), {coverage * 100:.2f}% of the frame masked")
    for x0, y0, x1, y1, px in boxes[:20]:
        print(f"    {px:>7} px  ({x0:>4},{y0:>4})-({x1:>4},{y1:>4})")
    if len(boxes) > 20:
        print(f"    ... and {len(boxes) - 20} more")

    # How much persistent-edge energy escaped the mask? Faint tagline strokes sit
    # below --threshold and are only partly swept up by the dilation. A handful of
    # pixels is fine; a large number means --threshold is set too high for this
    # watermark. Anything under sceneA's clean-scene p99 (24.7) is indistinguishable
    # from ordinary persistent scene structure and not worth chasing.
    leak = int(((gmin > max(40.0, p95)) & ~mask).sum())
    print(f"[INFO] residual persistent edge outside the mask: {leak} px "
          f"({leak / mask.size * 100:.3f}% of frame)"
          + ("" if leak < 2000 else "  <- consider lowering --threshold or --fill_boxes"))

    if coverage > args.max_coverage:
        print(f"[ERROR] Mask covers {coverage * 100:.1f}% of the frame, above the "
              f"{args.max_coverage * 100:.0f}% ceiling. That means the detection is wrong, not "
              "that the watermark is huge -- most likely the views are too similar for scene "
              "edges to move. Raise --threshold, or hand a drawn mask to --watermark_mask.")
        return 1

    mask_path = osp.join(output_dir, "watermark_mask.png")
    preview_path = osp.join(output_dir, "watermark_preview.png")
    cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
    write_preview(preview_path, first_path, mask)
    print(f"[INFO] wrote {mask_path}")
    print(f"[INFO] wrote {preview_path}  <- check this before trusting the mask")
    return 0


if __name__ == "__main__":
    sys.exit(main())
