# SfM Pointmaps, Depth Priors, and Coordinate Conventions

This document covers how the SfM stage's pointmaps become the metric depth prior RI3D consumes,
the coordinate conventions involved, and a set of defects that were fixed along the way. Several
of those defects were invisible at runtime — the pipeline ran to completion and produced
plausible-looking numbers while the geometry was wrong — so the symptoms and the checks that
expose them are recorded here as well.

Two SfM backends write this contract: `mast3r` (MASt3R-SfM) and `ggpt` (VGGT dense pointmap
refined by RoMaV2 matching, bundle adjustment and DLT triangulation — the default). Everything
below applies to both unless a section says otherwise. What differs is confined to §1 (artifacts),
§2 (grid geometry), §4 (confidence scales) and §8 (how the GGPT pointmap is built).

---

## 1. The two modalities

The SfM stage writes, per scene, into `<scene>/<backend>_sfm/`:

| artifact | contents | frame |
|---|---|---|
| `cameras.json` | `filepaths`, `focals`, `cams2world` | **OpenCV** c2w |
| `sparse/0/*.bin` | COLMAP cameras/images/points3D | **OpenCV** w2c |
| `pointmaps/<name>.json` | dense per-view world XYZ (flat) + `confs` (2-D) | world |
| `points.ply` | confidence-filtered fused point cloud | world |
| `images/` | the frames the exported intrinsics refer to | — |
| `masks/watermark_mask.png` | only when `--watermark_mask` was used; on the **pointmap grid** | — |
| `transforms.json` | written by `inference.py`, NeRF-style | **OpenGL** c2w |

The pointmap JSON stores `points` **flattened** row-major and `confs` as a 2-D array — the conf
array carries the grid shape, so reshape `points` using `confs.shape`. The GGPT backend adds a
`dlt_mask` key (which pixels carry a triangulated depth) and a sibling `ggpt_sfm_stats.json`;
`load_pointmap` ignores both.

**`images/` is not a copy of the input.** MASt3R centre-crops to its own aspect ratio; GGPT
transposes portrait input to landscape. Downstream stages read `<backend>_sfm/images/`, and every
intrinsic in `sparse/0` and `transforms.json` describes *those* frames, not the originals.

**Watermark masking never modifies pixels.** When `--watermark_mask` is supplied (see
[`pipeline.md`](pipeline.md) stage `wm`), masked pixels are excluded from matching, bundle
adjustment and triangulation, and their entry in `confs<N>.npy` is set to **0** so they fall below
the export threshold and never reach `points.ply` or `sparse/0`. `depth_rel` still carries a depth
value there — zeroing confidence does not remove depth, and `--depth_conf_thr` is off by default
and unsupported in `train_gs_init.py` (§4). The mask saved under `masks/` is stored on the
*pointmap* grid, the same grid it was applied on, not at full image resolution.

RI3D consumes a separate prior:

| artifact | consumed by | expected contents |
|---|---|---|
| `depth_rel/inpv2<name>_<N>.npy` | `scene/dataset_readers_flow.py`, `threestudio/data/loo_mip.py` | **metric z-depth, SfM world units** |
| `depth_rel/inp_dust3r<name>_<N>.npy` | same (init point cloud path) | same |
| `depths<N>.npy` | `GaussianModel.flows` | stacked per-view depth |
| `confs<N>.npy` | `GaussianModel.masks` | stacked per-view SfM confidence |

`depths<N>.npy` / `confs<N>.npy` are indexed **positionally** against the reader's final train
list. That order is: sort all cameras by `image_name`, index with `train_ids` from
`train_test_split_<N>.json`, then subsample with `linspace(0, n-1, N)`. `resolve_train_view_names()`
in `inference.py` reproduces it; if the two ever diverge the arrays are silently mismatched.

---

## 2. Coordinate conventions

This is the single most error-prone part of the integration.

```
cameras.json / sparse/0   OpenCV   +x right, +y down, +z forward
transforms.json           OpenGL   +x right, +y up,   -z forward
```

`inference.py:convert_colmap_to_transforms_json()` converts on write:

```python
c2w_gl = c2w_cv @ np.diag([1, -1, -1, 1])
```

and `scene/dataset_readers_flow.py:readMipTransforms()` undoes it on read. The flip is applied
on the **right**, so it negates rotation columns 1 and 2 but leaves the translation column
untouched — **camera centres are identical in both conventions, only orientation differs.** That
asymmetry is why a convention error shows up as "cameras are in the right place but facing the
wrong way" rather than as an obvious scatter.

`M @ M == I`, so anything reading `transforms.json` and needing OpenCV applies the same
`diag([1,-1,-1,1])` again.

### Resolution bookkeeping

`readMipTransforms()` sets `cam_info.width = resolution * transforms.w`, and the `K` build later
in the same file divides by `resolution`. These cancel, so `K` is correct **only if the depth map
matches `transforms.json`'s `w`×`h` exactly.**

The catch: the supported scene types disagree about what those fields mean.

- **mip-NeRF 360** — `transforms.json` stores the ÷4 (`images_4`) dimensions.
- **SfM output** — `convert_colmap_to_transforms_json` stores full-resolution dimensions.

**Any code writing a depth map must take its target size from `transforms.json`'s `w`/`h`, never
from the image file on disk.** `utils/pointmap_utils.load_target_intrinsics()` is the single place
that reads it.

### Pointmap grid → image grid

`resample_view_depth` upsamples the pointmap grid straight onto the target grid, which is only
valid when pointmap pixel (i, j) lands on image pixel `((i+.5)·W/Wp − .5, (j+.5)·H/Hp − .5)`.
`pointmap_grid_is_aligned` guards this at a 1.5 px, 99th-percentile tolerance; when it fails,
`--depth_source pointmap` silently drops to the ~20%-coverage projection path (§4).

The two backends satisfy it differently, and the GGPT case is the one that needs care:

- **MASt3R** — pointmap is exactly half the (already centre-cropped) image, e.g. 384×512 for
  1024×768. Isotropic, `fl_x == fl_y`.
- **GGPT** — VGGT runs at 518 wide with the height rounded to a multiple of 14, so a 1024×768
  photo becomes a 392×518 grid: aspect 1.3214 against the image's 1.3333. The pointmap is an
  **anisotropic stretch** of the *full* image, with no crop. It is still exactly linear, so the
  guard passes — but only because `ggpt_sfm/run_ggpt.py` scales `fx` by `W/Wp` and `fy` by `H/Hp`
  *independently* and puts the principal point at `(W/2 − 0.5, H/2 − 0.5)`. Using one isotropic
  scale, or COLMAP's `W/2` pixel-corner convention, reintroduces a several-pixel drift.

This is why `configs/ggpt/unposed.yaml` sets `camera_type: PINHOLE` rather than GGPT's own
`SIMPLE_PINHOLE` default: on that stretched grid `fx != fy`. Measured on sceneA the exported
camera is `fl_x 744.00, fl_y 735.46` — a 1.2% split that a shared-focal model would have to
absorb somewhere.

---

## 3. Defect: Depth-Anything disparity used as metric depth

### Symptom

Reconstruction quality quietly poor. No error, no warning. The initial point cloud was inverted
near-to-far and roughly 20× out of scale relative to the poses.

### Cause

`utils/depth_utils.estimate_depth()` returns Depth-Anything's raw output, which is
**affine-invariant disparity** — unnormalised, no metric meaning. It was written straight into
`depth_rel/*.npy`, which every consumer reads as metric depth.

The reader appeared to convert, but did not:

```python
vis_depths = [1 / depth_rel]
depth = torch.Tensor(1 / vis_depths[-1])   # 1/(1/x) == x  -- an exact identity
```

The non-flow reader (`scene/dataset_readers.py`) normalises and *then* inverts; the `_flow`
variant had the inversion removed because it was written for DUSt3R **metric** depth. Feeding it
Depth-Anything disparity inverts the geometry.

Measured on sceneA:

| quantity | value |
|---|---|
| `depth_rel` (DA disparity) | 768×1024, range [0, 389], median 37–76 |
| MASt3R z-depth, same views | 384×512, median 2.4–3.1 |
| `corr(disparity, 1/z)` | **+0.567** |
| `corr(disparity, z)` | **−0.517** |
| back-projected cloud vs `points.ply` | **36.4×** extent, centroid `[10.3, −27.0, −4.9]` |

Additionally `depths<N>.npy` was written as **all zeros** and `confs<N>.npy` as a **constant 5.0**,
both at ÷4 resolution — so `GaussianModel.flows`/`masks` carried no signal, and the reader's
confidence gate could not have worked even if enabled (192×256 confidences cannot index
768×1024 points).

### Fix

`depth_rel` now always holds metric depth. The identity round-trip is gone, the dummy arrays are
gone, and `_check_metric_depth()` in the reader warns when a loaded map still looks like
disparity (median > 20 — measured populations were 37–76 disparity vs 2.5–3.9 metric).

The five monocular depth losses were also inconsistent with each other: `train_gs.py` correlated
`mono_depth` directly, while `leave_one_out_stage{1,2}.py` and
`gaussian_object_system_mip.py` negated it (`-zoe_depth`, `1/(zoe_depth + 200.)`) — two different
ways of flipping disparity into a depth-like ordering. All now assume metric depth.

---

## 4. `--depth_source`: where the depth prior comes from

```
python inference.py -i <scene> --depth_source pointmap   # default
python inference.py -i <scene> --depth_source align
```

**`pointmap`** — resample the SfM backend's dense pointmap depth onto the target grid.
Metric and multi-view consistent by construction: it is the same geometry the poses came from.

**`align`** — run Depth-Anything and fit its disparity onto the pointmap's metric anchors
(`utils/depth_align.py`: robust IRLS-Huber affine in disparity space, then an optional monotone
piecewise refinement accepted only when it lowers the residual). Keeps the monocular network's
fine detail.

### Why `pointmap` is the default

The pointmap is a **dense depth image on its own grid** (512×384 under `mast3r`, 518×392 under
`ggpt`), and that grid maps linearly onto the full image — see §2. The original implementation
*splatted* it as individual 3D points into the 1024×768 target grid, which at 2× upsampling covers
only ~20% of pixels and leaves the monocular prior to invent the other 80%.

The global affine fit cannot carry that load. Its residual has a systematic depth-dependent
component and very heavy tails (measured on sceneA under `--sfm_backend mast3r`; the argument is
about the *fit*, not the backend, and applies unchanged to `ggpt`):

| view | p50 | p90 | p99 | max | `corr(err, z)` |
|---|---|---|---|---|---|
| image_2 | 1.46% | 10.86% | 60.85% | 2214% | −0.364 |
| image_3 | 1.31% | 6.10% | 47.83% | 220% | −0.156 |
| image_4 | 2.56% | 28.58% | 38.00% | 113% | −0.263 |

Negative `corr(err, z)` means the far field is progressively compressed toward the camera — each
view bends differently, so the per-view clouds disagree with each other and with MASt3R.

Resulting stage-1a initial point cloud, distance to the nearest MASt3R point:

| mode | coverage | p50 | p90 | mean |
|---|---|---|---|---|
| `align` | ~20% | 0.0288 | 0.1064 | 0.0467 |
| `pointmap` | **100%** | **0.0032** | **0.0240** | **0.0132** |

Cross-view consistency (reproject view *i*'s depth into view *j*; independent of MASt3R's own
cloud, so not biased toward either source): p50 **2.77% → 1.25%**.

> **Aggregate statistics hide this.** Bounding-box extent ratio is ~1.0 for *both* modes
> (0.98–1.04). Robust extent and centroid agreement are not sufficient checks — use per-point
> nearest-neighbour distance.

### Confidence scales are backend-specific

`confs<N>.npy` holds whatever confidence the backend produced, unrescaled — deliberately, since
silently renormalising a prior is the failure mode this document exists to prevent. The
populations are not comparable:

| backend | source | min | median | max |
|---|---|---|---|---|
| `mast3r` | MASt3R conf | 0.00 | 5.75 | 21.6 |
| `ggpt` | VGGT `depth_conf` | 1.00 | 10.4 | 24.4 |

Two consequences. A `--depth_conf_thr` value tuned on one backend means something different on the
other. And `project_pointmap`'s built-in `conf_thr=0.1` is **inert** under `ggpt`, because VGGT's
floor is 1.0 — nothing is ever filtered by it.

Because of this, `ggpt_sfm/run_ggpt.py` thresholds `points.ply` and `sparse/0` by *quantile*
(`--output_conf_quantile`, default 0.2, matching GGPT's own demo) rather than by an absolute
value, so the same setting ports across scenes. `--output_conf_thr` overrides with an absolute
floor. The value actually used is recorded in `ggpt_sfm_stats.json`.

### Low-confidence and invalid pixels

`pointmap` mode takes SfM depth at every pixel regardless of confidence. Two consequences:

- MASt3R occasionally places a pixel **behind** the camera (`z <= 0`) — rare (0.008% on sceneA)
  and low-confidence (conf median 0.61 vs 5.84 overall), but a negative value in `depth_rel`
  back-projects behind the view. `_depth_from_pointmap()` hole-fills these. Note the disparity
  guard cannot catch them: it filters to `z > 0` before taking its median. The GGPT worker clamps
  the same case to the view's median depth before export and reports the count as
  `fusion.nonpositive_pixels` (0–1 pixels of 203k on sceneA).
- Whole views can be low-confidence (one sceneA view had 0.2% of pixels above conf 1.5).
  `confs<N>.npy` is written so these can be gated at init time via `--depth_conf_thr`.

`--depth_conf_thr` (in `scripts/train_gs.py`) drops init points below a confidence threshold.
It is **off by default and unsupported in `train_gs_init.py`**: filtering makes per-view point
counts unequal, which breaks the `fused_point_cloud.shape[0] // num_cameras` reshape that
`create_from_pcd` performs when `mono_d_so_enable=True`.

---

## 5. Artifact: `<scene>/point_cloud.ply` is in CAMERA frame

**Symptom.** In the Rerun viewer, `points.ply` aligns across views but `point_cloud.ply` does not.

**Cause.** `point_cloud.ply` is not a MASt3R output. `scene/dataset_readers_flow.py` writes it as
an intermediate every time a `Scene` is constructed, and its points are in each view's **camera
frame**, concatenated with no camera-to-world transform (the variable is literally named
`xyz_cam`; the `w2c` block below it is commented out).

Verified: treating each per-view block as camera-frame points on an identity pose gives
`median |x/z − (u−cx)/fx| = 4.0e-08`. Against the actual cameras, no pose or convention
combination puts them on rays. Centroid `[−0.007, −0.039, 2.784]` versus `points.ply`'s
`[0.169, 0.348, −0.227]` — all views piled at the origin.

**This is by design, not a pipeline bug.** `GaussianModel.create_from_pcd()` under
`mono_d_so_enable=True` (used by `train_gs_init.py`) stores `_xy` (normalised image coords) and
`_z` (depth) plus the per-camera `c2w`, and `get_xyz` reconstructs world coordinates via
`torch.bmm(self.c2w, xyz_homo)`. `save_ply()` calls `get_xyz`, so the *saved* init cloud —
`<model_path>/point_cloud/iteration_*/point_cloud.ply` — **is** world-frame.

| stage | script | `mono_d_so` | `create_from_pcd` runs | outcome |
|---|---|---|---|---|
| 1a | `train_gs_init.py` | **True** | yes | camera-frame intentional; `get_xyz` applies `c2w`. Correct. |
| 1b | `train_gs.py` | False | **yes** | camera-frame `_xyz`, overwritten by `load_ply()` the next line. Correct. |
| 2 | `leave_one_out_stage{1,2}.py` | False | **yes** | as 1b. |
| eval | `scripts/render.py` | False | no (`load_ply=` passed into `Scene()`) | correct. |
| 5 | `threestudio/systems/...` | False | no (`load_ply(ply_path)`) | correct. |

Two things to be aware of:

1. **Correctness in 1b/2 rests on one unguarded line.** `Scene()` is called *without*
   `load_ply=`, so `create_from_pcd` runs on camera-frame points; `gaussians.load_ply(args.ply_path)`
   on the following line overwrites `_xyz` and every other tensor. `--ply_path` has no argparse
   default, so omitting it raises rather than failing silently — but guarding that call, or giving
   it a default, would silently train on camera-frame points.
2. **It is rebuilt constantly.** `create_from_pcd` allocates a ~2.4M-point cloud as `nn.Parameter`s
   and discards it one line later, and the reader deletes and rewrites `<scene>/point_cloud.ply`
   on every `Scene()` construction — including from stages 1b and 2.

`viz.py` no longer logs `<sfm_dir>/point_cloud.ply` as a world-frame cloud. To inspect the init
cloud, point it at the stage-1a output under `<model_path>/point_cloud/iteration_*/`.

---

## 6. Defect: cameras rendered facing backwards in `viz.py`

`viz.py:parse_transforms_json()` read `transform_matrix` straight into `rr.Transform3D`. The
comment `# OpenGL to OpenCV conversion for Rerun standard viewing if needed` was there; the
conversion was not. `rr.Pinhole` defaults to **RDF** (OpenCV), so an OpenGL pose was interpreted
as OpenCV: camera positions correct, orientations flipped 180° about x, and `rr.DepthImage`
unprojecting behind the camera.

Dot product of each camera's forward axis with the direction to its own pointmap centroid:

| view | before | after |
|---|---|---|
| image_2 | −0.9996 | +0.9996 |
| image_3 | −0.9963 | +0.9963 |
| image_4 | −0.9972 | +0.9972 |

Fixed by applying `c2w @ diag([1,-1,-1,1])` at load.

---

## 7. Defect: `fl_x` / `fl_y` paired with the wrong image axis

`readMipTransforms()` computed:

```python
FovY = focal2fov(focal_length_x, height)   # wrong
FovX = focal2fov(focal_length_y, width)    # wrong
```

Every consumer pairs `FovX`↔`width` and `FovY`↔`height` (the `K` build in the same file,
`GaussianModel.depth_densify`, `Camera.__init__`), and `readColmapCameras` in the *same file*
already did it correctly. A plain typo, and the only such site across both readers.

Invisible on every scene currently in the repo — all nine mip-NeRF 360 scenes and the MASt3R
output have `fl_x == fl_y`. It would skew geometry on any scene with non-square pixels. Verified
after the fix: square-pixel scenes unchanged (756.287/756.287 recovered exactly), and a synthetic
`fl_x=900, fl_y=600` scene now recovers `fx=900, fy=600` instead of the transpose.

Shared by the training reader and `threestudio/data/loo_mip.py`, so one fix covers stages 1–2 and 5.

---

## 8. GGPT: reconciling the VGGT pointmap with the RoMaV2 anchors

The `ggpt` backend has two sources of geometry that do not start out in the same frame:

- **VGGT's pointmap** — dense (every pixel of the 392×518 grid) but in VGGT's own world gauge.
- **DLT triangulation** — exact in the bundle-adjusted gauge, but only at pixels that survived
  RoMaV2 matching, track selection, and the epipolar / reprojection / parallax filters. On sceneA
  that is 54–71% of pixels per view.

`fuse_pointmaps()` never aligns the two clouds. It works in **per-view depth**, which is what the
pointmap grid actually stores, so the gauge problem reduces to a scalar:

1. camera-frame `z` of VGGT's points under VGGT's extrinsics — this is VGGT's own dense depth;
2. camera-frame `z` of the DLT points under the **BA** extrinsics;
3. a robust global scale `median(z_dlt / z_vggt)` over every anchor pixel;
4. `z_dlt` substituted wherever a triangulated depth exists — that measurement is strictly better;
5. unprojection with the BA intrinsics and poses.

Nothing is ever unprojected with mismatched poses, and no world-to-world transform is estimated.

`--fuse_mode` selects step 3:

| mode | what it corrects | when |
|---|---|---|
| `global_scale` (default) | the BA gauge only; VGGT's multi-view consistency is preserved exactly | per-view scales cluster tightly |
| `per_view_scale` | each view's own depth scale | one view is visibly mis-scaled |
| `per_view_affine` | a depth-*dependent* per-view bias — robust IRLS-Huber fit in disparity space, same construction as `utils/depth_align.py`, kept only when it beats a plain scale | anchors are dense and the far field is bending |

`global_scale` is the default because the correction it declines to make is usually tiny: on sceneA
the global scale was 1.0019 and the per-view scales 1.0043 / 0.9914 / 1.0137, a spread of ±1.4%.
Per-view modes trade multi-view consistency for per-view accuracy, which is the wrong trade when
the two already agree. All of these numbers land in `ggpt_sfm_stats.json`, so the decision can be
revisited per scene rather than guessed.

### Judging the solve

Ceres reports `Termination: No convergence` on sparse captures routinely — it hits the iteration
cap while still lowering the cost (0.1057 → 0.0491 px on sceneA), so it is a poor pass/fail signal.
GGPT's `run_sfm` also builds the bundle adjuster locally and drops it, so that report never reaches
the caller.

`ba_diagnostics()` measures the thing that matters instead: with the final poses and intrinsics,
does a DLT point reproject onto the pixel it was triangulated from? On sceneA that is p50 0.25 px /
p90 0.66 px against a `max_reproj_error` of 4. The worker warns when p90 exceeds that budget.

---

## 9. Operational notes

**Re-running SfM invalidates `depth_rel`.** Either backend's world gauge is arbitrary — a re-solve
can return a different rotation, translation *and scale*. Per-view depth is gauge-invariant up to
scale, so a re-solve that happens to land on the same scale leaves the old depth usable (observed:
1.3–2.6% agreement across one MASt3R re-solve), but nothing guarantees that and nothing currently
detects it. **Regenerate the depth prior after every SfM run.**

**Switching backends changes the gauge outright.** On sceneA the MASt3R solve spans depths of
0.75–19.8 and the GGPT solve 0.01–1.36 — roughly a 4× difference in scene scale. Each backend keeps
its artifacts inside its own `<backend>_sfm/` directory so they cannot be crossed by accident, but
any threshold expressed in world units (and `--depth_conf_thr`, see §4) has to be re-derived.

A re-solve may also **drop views** that fail to register, changing the effective `num_views` and
leaving `train_test_split_<N>.json` and `depth_rel/*_<N>.npy` inconsistent with the new solve.

**Depth is written to two locations.** `<scene>/depth_rel/` and `<image_dir>/depth_rel/`. The
reader walks a candidate list; `threestudio/data/loo_mip.py` historically hard-coded the
`image_dir` one (it now shares the same fallback chain). Both are written to keep them in sync.

**Scenes without pointmaps.** mip-NeRF 360 scenes ship their own `depth_rel` from the authors'
preprocessing. `setup_ri3d_scene_data()` uses it and errors if it is missing — it will **not**
synthesise a stand-in. Writing raw disparity or a constant into `depth_rel` is exactly the failure
mode this document exists to prevent, and it fails silently all the way through training.

---

## 10. Verification recipes

These are the checks that actually catch the defects above. Aggregate statistics do not. Express
results as a **fraction of scene scale**, not in world units — the two backends' gauges differ by
about 4× on the same scene, so an absolute threshold ports between neither.

**Grid alignment** — `pointmap_grid_is_aligned` must return `True` for every view, or
`--depth_source pointmap` silently falls back to ~20% coverage (§4). The GGPT worker runs this on
its own output and prints the result; `coverage 100.0%` in the depth-prep log is the downstream
confirmation.

**Poses and intrinsics unproject correctly** — unproject `depth_rel` with the reader's own `K`
construction and the `transforms.json` pose, compare against the pointmap's world XYZ:

```
p50 ≈ 0.0035 world units = 0.018% of scene scale   (mast3r, sceneA)
```

The residual is the nearest-upsampling grid offset (≈ 0.5 pointmap px × depth / focal), not a
pose error. Anything at the 0.1–1.0 range means a convention or staleness problem.

**Cameras face the scene** — `dot(c2w[:3,2], normalize(pointmap_centroid − c2w[:3,3]))` should be
close to **+1** for every view. A value near −1 is an OpenGL/OpenCV mix-up. Measured +0.9996 to
+0.9999 on sceneA under both backends, and under `ggpt` on a portrait-rotated copy of it.

**Init cloud quality** — per-point nearest-neighbour distance from the stage-1a cloud to
`points.ply`, normalised by the p1–p99 scene diagonal. Expect ≈ **0.05%** in `pointmap` mode
(sceneA: `mast3r` 0.054%, `ggpt` 0.050%). Do **not** rely on bounding-box extent ratio, which reads
~1.0 even when per-point error is 9× worse.

**Multi-view consistency** — reproject view *i*'s depth into view *j* and compare against view
*j*'s own depth, masking occlusions. Independent of the backend's own cloud, so it favours neither
depth source. sceneA p50: `mast3r` 0.45%, `ggpt` 0.17%.

**Anchor reprojection** (`ggpt` only) — `ggpt_sfm_stats.json → bundle_adjustment`. p50/p90 in
pixels against `dlt_config.max_reproj_error`. This, not Ceres's termination string, is the signal
that the bundle adjustment settled.

**Disparity leakage** — `_check_metric_depth()` warns on load. To check manually, a median
`depth_rel` value in the tens or higher on these scenes means disparity, not metric depth. Note
the guard's threshold (median > 20) is absolute, so it is a loose check on a small-gauge solve:
sceneA's `ggpt` medians are 0.71–0.89, three orders below it.

---

## 11. Relevant code

| file | role |
|---|---|
| `utils/pointmap_utils.py` | load pointmaps/poses/intrinsics; `resample_view_depth` (dense), `project_pointmap` (splat), grid-alignment guard, hole fill |
| `utils/depth_align.py` | robust affine + piecewise fit of monocular disparity onto pointmap anchors |
| `inference.py` | `resolve_train_view_names`, `_depth_from_pointmap`, `generate_aligned_depth`, `--depth_source`, `--sfm_backend` |
| `scripts/run_sfm.py` | backend routing; builds the worker command line |
| `tools/detect_watermark.py` | cross-image edge persistence -> watermark mask + preview |
| `ggpt_sfm/run_ggpt.py` | GGPT backend: VGGT + RoMaV2 + BA + DLT, `fuse_pointmaps`, `ba_diagnostics`, contract writer, self-check |
| `mast3r/run_mast3r.py` | MASt3R backend |
| `scene/dataset_readers_flow.py` | `readMipTransforms`, `_check_metric_depth`, init point-cloud back-projection, confidence gate |
| `threestudio/data/loo_mip.py` | stage-5 depth loading, `min_depth`/`max_depth` |
| `viz.py` | Rerun visualisation; convention flip, world-frame cloud selection |
