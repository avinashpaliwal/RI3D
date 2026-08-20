# MASt3R Pointmaps, Depth Priors, and Coordinate Conventions

This document covers how MASt3R-SfM's pointmaps become the metric depth prior RI3D consumes,
the coordinate conventions involved, and a set of defects that were fixed along the way. Several
of those defects were invisible at runtime — the pipeline ran to completion and produced
plausible-looking numbers while the geometry was wrong — so the symptoms and the checks that
expose them are recorded here as well.

---

## 1. The two modalities

MASt3R-SfM writes, per scene, into `<scene>/mast3r_sfm/`:

| artifact | contents | frame |
|---|---|---|
| `cameras.json` | `filepaths`, `focals`, `cams2world` | **OpenCV** c2w |
| `sparse/0/*.bin` | COLMAP cameras/images/points3D | **OpenCV** w2c |
| `pointmaps/<name>.json` | dense per-view world XYZ (flat) + `confs` (2-D) | world |
| `points.ply` | confidence-filtered fused point cloud | world |
| `transforms.json` | written by `inference.py`, NeRF-style | **OpenGL** c2w |

The pointmap JSON stores `points` **flattened** and `confs` as a 2-D array — the conf array
carries the grid shape, so reshape `points` using `confs.shape`.

RI3D consumes a separate prior:

| artifact | consumed by | expected contents |
|---|---|---|
| `depth_rel/inpv2<name>_<N>.npy` | `scene/dataset_readers_flow.py`, `threestudio/data/loo_mip.py` | **metric z-depth, SfM world units** |
| `depth_rel/inp_dust3r<name>_<N>.npy` | same (init point cloud path) | same |
| `depths<N>.npy` | `GaussianModel.flows` | stacked per-view depth |
| `confs<N>.npy` | `GaussianModel.masks` | stacked per-view MASt3R confidence |

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

The catch: the two supported scene types disagree about what those fields mean.

- **mip-NeRF 360** — `transforms.json` stores the ÷4 (`images_4`) dimensions.
- **MASt3R output** — `convert_colmap_to_transforms_json` stores full-resolution dimensions.

**Any code writing a depth map must take its target size from `transforms.json`'s `w`/`h`, never
from the image file on disk.** `utils/pointmap_utils.load_target_intrinsics()` is the single place
that reads it.

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

**`pointmap`** — resample MASt3R's dense pointmap depth onto the target grid.
Metric and multi-view consistent by construction: it is the same geometry the poses came from.

**`align`** — run Depth-Anything and fit its disparity onto the pointmap's metric anchors
(`utils/depth_align.py`: robust IRLS-Huber affine in disparity space, then an optional monotone
piecewise refinement accepted only when it lowers the residual). Keeps the monocular network's
fine detail.

### Why `pointmap` is the default

The pointmap is a **dense depth image on its own 512×384 grid**, and that grid maps linearly onto
the full image (verified: 99th-percentile deviation < 1 px, implied principal-point offset
0.002–0.019 px). The original implementation *splatted* it as individual 3D points into the
1024×768 target grid, which at 2× upsampling covers only ~20% of pixels and leaves the monocular
prior to invent the other 80%.

The global affine fit cannot carry that load. Its residual has a systematic depth-dependent
component and very heavy tails:

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

### Low-confidence and invalid pixels

`pointmap` mode takes MASt3R depth at every pixel regardless of confidence. Two consequences:

- MASt3R occasionally places a pixel **behind** the camera (`z <= 0`) — rare (0.008% on sceneA)
  and low-confidence (conf median 0.61 vs 5.84 overall), but a negative value in `depth_rel`
  back-projects behind the view. `_depth_from_pointmap()` hole-fills these. Note the disparity
  guard cannot catch them: it filters to `z > 0` before taking its median.
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

## 8. Operational notes

**Re-running SfM invalidates `depth_rel`.** MASt3R's world gauge is arbitrary — a re-solve can
return a different rotation, translation *and scale*. Per-view depth is gauge-invariant up to
scale, so a re-solve that happens to land on the same scale leaves the old depth usable (observed:
1.3–2.6% agreement across one re-solve), but nothing guarantees that and nothing currently detects
it. **Regenerate the depth prior after every SfM run.**

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

## 9. Verification recipes

These are the checks that actually catch the defects above. Aggregate statistics do not.

**Poses and intrinsics unproject correctly** — unproject `depth_rel` with the reader's own `K`
construction and the `transforms.json` pose, compare against the pointmap's world XYZ:

```
p50 ≈ 0.0035 world units = 0.018% of scene scale
```

The residual is the nearest-upsampling grid offset (≈ 0.5 pointmap px × depth / focal), not a
pose error. Anything at the 0.1–1.0 range means a convention or staleness problem.

**Cameras face the scene** — `dot(c2w[:3,2], normalize(pointmap_centroid − c2w[:3,3]))` should be
close to **+1** for every view. A value near −1 is an OpenGL/OpenCV mix-up.

**Init cloud quality** — per-point nearest-neighbour distance from the stage-1a cloud to
`points.ply`. Expect p50 ≈ 0.003 in `pointmap` mode. Do **not** rely on bounding-box extent ratio,
which reads ~1.0 even when per-point error is 9× worse.

**Multi-view consistency** — reproject view *i*'s depth into view *j* and compare against view
*j*'s own depth, masking occlusions. Independent of MASt3R's own cloud, so it does not favour
either depth source.

**Disparity leakage** — `_check_metric_depth()` warns on load. To check manually, a median
`depth_rel` value in the tens or higher on these scenes means disparity, not metric depth.

---

## 10. Relevant code

| file | role |
|---|---|
| `utils/pointmap_utils.py` | load pointmaps/poses/intrinsics; `resample_view_depth` (dense), `project_pointmap` (splat), grid-alignment guard, hole fill |
| `utils/depth_align.py` | robust affine + piecewise fit of monocular disparity onto pointmap anchors |
| `inference.py` | `resolve_train_view_names`, `_depth_from_pointmap`, `generate_aligned_depth`, `--depth_source` |
| `scene/dataset_readers_flow.py` | `readMipTransforms`, `_check_metric_depth`, init point-cloud back-projection, confidence gate |
| `threestudio/data/loo_mip.py` | stage-5 depth loading, `min_depth`/`max_depth` |
| `viz.py` | Rerun visualisation; convention flip, world-frame cloud selection |
