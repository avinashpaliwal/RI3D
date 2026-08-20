# RI3D Pipeline

Reconstructs a 3D Gaussian Splatting scene from sparse input views (typically 3).

With only 3 photographs, 3DGS has no supervision for any viewpoint other than the inputs, so it
produces artifacts where geometry is weakly constrained and nothing at all where no camera looked.
RI3D's answer is to train **two scene-specific diffusion priors** and use them to manufacture
pseudo-ground-truth for viewpoints that were never photographed:

- a **Repair** prior (ControlNet + LoRA) that turns an artifacted render of *this scene* into a clean image;
- an **Inpainting** prior (fine-tuned SD2-inpainting) that hallucinates *unobserved* regions in this scene's appearance.

The pipeline falls into three phases: build geometry (0–1), build the priors (2–4), then optimize
against them (5).

Depth-prior details, coordinate conventions and a set of related defects are documented separately
in [`point_map.md`](point_map.md).

---

## Stage reference

`$SCENE` is the scene name, `$N` the number of sparse views.

| stage | script / config | output |
|---|---|---|
| `sfm` | `scripts/run_sfm.py` → `mast3r/run_mast3r.py` | `<out>/mast3r_sfm/` |
| — | `inference.py` (depth prep) | `<scene>/depth_rel/`, `depths$N.npy`, `confs$N.npy` |
| `1a` | `scripts/train_gs_init.py` | `debug/gs_init/${SCENE}_${N}` |
| `1b` | `scripts/train_gs.py` | `output/gs_init/${SCENE}_${N}` |
| `2a` `2b` | `scripts/leave_one_out_stage{1,2}.py` | `output/gs_init/${SCENE}_loo_${N}` |
| `3` | `scripts/train_lora.py` | `output/controlnet_finetune/${SCENE}_${N}` |
| `4` | `train_realfill.py` (external) | `inpainting/${SCENE}_${N}` |
| `5a` | `scripts/train_repair.py` + `configs/gaussian-object.yaml` | `output_den${N}/gaussian_object/${SCENE}_${N}` |
| `5b` | `scripts/train_repair.py` + `configs/gaussian-object_inp.yaml` | `output_inp${N}/gaussian_object/${SCENE}_${N}` |
| `render` | `scripts/render.py` | orbit video + exported PLY |

---

## Phase 1 — Geometry

### Stage 0 · MASt3R-SfM

Recovers camera intrinsics, extrinsics and dense per-view pointmaps from **unposed** images.
Writes `cameras.json` (OpenCV c2w), COLMAP `sparse/0`, `pointmaps/*.json` (dense per-view world XYZ
+ confidence) and `points.ply` (fused, confidence-filtered). `inference.py` then converts the COLMAP
model to `transforms.json`.

Configs: `configs/mast3r/unposed.yaml` (solve everything) and `posed.yaml` (fix intrinsics/extrinsics
to a known calibration).

> This stage is specific to this fork. Upstream RI3D assumes COLMAP poses already exist, which is
> why the mip-NeRF 360 scenes ship their own `transforms.json` and `depth_rel/`.

### Depth preparation

Produces the metric depth prior every later stage reads. Controlled by `--depth_source`:

- **`pointmap`** (default) — resample MASt3R's dense pointmap depth onto the target grid.
- **`align`** — run Depth-Anything and fit its affine-invariant disparity onto the pointmap anchors.

Writes `depth_rel/inpv2<name>_<N>.npy` and `inp_dust3r<name>_<N>.npy` (metric z-depth in SfM world
units), plus `depths<N>.npy` / `confs<N>.npy`. See [`point_map.md`](point_map.md) §4 for the
measured comparison — `pointmap` gives a ~9× more accurate initial cloud.

### Stage 1a · Gaussian initialization

Back-projects the depth prior into per-view points and builds the initial `GaussianModel`.

**Despite the name this stage does not train.** It saves `iteration_1` and exits immediately —
there is a hard `exit()` in `scripts/train_gs_init.py` right after `scene.save(first_iter)`. Its
entire purpose is to produce the initial PLY.

This is also the one place in the pipeline where a coordinate-frame change happens: the reader hands
`create_from_pcd` **camera-frame** points, and `GaussianModel.get_xyz` transforms them to world via
`torch.bmm(self.c2w, xyz_homo)` (the `mono_d_so_enable=True` path). Details in
[`point_map.md`](point_map.md) §5.

### Stage 1b · Base 3DGS

Loads the stage-1a PLY and trains ~10k iterations on the sparse views with L1 + SSIM + a monocular
depth loss. This is the baseline reconstruction: accurate where observed, artifacted or empty
elsewhere.

Note `OptimizationParams.iterations` is **10_000** in this repo (`utils/arguments.py`), lowered from
upstream 3DGS's 30_000.

---

## Phase 2 — Scene-specific priors

### Stage 2 · Leave-one-out data generation

For each of the `$N` sparse views, train a Gaussian model that **excludes** that view, then render
the held-out viewpoint. The render shows exactly what this scene looks like when reconstructed
without supervision, and the withheld photograph is the matching clean target — a supervised pair
for artifact repair, generated from the scene itself.

`leave_one_out_stage1.py` trains from the init; `stage2.py` resumes with reset learning rates,
giving degradation at two severity levels.

This stage propagates **no point cloud forward** — its product is the set of LOO Gaussian models.

### Stage 3 · Repair prior (LoRA)

Fine-tunes ControlNet Tile + SD 1.5 with LoRA on those pairs. `utils/dataset_lora.py:GSCacheDataset`
renders the LOO models on the fly and injects additional noise (scale/dropout, annealed over
training) to widen the degradation distribution beyond what leave-one-out alone produces.

The result maps *artifacted render of this scene* → *clean image of this scene*, keyed to the rare
token `xxy5syt00`.

### Stage 4 · Inpainting prior (RealFill)

Fine-tunes SD2-inpainting on the sparse views so it can fill **unobserved** regions in this scene's
specific appearance. Repair fixes what was reconstructed badly; inpainting invents what was never
seen at all.

Requires the external `train_realfill.py`. If absent, `inference.py` falls back to the base SD2
inpainting checkpoint.

---

## Phase 3 — Prior-guided optimization

Both stages run through `scripts/train_repair.py` on the threestudio / PyTorch Lightning stack.

### Stage 5a · Repair + densification

Loads the **stage-1a** initialization (not 1b), renders novel viewpoints, passes each through the
Repair model to produce pseudo-ground-truth, and optimizes the Gaussians against it — periodically
resampling the novel-view pool as the model improves.

The refresh is driven from the data module (`threestudio/data/loo_mip.py`): `data.refresh_size` sets
how many novel viewpoints are held in the pool, `data.refresh_interval` (default 100) how often it
is resampled. `inference.py` also passes `system.refresh_size`, but that key is only read in
commented-out code — `data.refresh_size` is the one that takes effect.

### Stage 5b · Inpainting refinement

Chains from 5a's output and fills unobserved regions using the inpainting prior. This is where the
scene grows beyond what the cameras saw.

The two configs differ mainly in learning rates and densification parameters.

### Export

`inference.py` copies the best available PLY to `<output_dir>/<scene>_reconstructed_3dgs.ply`,
preferring 5b → 5a → the highest-iteration stage-1b checkpoint.

---

## Dependency graph

```
0 SfM ──► depth prep ──► 1a init ──┬──► 1b base 3DGS ──┐
                                   │                    ├──► 3 LoRA (Repair) ──┐
                                   ├──► 2 leave-one-out ┘                      │
                                   │                                            ├──► 5a repair
                                   └────────────────────────────────────────────┘        │
                     4 RealFill (inpainting) ──────────────────────────────────────► 5b refine ──► export
```

Two non-obvious edges:

- **1b, 2 and 5a all branch off 1a independently.** Only 5b chains from its predecessor. A bad
  initialization therefore propagates into three separate places rather than one.
- **Stage 1b's trained model is not consumed by 5a.** It feeds stage 3 (via `--gs_dir`) and serves
  as the evaluation baseline.

### Point cloud through the stages

Measured on sceneA (3 views, `--depth_source align`). Every stage after 1a is world-frame and applies
no further transformation:

| stage | gaussians | centroid | p1–p99 extent |
|---|---|---|---|
| 0 `points.ply` | 461,678 | `[0.15, 0.18, -0.11]` | — |
| 1a init | 2,359,296 | `[0.15, 0.08, -0.12]` | `[4.28, 2.58, 3.08]` |
| 1b base | 2,275,803 | `[0.14, 0.08, -0.12]` | `[4.29, 2.58, 3.09]` |
| 5a repair | 1,723,972 | `[0.14, 0.10, -0.10]` | `[4.31, 2.57, 3.11]` |
| 5b inpaint | 5,165,292 | `[-0.26, 2.00, 0.11]` | `[5.47, 6.01, 3.55]` |

The 5b jump (+3.44M gaussians, extent growing mostly in +y) is the inpainting prior adding geometry
where no view observed — the whole point of phases 2–3.

---

## Running it

End to end from raw photos:

```bash
python inference.py -i /path/to/photos -o output/my_scene \
    --sfm_config unposed --num_views 3
```

Pre-calibrated scene (`transforms.json` already present):

```bash
python inference.py -i data/mipnerf360/bicycle -o output/bicycle_3 \
    --sfm_config posed --num_views 3
```

Selected stages — comma-separated; `1` expands to `1a,1b`, `2` to `2a,2b`, `5` to `5a,5b`:

```bash
python inference.py -i <scene> --stages sfm,1,2,3,5a,5b -n 3
python inference.py -i <scene> --dry_run          # print commands without running
```

Alternatives: `bash scripts/run.sh <scene> <num_views>` (stages commented out — uncomment what you
need), `python scripts/run_parallel_mip.py` to distribute mip-NeRF 360 scenes across GPUs, and
`bash scripts/eval.sh <scene> <num_views>` for metrics.

### Geometry-only runs

Stages 0 + 1a + 1b produce a complete, renderable 3DGS model without any diffusion prior:

```bash
python inference.py -i /path/to/photos -o output/my_scene \
    --sfm_config unposed --stages sfm,1 -n 3
```

Result: `output/gs_init/<scene>_<N>/point_cloud/iteration_10000/point_cloud.ply` — a standard 3DGS
PLY any Gaussian viewer opens. Useful as a fast baseline, for sanity-checking SfM and depth, or for
A/B-ing `--depth_source`. It will be accurate where the cameras looked and empty elsewhere; on
sceneA that is 2.28M gaussians versus 5.17M after the full pipeline.

Use `--stages sfm,1` rather than `--stages 1` for raw photos — `1` alone skips SfM and assumes
`transforms.json` and the depth prior already exist.

---

## Key parameters

| flag | default | notes |
|---|---|---|
| `-n / --num_views` | 3 | number of sparse training views |
| `-r / --resolution` | 4 | downsample factor; must be consistent across all stages |
| `--sh_degree` | 2 | spherical-harmonics degree |
| `--depth_source` | `pointmap` | see [`point_map.md`](point_map.md) §4 |
| `--sfm_config` | `posed` | `unposed` runs MASt3R-SfM |
| `--prompt` | `xxy5syt00` | rare token for the Repair LoRA; must match across stages 3 and 5 |

Required checkpoints live in `models/` (gitignored): SD 1.5, ControlNet Tile v1.1, SD2-inpainting —
fetch with `python scripts/download_hf_models.py`. MASt3R weights are downloaded separately;
`inference.py` warns about anything missing but continues.

---

## Gotchas

**Re-running SfM invalidates the depth prior.** MASt3R's world gauge is arbitrary and a re-solve can
return a different rotation, translation *and scale*. A re-solve may also drop views that fail to
register, leaving `train_test_split_<N>.json` and `depth_rel/*_<N>.npy` inconsistent with the new
solve. Regenerate depth after every SfM run.

**`<scene>/point_cloud.ply` is not a world-frame model.** The dataset reader writes it as an
intermediate on every `Scene()` construction and its points are in each view's camera frame. Point
viewers at `points.ply` or at a stage output instead — see [`point_map.md`](point_map.md) §5.

**Stage 5 needs stages 3 and 4 to have run**, since it loads the LoRA checkpoint
(`system.exp_name`) and the inpainting model (`system.inpainting_dir`).

**Iteration counts are not fixed.** `--iterations` defaults to 10_000 here, so checkpoint
directories are `iteration_10000`, not the upstream `iteration_30000`. Code that locates a trained
model must scan `point_cloud/iteration_*` and take the highest rather than hardcoding a number.
