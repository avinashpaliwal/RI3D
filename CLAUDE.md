# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RI3D reconstructs novel views from sparse input images (e.g., 3 views) using 3D Gaussian Splatting with two diffusion priors: a **Repair model** (ControlNet+LoRA that enhances renders into pseudo ground truth) and an **Inpainting model** (fine-tuned SD inpainting for unobserved regions). Published at ICCV 2025.

## Environment Setup

Requires Python 3.10, CUDA 11.8, and `uv` for dependency management:
```bash
uv venv --python 3.10 && source .venv/bin/activate
uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt --no-build-isolation
python download_hf_models.py  # downloads SD v1.5 + ControlNet Tile to models/
```

The `--no-build-isolation` flag is required because `requirements.txt` includes local submodules (`simple-knn`, `minLoRA`, `CLIP`, `pytorch3d`) that build CUDA extensions.

## Pipeline Stages

The pipeline runs sequentially per scene. Key variables: `$SCENE` (e.g., `bicycle`) and `$NUM_VIEW` (e.g., `3`).

| Stage | Script | Output Directory |
|-------|--------|-----------------|
| 1. Gaussian init | `train_gs_init.py` | `debug/gs_init/${SCENE}_${NUM_VIEW}` |
| 1b. GS training | `train_gs.py` | `output/gs_init/${SCENE}_${NUM_VIEW}` |
| 2. Leave-one-out data gen | `leave_one_out_stage1.py`, `leave_one_out_stage2.py` | `output/gs_init/${SCENE}_loo_${NUM_VIEW}` |
| 3. LoRA fine-tune (repair) | `train_lora.py` | `output/controlnet_finetune/${SCENE}_${NUM_VIEW}` |
| 4. Inpainting fine-tune | External (RealFill) | `inpainting/${SCENE}_${NUM_VIEW}` |
| 5a. Repair optimization | `train_repair.py` + `gaussian-object.yaml` | `output_den${NUM_VIEW}/gaussian_object/${SCENE}_${NUM_VIEW}` |
| 5b. Inpainting refinement | `train_repair.py` + `gaussian-object_inp.yaml` | `output_inp${NUM_VIEW}/gaussian_object/${SCENE}_${NUM_VIEW}` |

Run a single scene: `bash run.sh <scene> <num_views>` (uncomment desired stages).
Run all mip-NeRF 360 scenes in parallel across GPUs: `python run_parallel_mip.py`.
Evaluate: `bash eval.sh <scene> <num_views>` or `python run_parallel_mip_eval.py`.

## Architecture

### Two independent frameworks share this repo

1. **3DGS framework** (stages 1-2, eval): Based on the original 3D Gaussian Splatting codebase.
   - `scene/` — scene loading (COLMAP), camera models, `GaussianModel` (the core splat representation)
   - `gaussian_renderer/` — differentiable rasterization (uses `gsplat`)
   - `arguments/` — `ModelParams`, `OptimizationParams`, `PipelineParams` (argparse-based config)
   - `utils/` — loss functions, camera utilities, depth utilities, graphics math
   - Training scripts (`train_gs_init.py`, `train_gs.py`, `leave_one_out_*.py`) use argparse + these modules directly

2. **threestudio framework** (stage 5): PyTorch Lightning-based system for diffusion-guided optimization.
   - `threestudio/systems/` — `BaseSystem` (PL LightningModule); the Gaussian object system lives here
   - `threestudio/data/loo_mip.py` — data module that loads leave-one-out + sparse views for guided optimization
   - `threestudio/utils/config.py` — OmegaConf-based config loading from YAML files in `configs/`
   - `threestudio/__init__.py` — module registry (`@register` decorator pattern); `find(name)` resolves registered classes
   - `train_repair.py` is the entry point; it loads config, then uses PL `Trainer`

### Diffusion components (stage 3)

- `cldm/` — ControlNet implementation: `ControlNet`, `ControlledUnetModel`, `ControlLDM` (extends `LatentDiffusion`)
- `ldm/` — Stable Diffusion backbone (latent diffusion, DDIM/DDPM/DPM samplers, autoencoder, attention)
- `train_lora.py` — fine-tunes ControlNet Tile + SD with LoRA (uses `minlora` from submodules)
- `dataset_lora.py` — `GSCacheDataset` renders GS views on-the-fly as training data for LoRA

### Other components

- `depth_anything/` — monocular depth estimation network (DPT architecture)
- `pred_monodepth.py` — predicts monocular depth for scene initialization
- `visual_hull.py` — computes visual hull from sparse views for point cloud initialization
- `sparse_pc.py` — COLMAP database creation for sparse point clouds
- `depth_layering.py` — depth-based scene layering via agglomerative clustering

## Key Conventions

- The placeholder prompt `xxy5syt00` is used as a rare-token identifier for textual inversion-style personalization throughout the pipeline. It appears in configs, training scripts, and CLI args.
- Configs in `configs/` use OmegaConf variable interpolation (e.g., `${trainer.max_steps}`).
- `gaussian-object.yaml` = repair/densification stage; `gaussian-object_inp.yaml` = inpainting refinement stage. They differ mainly in learning rates and densification parameters.
- Pre-trained model weights go in `models/` (gitignored). The YAML config for ControlNet Tile is version-controlled at `models/control_v11f1e_sd15_tile.yaml`.
- Dataset expected at `data/mipnerf360/` with COLMAP sparse reconstruction format.
- The `run_parallel*.py` scripts distribute scenes across multiple GPUs via `CUDA_VISIBLE_DEVICES` and subprocess management.
