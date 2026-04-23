# RI3D

> **RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors**  
> [Avinash Paliwal](http://avinashpaliwal.com/),
> [Xilong Zhou](https://xilongzhou.github.io/), 
> [Wei Ye](https://ywwwer.github.io/), 
> [Jinhui Xiong](https://jhxiong.github.io/), 
> [Rakesh Ranjan](https://scholar.google.co.in/citations?user=8KF99lYAAAAJ&hl=en), 
> [Nima Khademi Kalantari](https://people.engr.tamu.edu/nimak/index.html)  
> **ICCV 2025**

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2503.10860)
[![Project Page](https://img.shields.io/badge/RI3D-Website-blue?logo=googlechrome&logoColor=blue)](https://people.engr.tamu.edu/nimak/Papers/RI3D/index.html)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=diva4AN7f3k)

<p align="center">
  <img src="assets/teaser.gif?raw=true" alt="RI3D teaser" width="100%">
</p>

## Overview

RI3D reconstructs high-quality novel views from sparse input images using 3D Gaussian Splatting with two specialized diffusion priors:

- **Repair model**: Enhances rendered images of visible regions into pseudo ground truth via a fine-tuned ControlNet with LoRA.
- **Inpainting model**: Hallucinations for missing/unobserved regions via a fine-tuned Stable Diffusion inpainting model.

A two-stage optimization first reconstructs visible areas, then fills in missing regions while maintaining multi-view coherence.

## Installation

Tested on Ubuntu with NVIDIA 2080 Ti (CUDA 11.8) and Python 3.10.

### 1. Create environment

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

### 2. Download pre-trained models

```bash
python scripts/download_hf_models.py
```

This downloads Stable Diffusion v1.5, ControlNet Tile, and SD2 Inpainting to `models/`.

## Dataset

Download the preprocessed sparse mip-NeRF 360 dataset (based on ReconFusion splits):

[Google Drive](https://drive.google.com/file/d/1CS1pv2uQQhuX92QAYfXVwZC6-imGTunq/view?usp=sharing)

Extract it to `data/mipnerf360/`.

## Evaluation

### Pretrained Models

Download pretrained Gaussian Splatting models (mip-NeRF 360, all 9 scenes):

| Views | Link | Size |
|-------|------|------|
| 3 | [Google Drive](https://drive.google.com/open?id=1j3T6DuuMU5o7qt3UpWkd9TLJgSbYYi1z) | 6.4 GB |
| 6 | [Google Drive](https://drive.google.com/open?id=1M981nzmCoWS_5tf4sK3hnHr2ZPb4-K3D) | 6.1 GB |
| 9 | [Google Drive](https://drive.google.com/open?id=12ZIZEDyiqd21ZnYLw7BPaoUzzPy9LH-B) | 5.1 GB |

Each archive contains `stage1.ply` (repair) and `stage2.ply` (repair + inpainting) per scene. Extract to the project root:

```bash
tar -xzf ri3d_mipnerf360_3views.tar.gz
```

### Render Videos

```bash
# Render novel view video (stage 1 = repair, stage 2 = inpainting)
bash scripts/eval.sh $SCENE $NUM_VIEW $STAGE

# Example: bicycle, 3 views, final (stage 2) result
bash scripts/eval.sh bicycle 3 2
```

### Compute Metrics

```bash
python scripts/render.py \
    -m output/gs_init/${SCENE}_${NUM_VIEW} \
    --sparse_view_num $NUM_VIEW --sh_degree 2 \
    --white_background \
    --postfix _stage2 \
    --load_ply ${NUM_VIEW}_views/${SCENE}/stage2.ply
```

## Training

The full pipeline consists of the following stages. See `scripts/run.sh` and `scripts/run_parallel_mip.py` for running the full pipeline across all scenes.

### Stage 1: Gaussian Initialization

```bash
# 1a: Initialize point cloud from sparse views
python scripts/train_gs_init.py -s data/mipnerf360/$SCENE \
    -m debug/gs_init/${SCENE}_${NUM_VIEW} \
    -r 4 --sparse_view_num $NUM_VIEW --sh_degree 2 \
    --white_background --random_background

# 1b: Train Gaussians from initialized point cloud
python scripts/train_gs.py -s data/mipnerf360/$SCENE \
    -m output/gs_init/${SCENE}_${NUM_VIEW} \
    -r 4 --sparse_view_num $NUM_VIEW --sh_degree 2 \
    --white_background --random_background \
    --ply_path debug/gs_init/${SCENE}_${NUM_VIEW}/point_cloud/iteration_1/point_cloud.ply
```

### Stage 2: Leave-One-Out Data Generation

```bash
# Stage 2a
python scripts/leave_one_out_stage1.py -s data/mipnerf360/$SCENE \
    -m output/gs_init/${SCENE}_loo_${NUM_VIEW} \
    -r 4 --sparse_view_num $NUM_VIEW --sh_degree 2 \
    --white_background --random_background \
    --ply_path debug/gs_init/${SCENE}_${NUM_VIEW}/point_cloud/iteration_1/point_cloud.ply

# Stage 2b
python scripts/leave_one_out_stage2.py -s data/mipnerf360/$SCENE \
    -m output/gs_init/${SCENE}_loo_${NUM_VIEW} \
    -r 4 --sparse_view_num $NUM_VIEW --sh_degree 2 \
    --white_background --random_background \
    --ply_path debug/gs_init/${SCENE}_${NUM_VIEW}/point_cloud/iteration_1/point_cloud.ply
```

### Stage 3: LoRA Fine-tuning (Repair Model)

```bash
python scripts/train_lora.py --exp_name controlnet_finetune/${SCENE}_${NUM_VIEW} \
    --prompt xxy5syt00 --sh_degree 2 --resolution 4 --sparse_num $NUM_VIEW \
    --data_dir data/mipnerf360/$SCENE \
    --gs_dir output/gs_init/${SCENE}_${NUM_VIEW} \
    --loo_dir output/gs_init/${SCENE}_loo_${NUM_VIEW} \
    --bg_white --sd_locked --train_lora \
    --add_diffusion_lora --add_control_lora --add_clip_lora
```

### Stage 4: Inpainting Model Fine-tuning

We use [RealFill](https://github.com/thuanz123/realfill) to fine-tune a Stable Diffusion inpainting model on the sparse input views. Please refer to their repository for setup instructions.

```bash
python train_realfill.py \
    --pretrained_model_name_or_path=models/stable-diffusion-2-inpainting \
    --train_data_dir=data/mipnerf360/$SCENE/$NUM_VIEW \
    --output_dir=inpainting/${SCENE}_${NUM_VIEW} \
    --resolution=512 --train_batch_size=16 \
    --gradient_accumulation_steps=1 \
    --unet_learning_rate=2e-4 --text_encoder_learning_rate=4e-5 \
    --lr_scheduler=constant --lr_warmup_steps=100 \
    --max_train_steps=2000 \
    --lora_rank=8 --lora_dropout=0.1 --lora_alpha=16
```

### Stage 5: Repair + Inpainting Optimization

```bash
# Stage 5a: Repair (densification)
python scripts/train_repair.py \
    --config configs/gaussian-object.yaml \
    --train --gpu 0 \
    tag="${SCENE}_${NUM_VIEW}" \
    exp_root_dir="output_den${NUM_VIEW}" \
    system.init_dreamer="debug/gs_init/${SCENE}_${NUM_VIEW}" \
    system.exp_name="output/controlnet_finetune/${SCENE}_${NUM_VIEW}" \
    system.refresh_size=8 \
    data.data_dir="data/mipnerf360/$SCENE" \
    data.resolution=4 \
    data.sparse_num=$NUM_VIEW \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=8 \
    system.sh_degree=2

# Stage 5b: Inpainting refinement
python scripts/train_repair.py \
    --config configs/gaussian-object_inp.yaml \
    --train --gpu 0 \
    tag="${SCENE}_${NUM_VIEW}" \
    exp_root_dir="output_inp${NUM_VIEW}" \
    system.init_dreamer="output_den${NUM_VIEW}/gaussian_object/${SCENE}_${NUM_VIEW}" \
    system.exp_name="output/controlnet_finetune/${SCENE}_${NUM_VIEW}" \
    system.inpainting_dir="inpainting" \
    system.refresh_size=10 \
    data.data_dir="data/mipnerf360/$SCENE" \
    data.resolution=4 \
    data.sparse_num=$NUM_VIEW \
    data.prompt="a photo of a xxy5syt00" \
    data.refresh_size=10 \
    system.sh_degree=2
```

## Acknowledgements

This project builds on several excellent works:

- [GaussianObject](https://github.com/GaussianObject/GaussianObject) (Yang et al., SIGGRAPH Asia 2024)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) (Kerbl et al., SIGGRAPH 2023)
- [ControlNet](https://github.com/lllyasviel/ControlNet) (Zhang et al.)
- [RealFill](https://github.com/thuanz123/realfill) (Tang et al., SIGGRAPH 2024)
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) (Rombach et al.)

## Citation

```bibtex
@inproceedings{paliwal2025ri3d,
    title={RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors},
    author={Avinash Paliwal and Xilong Zhou and Wei Ye and Jinhui Xiong and Rakesh Ranjan and Nima Khademi Kalantari},
    journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    year={2025}
}
```
