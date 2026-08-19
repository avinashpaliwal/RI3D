# RI3D (Enhanced & Automated Fork)

This repository is an enhanced, standalone fork of **RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors** (ICCV 2025).

> **Original Paper:**  
> **RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors**  
> [Avinash Paliwal](http://avinashpaliwal.com/), [Xilong Zhou](https://xilongzhou.github.io/), [Wei Ye](https://ywwwer.github.io/), [Jinhui Xiong](https://jhxiong.github.io/), [Rakesh Ranjan](https://scholar.google.co.in/citations?user=8KF99lYAAAAJ&hl=en), [Nima Khademi Kalantari](https://people.engr.tamu.edu/nimak/index.html)  
> *IEEE/CVF International Conference on Computer Vision (ICCV) 2025*  
> [![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2503.10860)
> [![Project Page](https://img.shields.io/badge/RI3D-Website-blue?logo=googlechrome&logoColor=blue)](https://people.engr.tamu.edu/nimak/Papers/RI3D/index.html)
> [![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=diva4AN7f3k)

---

## Key Enhancements in This Fork

This fork modernizes the codebase, eliminates external preprocessing friction, and provides end-to-end automation and interactive visualization:

1. **Unposed In-the-Wild Reconstruction (MASt3R-SfM Integration)**:
   - Integrated [MASt3R](https://github.com/naver/mast3r) / DUSt3R for automated camera pose estimation, global coordinate alignment, and dense pointmap extraction directly from raw, unposed sparse images.
   - Eliminates mandatory COLMAP failure points on few-view captures.

2. **Unified End-to-End CLI (`inference.py`)**:
   - Single command orchestrates the entire pipeline: SfM extraction → Stage 1a/1b Gaussian initialization & base 3DGS → Stage 2 Leave-One-Out noise perturbation modeling → Stage 3 ControlNet LoRA training → Stage 4/5 Dual-Prior optimization → novel-view rendering.
   - Built-in stage resumption (`--stages 5a,5b,render`) and automatic detection of prior outputs.

3. **Interactive 3D / 2D Rerun Visualizer (`viz.py`)**:
   - Real-time web viewer server (`--serve --port 9876`) and recording exporter (`--save`).
   - Visualizes camera frustums, input views, depth maps, MASt3R pointmaps, Leave-One-Out parameter distributions, and 3D Gaussian evolution.
   - Supports timeline scrubbing (`--timeline`) across all reconstruction stages.

4. **Modern PyTorch 2.6+ & SDPA Compatibility**:
   - Full compatibility with PyTorch 2.6+ `weights_only=False` deserialization across all checkpoint loaders.
   - Replaced brittle C++ `xformers` dependencies with native PyTorch `F.scaled_dot_product_attention` (SDPA) fallbacks.
   - Decoupled `open_clip` imports and updated submodules for zero-friction setup.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/nguyenntt97/RI3D.git
cd RI3D

# Install with uv (or conda)
uv sync

# Download pre-trained diffusion models (SD 1.5, ControlNet Tile, SD2 Inpainting)
python scripts/download_hf_models.py

# Download MASt3R-SfM camera pose estimation models (for unposed images)
python mast3r/download_mast3r_models.py
```

---

## Quick Start

### 1. End-to-End Reconstruction (`inference.py`)

```bash
# A. Reconstruct from raw unposed sparse images (auto camera pose estimation via MASt3R-SfM)
python inference.py -i examples/sceneA/ -o output/sceneA --sfm_config unposed --num_views 3

# B. Reconstruct from pre-posed COLMAP / transforms datasets
python inference.py -i data/mipnerf360/bicycle -o output/bicycle --sfm_config posed --num_views 3

# C. Resume specific stages (e.g. Stage 5a repair & Stage 5b inpainting)
python inference.py -i examples/sceneA/ -o output/sceneA --sfm_config unposed --num_views 3 --stages 5a,5b,render
```

### 2. Interactive Visualization (`viz.py`)

```bash
# Serve Web Viewer (open http://localhost:9876 in your browser)
python viz.py -i output/sceneA --serve --port 9876

# Save recording to .rrd file and serve simultaneously
python viz.py -i output/sceneA --save sceneA.rrd --serve --port 9876

# Timeline scrubbing mode across reconstruction stages
python viz.py -i output/sceneA --serve --port 9876 --timeline
```

---

## Original Work & Citation

Please cite the original authors if you use RI3D in your research:

```bibtex
@inproceedings{paliwal2025ri3d,
    title={RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors},
    author={Avinash Paliwal and Xilong Zhou and Wei Ye and Jinhui Xiong and Rakesh Ranjan and Nima Khademi Kalantari},
    journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2025}
}
```

## Acknowledgements

This fork builds upon:
- **[RI3D](https://github.com/avinashpaliwal/RI3D)** (Paliwal et al., ICCV 2025)
- **[MASt3R](https://github.com/naver/mast3r)** / **[DUSt3R](https://github.com/naver/dust3r)** (Naver Labs)
- **[GaussianObject](https://github.com/GaussianObject/GaussianObject)** (Yang et al., SIGGRAPH Asia 2024)
- **[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)** (Kerbl et al., SIGGRAPH 2023)
- **[Rerun](https://github.com/rerun-io/rerun)** (Rerun.io)
- **[ControlNet](https://github.com/lllyasviel/ControlNet)** & **[Stable Diffusion](https://github.com/CompVis/stable-diffusion)**
