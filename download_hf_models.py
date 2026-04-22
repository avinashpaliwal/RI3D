from huggingface_hub import hf_hub_download, snapshot_download

# Stable Diffusion v1.5 checkpoint (for ControlNet backbone)
hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned.ckpt",
    local_dir="models",
)

# ControlNet Tile v1.1 weights
hf_hub_download(
    repo_id="lllyasviel/ControlNet-v1-1",
    filename="control_v11f1e_sd15_tile.pth",
    local_dir="models",
)

# SD2 Inpainting model (used as base for RealFill fine-tuning in Stage 4)
snapshot_download(
    repo_id="sd2-community/stable-diffusion-2-inpainting",
    local_dir="models/stable-diffusion-2-inpainting",
)
