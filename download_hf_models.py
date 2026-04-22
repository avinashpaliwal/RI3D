from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    filename="v1-5-pruned.ckpt",
    local_dir="models",
)

hf_hub_download(
    repo_id="lllyasviel/ControlNet-v1-1",
    filename="control_v11f1e_sd15_tile.pth",
    local_dir="models",
)
