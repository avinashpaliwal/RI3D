import torch
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel

class InPaint():
    
    def __init__(self, model_path):

        # create & load model
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "models/stable-diffusion-2-inpainting",
            torch_dtype=torch.float32,
            revision=None
        )

        self.pipe.unet = UNet2DConditionModel.from_pretrained(
            model_path, subfolder="unet", revision=None,
        )
        self.pipe.text_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", revision=None,
        )
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to("cuda")

        self.generator = torch.Generator(device="cuda").manual_seed(1234)

    
    def inpaint(self, image, mask_image, prompt="a photo of xxy5syt00", strength=1.0):

        print("Inpainting prompt...", prompt)
        # image = Image.open(image)
        # mask_image = Image.open(mask)

        results = self.pipe(
            [prompt], image=image, mask_image=mask_image, negative_prompt = "blur, lowres, bad anatomy, bad hands, cropped, worst quality",
            num_inference_steps=200, guidance_scale=1, generator=self.generator, strength=strength, 
        ).images

        # result = Image.composite(results[0], image, mask_image)

        return np.array(results[0])