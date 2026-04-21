import torch
from torchvision.transforms import Compose
import torch.nn.functional as F

import cv2

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
depth_anything.to(device)
depth_anything.eval()
print(f'depth_anything loaded on {device}')


transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

def estimate_depth(img, mode='test'):

    h, w = img.shape[1:3]

    img = transform({'image': img.permute(1, 2, 0).detach().cpu().numpy()})['image']
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    
    with torch.no_grad():
        depth = depth_anything(img)

    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    return depth