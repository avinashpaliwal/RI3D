import os
import io
import threestudio
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import cv2
import einops
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from PIL import Image
import clip
from gaussian_renderer import render
from scene import GaussianModel
from utils.arguments import PipelineParams, OptimizationParams
from scene.cameras import Render_Camera
from utils.sh_utils import SH2RGB
from utils.loss_utils import l1_loss, l2_loss, ssim, monodisp, masked_l1_loss, masked_ssim
from utils.graphics_utils import focal2fov, fov2focal
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import save_image, make_grid
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM, LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional.regression import pearson_corrcoef
from cldm.ddim_hacked import DDIMSampler
from cldm.annotator_util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from minlora import add_lora, LoRAParametrization
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.typing import *
from tqdm import tqdm
from ropwr import RobustPWRegression

from utils.depth_utils import estimate_depth
from utils.poisson_blend_utils import get_merged_depth
from utils.bilateral_filtering import sparse_bilateral_filtering
from utils.realfill_utils import InPaint
from scene.xfields import XfieldsFlow

from utils.depth_layering import get_depth_bins
from kornia.morphology import dilation

import yaml
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config = yaml.safe_load(open(os.path.join(_project_root, 'configs', 'argument.yaml'), 'r'))

@torch.no_grad()
def process(
    model,
    ddim_sampler: DDIMSampler,
    input_image: np.ndarray,
    prompt: str,
    a_prompt: str = '',
    n_prompt: str = '',
    num_samples: int = 1,
    image_resolution: int = 512,
    ddim_steps: int = 50,
    guess_mode: bool = False,
    strength: float = 1.0,
    scale: float = 1.0,
    eta: float = 1.0,
    denoise_strength: float = 1.0
):
    input_image = HWC3(input_image)
    detected_map = input_image.copy()

    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
    img = torch.stack([img for _ in range(num_samples)], dim=0)
    img = einops.rearrange(img, 'b h w c -> b c h w').clone()

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

    ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=False)
    t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
    z = model.get_first_stage_encoding(model.encode_first_stage(img))
    z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))
    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]


    alphas = ddim_sampler.alphas_cumprod.cuda()
    sds_w = (1 - alphas[t_enc]).view(-1, 1)

    return results, sds_w


from scipy.optimize import curve_fit

def piecewise_func(x, y, mask):
    split_point_x = np.percentile(x[mask.nonzero()], 95)
    split_point_x0 = x[mask.nonzero()].min()

    # Define the higher-order function (cubic in this case)
    def poly_func(x, a, b, c, d, e):
        return a + b * x + c * x**2 + d * x**3 #+ e * x**4 #+ f * x**5
    
    # Fit the higher-order function to the first segment
    params1, _ = curve_fit(poly_func, x[mask.nonzero()], y[mask.nonzero()])
    
    split_point_y = poly_func(split_point_x, *params1)

    # Define the linear function
    def linear_func(x, a, b):
        # return m*x + c
        # return a*(x-split_point_x)**2 + b*(x-split_point_x) + split_point_y
        return b*(x-split_point_x) + split_point_y
    

    split_point_y0 = poly_func(split_point_x0, *params1)
    # Define the linear function
    def linear_func0(x, a, b, c):
        # return m*x + c
        # return   b*x + c
        return c*(x-split_point_x0)**3 + a*(x-split_point_x0)**2 + b*(x-split_point_x0) + split_point_y0
        # return b*(x-split_point_x0) + split_point_y0


    # Fit the linear function to the second segment
    params2, _ = curve_fit(linear_func, x[(1-mask).nonzero()], y[(1-mask).nonzero()])


    x_test = np.arange(0, 1, 0.01)
    x_test = x_test * (x.max() - x.min()) + x.min()

    if round(split_point_x0, 3) > round(x.min(), 3):
        params3, _ = curve_fit(linear_func0, x[(x < split_point_x0).nonzero()], y[(x < split_point_x0).nonzero()])

        y_test = np.piecewise(x_test, [x_test < x[mask.nonzero()].min(), np.logical_and(x_test >= x[mask.nonzero()].min(), x_test < split_point_x), x_test >= split_point_x], 
                            [lambda x: linear_func0(x, *params3), lambda x: poly_func(x, *params1), lambda x: linear_func(x, *params2)])
    else:
        y_test = np.piecewise(x_test, [x_test < split_point_x, x_test >= split_point_x], 
                            [lambda x: poly_func(x, *params1), lambda x: linear_func(x, *params2)])
    
    if round(split_point_x0, 3) > round(x.min(), 3):

        return np.piecewise(x, [x < x[mask.nonzero()].min(), np.logical_and(x >= x[mask.nonzero()].min(), x < split_point_x), x >= split_point_x], 
                            [lambda x: linear_func0(x, *params3), lambda x: poly_func(x, *params1), lambda x: linear_func(x, *params2)])
    else:
        return np.piecewise(x, [x < split_point_x, x >= split_point_x], 
                            [lambda x: poly_func(x, *params1), lambda x: linear_func(x, *params2)])


def compute_tv_norm(values: torch.Tensor, losstype='l2') -> torch.Tensor:
    v00 = values[:, :-1, :-1]
    v01 = values[:, :-1, 1:]
    v10 = values[:, 1:, :-1]

    if losstype == 'l2':
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
    elif losstype == 'l1':
        loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
    else:
        raise ValueError('Not supported losstype.')
    return loss


def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        sh = np.random.random((vertices.count, 3)) / 255.0
        colors = SH2RGB(sh)
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
    

@threestudio.register("gaussian-object-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        sparse_num: int = 5
        model_name: str = "control_v11f1e_sd15_tile"
        exp_name: str = ""
        lora_name: str = "lora-step=1799.ckpt"
        lora_rank: int = 64
        add_diffusion_lora: bool = True
        add_control_lora: bool = True
        add_clip_lora: bool = True
        around_gt_steps: int = 0
        scene_extent: float = 5.0
        min_strength: float = 0.1
        max_strength: float = 1.0
        novel_image_size: int = 512
        refresh_interval: int = 100
        refresh_size: int = 20
        controlnet_num_samples: int = 1
        sh_degree: int = 2
        inpainting_dir: str = "inpainting"
        enable_inpainting: bool = False

        ctrl_steps: int = 1000
        ctrl_loss_ratio_begin: float = 1.0
        ctrl_loss_ratio_final: float = 0.5

    cfg: Config
    def configure(self) -> None:
        self.gaussian = GaussianModel(sh_degree = self.cfg.sh_degree, device="cuda")
        # self.gaussian_ref = GaussianModel(sh_degree = self.cfg.sh_degree)
        self.cameras_extent = self.cfg.scene_extent
        # self.bg_color = [1, 0, 1]#[1, 1, 1] if True else [0, 0, 0]
        # self.background_tensor = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.init_dreamer = self.cfg.init_dreamer
        self.point_cloud = self.init_pointcloud(self.init_dreamer)
        self.num_gauss   = self.gaussian.get_xyz.shape[0]
        self.gaussian.active_sh_degree = 1

        scene, num_view = self.cfg.init_dreamer.split('/')[-1].split('_')[:2]
        self.bg_color = [0, 0, 0]#bg_color_dict[scene]
        self.background_tensor = None#torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

        self.enable_inpainting = self.cfg.enable_inpainting

        if self.enable_inpainting:
            self.inpainter = InPaint(os.path.join(self.cfg.inpainting_dir, f'{scene}_{num_view}'))

        # metrics
        self.psnr = PSNR(data_range=1.0).to(self.gaussian.device)
        self.ssim = SSIM(data_range=1.0).to(self.gaussian.device)
        # self.lpips = LPIPS('vgg').to(self.gaussian.device)
        self.lpips_loss = LPIPS('vgg').to(self.gaussian.device)

        # data type align
        self.pil_to_tensor = ToTensor()
        self.tensor_to_pil = ToPILImage()

        # controlnet cache
        self.controlnet_outs: List[torch.Tensor] = []
        self.sds_ws: List[torch.Tensor] = []
        self.monodepths: List[torch.Tensor] = []
        self.visibility: List[torch.Tensor] = []
        self.all_T: torch.Tensor = torch.zeros((0, 3))
        self.max_cam_dis: float = 0.
        self.inpaint_counter: int = 0

        self.inpaint_batch = []

        # clip model
        self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', device=self.device)
        self.gt_features_all = []

        # lr scheduler
        self.novel_image_size = self.cfg.novel_image_size
        self.ctrl_steps = self.cfg.ctrl_steps
        self.ctrl_loss_ratio_begin = self.cfg.ctrl_loss_ratio_begin
        self.ctrl_loss_ratio_final = self.cfg.ctrl_loss_ratio_final
        self.ctrl_loss_ratio = self.ctrl_loss_ratio_begin


    def save_gif_to_file(self, images, output_file):  
        with io.BytesIO() as writer:  
            images[0].save(  
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
            )  
            writer.seek(0)  
            with open(output_file, 'wb') as file:  
                file.write(writer.read())


    def update_learning_rate(self):
        if self.global_step < self.ctrl_steps:
            self.ctrl_loss_ratio = self.ctrl_loss_ratio_begin + (self.ctrl_loss_ratio_final - self.ctrl_loss_ratio_begin) * self.global_step / self.ctrl_steps
        else:
            self.ctrl_loss_ratio = 0.0
        self.log("train/ctrl_loss_ratio", self.ctrl_loss_ratio)


    def cal_loss(self, args, image, render_dep, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid"):
        """
        Calculate the loss of the image, contains l1 loss and ssim loss.
        l1 loss: Ll1 = l1_loss(image, gt_image)
        ssim loss: Lssim = 1 - ssim(image, gt_image)
        Optional: [silhouette loss, monodepth loss]
        """
        gt_image = viewpoint_cam.original_image.to(image.dtype).to(image.device)
        # if self.opt.random_background:
        #     gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
        loss_mask = getattr(viewpoint_cam, "loss_mask", None)
        Ll1 = torch.nan_to_num(masked_l1_loss(image, gt_image, loss_mask))
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - masked_ssim(image, gt_image, loss_mask))

        self.lpips_loss = self.lpips_loss.to(image.device)
        # LPIPS has a deep receptive field, so there is no meaningful per-pixel
        # mask for it. Copying the render into the ground truth over the ignored
        # region makes the two identical there, which costs nothing and leaves no
        # gradient, rather than pulling the render toward the watermark.
        lp_image, lp_gt = image, gt_image
        if loss_mask is not None:
            keep = loss_mask.to(image.dtype).expand_as(image)
            lp_gt = gt_image * keep + image.detach() * (1.0 - keep)
        loss_lpips = torch.nan_to_num(self.lpips_loss(lp_image[None], lp_gt[None]))
        loss += loss_lpips * self.C(self.cfg.loss['lambda_lpips'])# * distance_weight
        # if silhouette_loss_type == "bce":
        #     silhouette_loss = torch.nan_to_num(F.binary_cross_entropy(render_pkg["rendered_alpha"][0], viewpoint_cam.mask))
        # elif silhouette_loss_type == "mse":
        #     silhouette_loss = torch.nan_to_num(F.mse_loss(render_pkg["rendered_alpha"][0], viewpoint_cam.mask))
        # else:
        #     raise NotImplementedError
        # loss = loss + self.opt.lambda_silhouette * silhouette_loss

        if hasattr(viewpoint_cam, "mono_depth") and viewpoint_cam.mono_depth is not None:
            
            
            # disp_mono = 1 / viewpoint_cam.mono_depth.clamp(1e-6).reshape(-1) # shape: [N]
            # disp_render = 1 / render_pkg["rendered_depth"][0].clamp(1e-6).reshape(-1) # shape: [N]
            # depth_loss = l1_loss(disp_mono, disp_render)
            if mono_loss_type == "mid":
                # we apply masked monocular loss
                # gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
                # render_mask = torch.where(render_pkg["rendered_alpha"][0] > 0.5, True, False)
                # mask = torch.logical_and(gt_mask, render_mask)
                # if mask.sum() < 10:
                #     depth_loss = 0.0
                # else:
                disp_mono = 1 / viewpoint_cam.mono_depth.clamp(1e-6).reshape(-1) # shape: [N]
                disp_render = 1 / render_dep[0].clamp(1e-6).reshape(-1) # shape: [N]
                depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]
            elif mono_loss_type == "pearson":
                # mono_depth is metric depth in SfM world units, so it correlates
                # directly with the render. The old -depth / 1/(depth+200) pair
                # were two ways of flipping Depth-Anything disparity into a
                # depth-like ordering, and both anti-correlate now.
                mono_depth = viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6)
                rendered_depth = render_dep[0][viewpoint_cam.mask > 0.5].clamp(1e-6)
                depth_loss = torch.nan_to_num(1 - pearson_corrcoef(mono_depth, rendered_depth))
            else:
                raise NotImplementedError

            loss += args.mono_rate * depth_loss

        else:
            depth_loss = 0.

        return {
            'loss': loss,
            'l1_loss': Ll1,
            'ssim_loss': 1.0 - ssim(image, gt_image),
            'silhouette_loss': torch.tensor([0]),
            'depth_loss': depth_loss
        }


    def render_gs(self, batch: Dict[str, Any], renderbackground=None, need_loss=False, supp_floaters=False) -> Dict[str, Any]:
        # if renderbackground is None:
        renderbackground = torch.rand((3), device=self.gaussian.device) #if self.opt.random_background else self.background_tensor
            # renderbackground = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda") if random.randint(0, 1) else torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
            
            # renderbackground = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        images, depths, alphas, viewcams, alphas_ref = [], [], [], [], []
        self.viewspace_point_list, self.radii = [], None # register one empty list for each image rendered
        loss_all = {
            'loss': 0.,
            'l1_loss': 0.,
            'ssim_loss': 0.,
            'silhouette_loss': 0.,
            'depth_loss': 0.
        }
        for id in range(batch['index'].shape[0]):
            viewpoint_cam = Render_Camera(
                batch['R'][id],
                batch['T'][id],
                batch['fovx'][id],
                batch['fovy'][id],
                batch['image'][id],
                None,
                batch['depth'][id],
                # batch['mask'] is the photometric-loss mask from loo_mip
                # (1 = supervise): it carries burned-in watermarks so they are not
                # learned as scene content. Not gt_alpha_mask, which would blank
                # the image there instead of leaving it unsupervised.
                batch['mask'][id] if 'mask' in batch else None,
                white_background = (self.bg_color == [1, 1, 1]),
                data_device=self.gaussian.device,
                distance=batch['distance'][id]
            )
            # if self.gaussian_ref is not None:
            #     with torch.no_grad():
            #         render_pkg_ref = render(viewpoint_cam, self.gaussian_ref, self.pipe, renderbackground)
            #         alphas_ref.append(render_pkg_ref["rendered_alpha"][0])

            clean_indices = None if not supp_floaters else self.gaussian.get_clean_indices(viewpoint_cam, scale=0.5)
            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground, clean_indices=clean_indices)
            viewpoint_cam.refactor(1)
            self.viewspace_point_list.append(render_pkg["viewspace_points"])
            self.radii = render_pkg["radii"] if id == 0 else torch.max(render_pkg["radii"], self.radii)
            images.append(render_pkg["render"]) # CHW
            depths.append(render_pkg["rendered_depth"][0])
            alphas.append(render_pkg["rendered_alpha"][0])
            viewcams.append(viewpoint_cam)
            if need_loss:
                loss = self.cal_loss(self.opt, render_pkg["render"], render_pkg["rendered_depth"], viewpoint_cam, renderbackground)
                for k, v in loss.items():
                    loss_all[k] += v
        self.visibility_filter = self.radii > 0.0 # update visibility filter
        return {
            "images": torch.stack(images, 0),
            "depths": torch.stack(depths, 0),
            "alphas": torch.stack(alphas, 0),
            "alphas_ref": torch.stack(alphas_ref, 0) if len(alphas_ref) > 0 else alphas_ref,
            "camera": viewcams,
            "loss": loss_all
        }


    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.controlnet = create_model(f'models/{self.cfg.model_name}.yaml').cpu()
        self.controlnet.load_state_dict(load_state_dict('models/v1-5-pruned.ckpt', location='cuda'), strict=False)
        self.controlnet.load_state_dict(load_state_dict(f'models/{self.cfg.model_name}.pth', location='cuda'), strict=False)
        lora_config = {
            nn.Embedding: {
                "weight": partial(LoRAParametrization.from_embedding, rank=self.cfg.lora_rank)
            },
            nn.Linear: {
                "weight": partial(LoRAParametrization.from_linear, rank=self.cfg.lora_rank)
            },
            nn.Conv2d: {
                "weight": partial(LoRAParametrization.from_conv2d, rank=self.cfg.lora_rank)
            }
        }
        if self.cfg.add_diffusion_lora:
            for name, module in self.controlnet.model.diffusion_model.named_modules():
                if name.endswith('transformer_blocks'):
                    add_lora(module, lora_config=lora_config)
        if self.cfg.add_control_lora:
            for name, module in self.controlnet.control_model.named_modules():
                if name.endswith('transformer_blocks'):
                    add_lora(module, lora_config=lora_config)
        if self.cfg.add_clip_lora:
            add_lora(self.controlnet.cond_stage_model, lora_config=lora_config)
        self.controlnet.load_state_dict(load_state_dict(f'{self.cfg.exp_name}/ckpts-lora/{self.cfg.lora_name}', location='cuda'), strict=False)
        self.controlnet = self.controlnet.cuda()
        self.ddim_sampler = DDIMSampler(self.controlnet)

    def get_dis_from_ts(self, T):
        return torch.sort(torch.sqrt(torch.sum((T - self.all_T) ** 2, dim=-1)))[0]


    def get_random_view_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        focal_length_x = fov2focal(batch['fovx'], batch['width'])
        focal_length_y = fov2focal(batch['fovy'], batch['height'])
        # return {
        #     'index': batch['random_index'],
        #     'R': batch['random_R'],
        #     'T': batch['random_T'],
        #     'height': torch.tensor([self.novel_image_size]),
        #     'width': torch.tensor([self.novel_image_size]),
        #     'fovx': torch.tensor([focal2fov(focal_length_x * batch['fov_scale'], self.novel_image_size)]),
        #     'fovy': torch.tensor([focal2fov(focal_length_y * batch['fov_scale'], self.novel_image_size)]),
        #     'image': torch.zeros((batch['image'].shape[0], batch['image'].shape[1], self.novel_image_size, self.novel_image_size), device=batch['image'].device),
        #     'mask': torch.zeros((batch['mask'].shape[0], self.novel_image_size, self.novel_image_size), device=batch['mask'].device),
        #     'depth': torch.zeros((batch['depth'].shape[0], self.novel_image_size, self.novel_image_size), device=batch['depth'].device),
        #     'txt': batch['txt']
        # }
        # return {
        #     'index': batch['random_index'],
        #     'R': batch['random_R'],
        #     'T': batch['random_T'],
        #     'height': batch['random_height'],
        #     'width': batch['random_width'],
        #     'fovx': torch.tensor([focal2fov(focal_length_x * batch['fov_scale'], batch['random_width'].item())]),
        #     'fovy': torch.tensor([focal2fov(focal_length_y * batch['fov_scale'], batch['random_height'].item())]),
        #     'image': torch.zeros((batch['image'].shape[0], batch['image'].shape[1], batch['height'], batch['width']), device=batch['image'].device),
        #     'mask': torch.zeros((batch['mask'].shape[0], batch['height'], batch['width']), device=batch['mask'].device),
        #     'depth': torch.zeros((batch['depth'].shape[0], batch['height'], batch['width']), device=batch['depth'].device),
        #     'txt': batch['txt']
        # }
        
        return {
            'index': batch['random_index'],
            'R': batch['random_R'],
            'T': batch['random_T'],
            # 'height': batch['random_height'],
            # 'width': batch['random_width'],
            # 'fovx': batch['random_fovx'],
            # 'fovy': batch['random_fovy'],
            'height': torch.tensor([self.novel_image_size]),
            'width': torch.tensor([self.novel_image_size]),
            'fovx': batch['fovx'] * batch['fov_scale'], #torch.tensor([focal2fov(focal_length_x, self.novel_image_size)]),
            'fovy': batch['fovy'] * batch['fov_scale'], #torch.tensor([focal2fov(focal_length_y, self.novel_image_size)]),
            'image': torch.zeros((batch['image'].shape[0], batch['image'].shape[1], self.novel_image_size, 768), device=batch['image'].device),      
            'distance': batch['random_distance'],     
            # 'image': torch.zeros((batch['image'].shape[0], batch['image'].shape[1], self.novel_image_size, int(self.novel_image_size * batch['width'] / batch['height'])), device=batch['image'].device),
            'mask': torch.zeros((batch['mask'].shape[0], self.novel_image_size, self.novel_image_size), device=batch['mask'].device),
            'depth': torch.zeros((batch['depth'].shape[0], self.novel_image_size, self.novel_image_size), device=batch['depth'].device),
            'txt': batch['txt']
        }
    
    def get_masks(self, depth_rel, num_bins=5):

        bins = get_depth_bins(disparity=depth_rel, num_bins=num_bins)
        bins = [1 / x for x in bins]
        bins.reverse()


        dep = depth_rel[0, 0]
        masks = []

        for i in range(len(bins) - 1):
            if i == 0:
                mask = torch.where((dep >= bins[i]) & (dep <= bins[i+1]), 1, 0)#.astype(np.uint8)
            else:
                mask = torch.where((dep > bins[i]) & (dep <= bins[i+1]), 1, 0)#.astype(np.uint8)
            masks.append(mask[None])
        
        masks = torch.cat(masks, dim=0)[None].cuda()
        return masks

    @torch.enable_grad()
    def align_depth(self, ren_depth, monodepth, alpha):
        h, w = ren_depth.shape
        decoder = XfieldsFlow(h, w, ngf=15, outChannels=5).cuda()
        decoder_scaling = torch.nn.Parameter(torch.ones(1).cuda().requires_grad_(True))

        monodepth_scaling = torch.nn.Parameter(torch.ones(1).cuda().requires_grad_(True))
        monodepth_offset = torch.nn.Parameter(torch.zeros(1).cuda().requires_grad_(True))
        optimizer = torch.optim.Adam([{'params': [monodepth_scaling], 'lr': 1}, {'params': [monodepth_offset], 'lr': 5}, {'params': [decoder_scaling], 'lr': 0.01}])
        decoder_optimizer = torch.optim.Adam([{'params': decoder.parameters(), 'lr': 1e-4}])

        decoder_scheduler = torch.optim.lr_scheduler.CyclicLR(decoder_optimizer, base_lr=1e-6, max_lr=1e-4, mode='triangular2', step_size_up=300, cycle_momentum=False)
        monoso_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        dec_inp = torch.tensor([[[[0]]]]).permute(3, 0, 1, 2).float().cuda()

        monodepth = torch.tensor(monodepth).float().cuda()[None, None]
        ren_depth = torch.tensor(ren_depth).float().cuda()[None, None]
        alpha = torch.tensor(alpha).float().cuda()[None, None, :, :, 0]

        masks = self.get_masks(monodepth.cpu(), num_bins=5)
        scaled_depth = 1 / (0.5 + (monodepth / monodepth.max()))

        for i in tqdm(range(1000)):

            optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            delta = decoder(dec_inp) * masks* 20 
            opt_depth = torch.sum(delta, dim=1, keepdim=True)+ scaled_depth * monodepth_scaling + monodepth_offset
            loss = torch.nn.functional.mse_loss(opt_depth * alpha, ren_depth * alpha)
            
            disp_mono =    1/scaled_depth.clamp(1e-6).reshape(-1) # shape: [N]
            disp_render =  1/opt_depth.clamp(1e-6).reshape(-1) # shape: [N]
            depth_loss =   monodisp(disp_mono, disp_render, 'l1')[-1]

            loss = loss + depth_loss
            
            loss.backward()
            if i < 500:
                optimizer.step()
            decoder_optimizer.step()

            decoder_scheduler.step()
            monoso_scheduler.step()

        return opt_depth.cpu().detach().numpy()[0, 0]
    
    def get_bgmask(self, depth_rel, num_bins=10, start=6, start_depth=None):
        bins = get_depth_bins(depth=depth_rel, num_bins=num_bins)
        
        dep = depth_rel[0, 0].detach().cpu().numpy() if isinstance(depth_rel, torch.Tensor) else depth_rel[0, 0]
        max_bin = bins[-1]
        start_idx = min(start, len(bins) - 1)
        if start_depth is not None:
            mask = np.where((dep >= start_depth) & (dep <= max_bin), 255, 0).astype(np.uint8)
        else:
            mask = np.where((dep >= bins[start_idx]) & (dep <= max_bin), 255, 0).astype(np.uint8)

        return mask

    # def inpaint_gs(self, image, alpha, depth, viewpoint_cam, batch):

    #     mask_orig = alpha.copy()
    #     mask_orig[mask_orig <  0.8] = 0
    #     mask_orig[mask_orig >= 0.8] = 255
    #     mask_orig = 255 - mask_orig
    #     # kernel = torch.ones(5, 5).cuda()
    #     # mask = dilation(mask[None], kernel)[0]
    #     # kernel = np.ones((5, 5), np.uint8)
    #     # mask_inp = cv2.dilate(mask_orig, kernel, iterations=1)#[..., None]
    #     # mask = cv2.morphologyEx(mask_orig, cv2.MORPH_OPEN, kernel)#[..., None].morphologyEx(img, cv.MORPH_OPEN, kernel)
    #     # mask = Image.fromarray(mask)
    #     mask_inp = mask_orig.copy()[..., 0]
    #     print(mask_inp.shape)

    #     image_wh = self.tensor_to_pil(image)

    #     image_np = self.inpainter.inpaint(image_wh, Image.fromarray(mask_inp))

    #     # np.save('alpha.npy', alpha)
    #     # np.save('depth.npy', depth)
    #     # np.save('image.npy', image_np)
    #     # np.save('diffout.npy', best_controlnet_out)
    #     # cv2.imwrite('alpha.jpg', alpha)
    #     # cv2.imwrite('depth.jpg', depth)
    #     # cv2.imwrite('image.jpg', image_np)
    #     # cv2.imwrite('diffout.jpg', best_controlnet_out)
    #     monodepth = estimate_depth(self.pil_to_tensor(image_np).cuda()).cpu().numpy()
    #     monodepth = 1.1 * abs(monodepth.min()) + monodepth
    #     monodepth = 1/monodepth

    #     # kernel = np.ones((5, 5), np.uint8)
    #     # mask = cv2.close(mask, kernel, iterations=1)#[..., None]
    #     # kernel = np.ones((15, 15), np.uint8)
    #     # mask_bg = self.get_bgmask(torch.tensor(1/monodepth)[None, None])
    #     # print(mask_bg.shape)
    #     # mask_bg = cv2.erode(mask_bg, kernel, iterations=1)
    #     # print(mask_bg.shape)
    #     # mask_proj = 
    #     # depth = depth * mask_bg / 255.
    #     blended_depth = get_merged_depth(depth, monodepth, alpha[..., None], mask_bg =None)[0][..., 0]
    #     # cv2.imwrite('blendepth.jpg', blended_depth)
    #     # cv2.imwrite('blenmask.jpg', mask)

    #     # np.save('arr.npy', {'blended_depth': blended_depth, 'best_controlnet_out':best_controlnet_out, 'mask':mask, 'viewpoint_cam':viewpoint_cam})
    #     # exit()

    #     depth_rel = 1 / blended_depth
    #     _, vis_depths = sparse_bilateral_filtering((depth_rel).copy(), image_np.copy()[..., :3], config, num_iter=config['sparse_iter'], spdb=False)
    #     blended_depth = 1 / vis_depths[-1]

    #     kernel = np.ones((15, 15), np.uint8)
    #     mask_proj = cv2.erode(mask_orig, kernel, iterations=1)[..., None]
    #     self.gaussian.depth_densify(blended_depth, image_np, mask_proj/255., viewpoint_cam)

    #     # print(blended_depth.shape, mask.shape, image_np.shape)
    #     # exit()
    #     cv2.imwrite('inpainted.png', image_np)
    #     cv2.imwrite("inp_mask.png", mask_inp)
    #     cv2.imwrite("inp_mask_proj.png", mask_proj)
    #     cv2.imwrite("inp_origmask.png", mask_orig)
    #     # cv2.imwrite("inp_mask_bg.png", mask_bg)
    #     cv2.imwrite("inp_inp.png", np.array(image_wh))
    #     cv2.imwrite("inp_dep.png", (depth - depth.min()) * 255 / (depth.max() - depth.min()).astype(np.uint8))
    #     cv2.imwrite("inp_monodepth.png", (monodepth - monodepth.min()) * 255 / (monodepth.max() - monodepth.min()).astype(np.uint8))
    #     cv2.imwrite("inp_dep_rel.png", (blended_depth - blended_depth.min()) * 255 / (blended_depth.max() - blended_depth.min()).astype(np.uint8))
    #     # controlnet_outs_image = make_grid(torch.cat([torch.tensor(blended_depth).permute(2, 0, 1).float(), torch.tensor(1 - mask[..., None]).permute(2, 0, 1).float(), torch.tensor(image_np/255.).permute(2, 0, 1).float()], dim=0), nrow=5)
    #     # save_image(controlnet_outs_image, self.get_save_path(f"controlnet_out/inp_it{self.true_global_step}.png"))

    #     return image_np
    #     # exit()
    def reject_outliers(self, data, m = 3.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else np.zeros(len(d))
        return np.where(s<m, 0, 1)

    @torch.no_grad()
    def inpaint_gs(self, image, alpha, depth, viewpoint_cam, batch):
        mask_orig = alpha.copy()
        mask_orig[mask_orig <  0.5] = 0
        mask_orig[mask_orig >= 0.5] = 255
        mask_orig = 255 - mask_orig
        # kernel = np.ones((5, 5), np.uint8)
        # mask_inp = cv2.dilate(mask_orig, kernel, iterations=1)#[..., None]
        kernel = np.ones((7, 7), np.uint8)
        mask_inp = cv2.dilate(mask_orig, kernel, iterations=1)#[..., None]
        # mask_inp = cv2.morphologyEx(mask_orig, cv2.MORPH_OPEN, kernel)

        mask_inp_left = mask_inp[:, :512].copy()
        image_left    = image[..., :512].clone()

        image_wh_left = self.tensor_to_pil(image_left)
        
        image_np_left = self.inpainter.inpaint(image_wh_left, Image.fromarray(mask_inp_left))
        image_np_left = self.pil_to_tensor(image_np_left)

        offset = mask_inp.shape[1] - 512
        mask_inp_right = mask_inp[:, offset:].copy()
        mask_inp_right[:, :512 - offset] = 0
        image_right   = image[..., offset:].clone()
        image_right[..., :512-offset] = image_np_left[..., offset:]

        image_wh_right = self.tensor_to_pil(image_right)
        
        image_np_right = self.inpainter.inpaint(image_wh_right, Image.fromarray(mask_inp_right))
        image_np_right = self.pil_to_tensor(image_np_right)

        image_np = torch.cat([image_np_left, image_np_right[..., 512-offset:]], dim=2)
        # print(image_np.shape, image_np_left.shape, image_np_right.shape)
        image_np = np.array(self.tensor_to_pil(image_np))

        image_wh = self.tensor_to_pil(image)

        cv2.imwrite("inp_mask.png", mask_inp)
        cv2.imwrite("inp_origmask.png", mask_orig)
        cv2.imwrite("inp_inp.png", np.array(image_wh))
        cv2.imwrite("inp_dep.png", ((depth - depth.min()) * 255 / (depth.max() - depth.min())).astype(np.uint8))

        # image_np = self.inpainter.inpaint(image_wh, Image.fromarray(mask_inp), '', strength=1.0)
        
        cv2.imwrite('inpainted.png', image_np)

        
        monodisparity = estimate_depth(self.pil_to_tensor(image_np).cuda()).cpu().numpy()
        
        rendepth_max = depth[alpha[..., 0] > 0.5].max()
        rendepth_min = depth[alpha[..., 0] > 0.5].min()
        monodisparity_max = monodisparity[alpha[..., 0] > 0.5].max()
        monodisparity_min = monodisparity[alpha[..., 0] > 0.5].min()

        scale = (1/rendepth_min - 1/rendepth_max) / (monodisparity_max - monodisparity_min)

        monodisparity = (monodisparity - monodisparity.min()) * scale + 1/rendepth_max
        monodepth = 1 / monodisparity

        mask = np.where(alpha > 0.5, 1, 0)
                        
        bins = get_depth_bins(depth=torch.tensor(monodepth, dtype=torch.float32)[None, None], num_bins=5, mask=mask[None, None])
        bins = [1/x for x in bins]

        try:
            pw = RobustPWRegression(objective="huber", degree=1, monotonic_trend="ascending", extrapolation="continue", extrapolation_bounds=(1e-3, 0.5))
            pw.fit(1/monodepth.reshape(-1)[mask.reshape(-1) == 1], 1/depth.reshape(-1)[mask.reshape(-1) == 1], splits=bins[1:-1])
            pred_disp = pw.predict(1/monodepth.reshape(-1))
            pred_disp = np.clip(pred_disp, 1e-4, None)
            refined = (1.0 / pred_disp).reshape(monodepth.shape)
            if np.isfinite(refined).all() and refined.min() > 0:
                monodepth = refined
        except Exception:
            pass

        monodepth = np.nan_to_num(monodepth, nan=float(depth.max()), posinf=float(depth.max()), neginf=float(depth.min()))
        monodepth = np.clip(monodepth, 1.5e-2, None)

        dep_range = max(float(depth.max() - depth.min()), 1e-6)
        cv2.imwrite("inp_monodepth.png", ((monodepth - depth.min()) * 255 / dep_range).astype(np.uint8))
        
        if self.inpaint_counter in [0, 1]:
            startt = 8
            start_d = 1.8
        elif self.inpaint_counter in [2, 3]:
            startt = 6
            start_d = 1.6
        elif self.inpaint_counter in [4, 5, 6, 7]:
            startt = 4
            start_d = 1.4
        else:
            startt = 2
            start_d = 1.2
        mask_bg = self.get_bgmask(torch.tensor(monodepth, dtype=torch.float32)[None, None], start=startt)
        # print(mask_bg.shape)
        kernel = np.ones((9, 9), np.uint8)
        mask_bg = cv2.erode(mask_bg, kernel, iterations=1)[..., None] / 255.
        # dep_max = (depth * (1 - mask_bg[..., 0])).min()
        # mask_bg_depth = np.where(depth < dep_max, 1, 0)[..., None]
        mask_bg_depth = self.get_bgmask(torch.tensor(depth + 1.5e-2, dtype=torch.float32)[None, None], start=8, start_depth=start_d*batch['distance'].item())[..., None]
        mask_bg_depth = cv2.erode(mask_bg_depth, kernel, iterations=1)[..., None] / 255.
        # print(mask_bg.shape)
        # mask_proj = 
        # depth = depth * mask_bg / 255.
        # mask_bg = cv2.erode(mask_bg, kernel, iterations=1)
        # inp_bg = np.median(depth * mask_bg * alpha)
        # print(f"inp_bg {inp_bg}")
        # blended_depth, mask_proj = get_merged_depth(depth * mask_bg / 255. + (1-mask_bg/255.)*2*batch['min_depth'].item(), monodepth, alpha[..., None])
        # print((1/depth).min(), (1/depth).max(), monodepth.min(), monodepth.max(), 1/monodepth.min(), 1/monodepth.max())
        # print(depth.max(), depth.min())
        # exit()
        # blended_depth, mask_proj = get_merged_depth(np.nan_to_num(depth), np.nan_to_num(monodepth), alpha[..., None], mask_fg=(1-mask_bg/255.))
        # # get_merged_depth(target * mask_bg1 + 2*target.min() * (1 - mask_bg1), depth_rel.numpy()[0, 0], mask
        # blended_depth = blended_depth[..., 0]
        # blended_depth = self.align_depth(depth, monodepth, alpha * mask_bg * mask_bg_depth)
        np.save('inp_ren_depth.npy', depth)
        np.save('inp_mono_depth.npy', monodepth)
        np.save('inp_alpha.npy', alpha * mask_bg * mask_bg_depth)

        cv2.imwrite("inp_mask_bgd.png", (255*mask_bg_depth).astype(np.uint8))
        cv2.imwrite("inp_mask_bg.png", (255*mask_bg).astype(np.uint8))
        cv2.imwrite("inp_mask_bga.png", (255*mask_bg*alpha).astype(np.uint8))
        # if self.inpaint_counter <= 5:
        blended_depth, _ = get_merged_depth(depth, monodepth, (alpha * mask_bg * mask_bg_depth)[..., 0])
        mask_proj     = (mask_inp[..., None] * mask_bg) / 255.
        # else:
        #     # # blended_depth = self.align_depth(depth, monodepth, alpha)
        #     # blended_depth = get_merged_depth(depth, monodepth, alpha[..., 0])
        #     blended_depth, _ = get_merged_depth(depth, monodepth, (alpha)[..., 0])
        #     mask_proj     = (mask_inp[..., None]) / 255.
        cv2.imwrite("inp_mask_proj_bg.png", (255*mask_proj*mask_bg).astype(np.uint8))

        cv2.imwrite("inp_mask_proj.png", (255*mask_proj).astype(np.uint8))
        cv2.imwrite("inp_dep_rel.png", ((blended_depth - depth.min()) * 255 / (depth.max() - depth.min())).astype(np.uint8))

        depth_rel = 1 / blended_depth
        _, vis_depths = sparse_bilateral_filtering((depth_rel).copy(), image_np.copy()[..., :3], config, num_iter=config['sparse_iter'], spdb=False)
        blended_depth = 1 / vis_depths[-1]

        self.inpaint_batch.append({'gt': image_np, 'mask': mask_proj[..., 0], 'batch': batch})
        # print(image_np.size, mask_proj.shape, batch)
        # exit()
        
        # if self.global_step < 3000:
        #     mask_proj = mask_proj * (1 - self.get_bgmask(torch.tensor(blended_depth)[None, None], num_bins=2) / 255.)[..., None]
        # mask_proj = mask_proj * (self.get_bgmask(torch.tensor(1/blended_depth)[None, None], num_bins=2) / 255.)[..., None]

        # mask_outliers = self.reject_outliers(blended_depth)
        # blended_depth = blended_depth * (1-mask_outliers) + mask_outliers * batch['max_depth'].item()

        # print(f"outliers {mask_outliers.nonzero()[0].shape}")
        
        # mask_proj[blended_depth == 2*batch['min_depth'].item()] = 0
        # cv2.imwrite('blendepth.jpg', blended_depth)
        # cv2.imwrite('blenmask.jpg', mask)

        # np.save('arr.npy', {'blended_depth': blended_depth, 'best_controlnet_out':best_controlnet_out, 'mask':mask, 'viewpoint_cam':viewpoint_cam})
        # exit()


        # kernel = np.ones((15, 15), np.uint8)
        # mask_proj = cv2.erode(mask_orig, kernel, iterations=1)[..., None]/255.
        # self.gaussian.depth_densify(blended_depth, image_np, (mask_bg[..., None]/255. * mask_proj), viewpoint_cam)

        # print(blended_depth.shape, mask.shape, image_np.shape)
        # exit()
        # controlnet_outs_image = make_grid(torch.cat([torch.tensor(blended_depth).permute(2, 0, 1).float(), torch.tensor(1 - mask[..., None]).permute(2, 0, 1).float(), torch.tensor(image_np/255.).permute(2, 0, 1).float()], dim=0), nrow=5)
        # save_image(controlnet_outs_image, self.get_save_path(f"controlnet_out/inp_it{self.true_global_step}.png"))

        # return blended_depth, image_np, (mask_bg[..., None]/255. * mask_proj), viewpoint_cam
        torch.cuda.empty_cache()
        return blended_depth, image_np, (mask_proj), viewpoint_cam
        # exit()

    @torch.no_grad()
    def denoise_gs(self, batch, image_np, image, denoise_strength):
        
            controlnet_outs, sds_w = process(
                self.controlnet,
                self.ddim_sampler,
                image_np,
                prompt = batch['txt'][0],
                a_prompt = 'best quality',
                n_prompt = 'blur, lowres, bad anatomy, bad hands, cropped, worst quality',
                num_samples = self.cfg.controlnet_num_samples,
                image_resolution = min(image_np.shape[0], image_np.shape[1]),
                ddim_steps = 50,
                guess_mode = False,
                strength = 1.0,
                scale = 1.0,
                eta = 1.0,
                denoise_strength = denoise_strength
            )
            best_controlnet_out = controlnet_outs[0]
            best_controlnet_out_score = 0.
            for controlnet_out in controlnet_outs:
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(self.clip_preprocess(Image.fromarray(controlnet_out)).unsqueeze(0).to(self.device))
                score = sum([torch.cosine_similarity(image_features, gt_features, dim=-1).mean() for gt_features in self.gt_features_all])
                if score > best_controlnet_out_score:
                    best_controlnet_out = controlnet_out
                    best_controlnet_out_score = score

            best_controlnet_out = cv2.resize(best_controlnet_out, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_CUBIC)
            self.cond_images.append(image.unsqueeze(0))
            self.controlnet_outs.append(self.pil_to_tensor(best_controlnet_out.astype(np.uint8)).to(torch.float32).unsqueeze(0).to(self.gaussian.device))
            self.sds_ws.append(sds_w.to(self.gaussian.device))
            torch.cuda.empty_cache()

    def training_step(self, batch, batch_idx):
        if self.max_cam_dis == 0.:
            Ts = batch['gt_Ts']
            self.all_T = torch.cat(Ts)
            for T in Ts:
                distances = self.get_dis_from_ts(T)
                self.max_cam_dis = max(self.max_cam_dis, distances[2].cpu().item())
            # TODO: magic number here
            self.max_cam_dis *= 1.2

        #REMOVE FLOATERS
        # if self.global_step % 50 == 0:
        #     # self.gaussian.prune(self.opt.prune_opacity_threshold, self.cameras_extent, None)
        #     self.gaussian.remove_floaters(batch['min_depth'], self.dataset.render_cam_infos, self.dataset.scene_center)

        # inpainting_index = {'inp':-1}
        with torch.no_grad():
            if self.global_step % self.cfg.refresh_interval == 0 and self.global_step <= self.cfg.around_gt_steps and self.global_step < self.ctrl_steps:
                self.controlnet_outs = []
                self.cond_images = []
                if self.global_step < self.cfg.around_gt_steps:
                    if len(self.gt_features_all) == 0:
                        for gt_image in batch['gt_images']:
                            with torch.no_grad():
                                gt_features = self.clip_model.encode_image(self.clip_preprocess(self.tensor_to_pil(gt_image[0])).unsqueeze(0).to(self.device))
                            self.gt_features_all.append(gt_features)
                    
                    self.inpainting_index = -1
                    # if self.global_step >2000:


                    # # ##############Inpainting #####################
                    if self.enable_inpainting and self.inpaint_counter < 8:
                        floaters = False
                        counter = 0
                        # for i in range(self.cfg.refresh_size):
                        # self.inpainting_index = 0#random.randint(0, self.cfg.refresh_size - 1)
                        R, T = batch['random_poses'][self.inpainting_index]
                        retuns = []
                        batch_inp = batch['random_poses'][::2] #if random.randint(0,1) else batch['random_poses'][1::2]
                        for ind, (R, T) in enumerate(batch_inp): # Use 4 of 8 at even angles
                            controlent_batch = batch.copy()
                            controlent_batch['random_R'] = R
                            controlent_batch['random_T'] = T
                            controlent_batch = self.get_random_view_batch(controlent_batch)
                            render_results = self.render_gs(controlent_batch, renderbackground=self.background_tensor, need_loss=False, supp_floaters=True)
                            
                            depths = render_results['depths']

                            # if not floaters:
                            images = render_results['images']
                            image = torch.clamp(images[0], 0, 1)

                            alphas = render_results['alphas']
                            alpha  = alphas[0].detach().cpu().numpy()
                            depth  = depths[0].detach().cpu().numpy() / (alpha + 1e-6)
                            alpha  = alpha[..., None]
                            viewpoint_cam = render_results['camera'][0]

                            retuns.append(self.inpaint_gs(image, alpha, depth, viewpoint_cam, controlent_batch))
                            
                            for retun in retuns:
                                            
                                self.gaussian.depth_densify(*retun)

                        self.inpaint_counter += 1

                    self.inpainting_index = -1

                    # ########################################################

                    for ind, (R, T) in tqdm(enumerate(batch['random_poses'])):
                        controlent_batch = batch.copy()
                        controlent_batch['random_R'] = R
                        controlent_batch['random_T'] = T
                        controlent_batch = self.get_random_view_batch(controlent_batch)
                        render_results = self.render_gs(controlent_batch, renderbackground=self.background_tensor, need_loss=False, supp_floaters=True)
                        images = render_results['images']
                        image = images[0]

                        alphas = render_results['alphas']
                        alpha  = alphas[0].detach().cpu().numpy()[..., None]
                        depths = render_results['depths']
                        depth  = depths[0].detach().cpu().numpy()
                        viewpoint_cam = render_results['camera'][0]

                        # if self.global_step > 2000 and ind == inpainting_index:
                            
                        #     image_np = self.inpaint_gs(image, alpha, depth, viewpoint_cam)

                        # else:

                        image_np = np.array(self.tensor_to_pil(image))
                        
                        # denoise_strength = 0.5 * (1.2 - self.global_step / self.cfg.around_gt_steps)#0.0 * (self.cfg.max_strength - self.cfg.min_strength) + self.cfg.min_strength
                        # if self.global_step <= 4000:
                        #     denoise_strength = 0.5
                        # else:
                        #     denoise_strength = 0.3 * (1.1 - self.global_step / self.cfg.around_gt_steps)#0.0 * (self.cfg.max_strength - self.cfg.min_strength) + self.cfg.min_strength
                        denoise_strength = 1.0
                        self.denoise_gs(batch, image_np, image, denoise_strength)
                        self.visibility.append(alphas[0].detach())
                        
                        # del blended_depth, viewpoint_cam, monodepth, mask
                
                    controlnet_outs_image = make_grid(torch.cat(self.controlnet_outs, dim=0), nrow=5)
                    save_image(controlnet_outs_image, self.get_save_path(f"controlnet_out/it{self.true_global_step}.png"))
                    cond_images = make_grid(torch.cat(self.cond_images, dim=0), nrow=5)
                    save_image(cond_images, self.get_save_path(f"controlnet_out/it{self.true_global_step}c.png"))
                    # cond_depths = make_grid(torch.cat(self.monodepths, dim=0), nrow=5)
                    # save_image(cond_depths, self.get_save_path(f"controlnet_out/it{self.true_global_step}d.png"))
                    print("controlnet_outs saved")
            
                # if self.global_step > 0:
                #     self.gaussian.reset_opacity(val=0.2)

        # self.gaussian.update_learning_rate(self.true_global_step)
        if self.inpaint_counter >= 8:
            self.gaussian.set_learning_rate_xyz(lr_xyz=0.00016, lr_scaling=0.005)

        render_results = self.render_gs(batch, need_loss=True, supp_floaters=True)

        if (self.global_step + 1) % self.cfg.refresh_interval == 0:
            print("saving rendered +++++++++++++++++++++++++++++++++++")
            save_arr = []
            with torch.no_grad():
                for ind, (R, T) in enumerate(batch['random_poses']):
                    controlent_batch = batch.copy()
                    controlent_batch['random_R'] = R
                    controlent_batch['random_T'] = T
                    controlent_batch = self.get_random_view_batch(controlent_batch)
                    render_results = self.render_gs(controlent_batch, renderbackground=self.background_tensor, need_loss=False, supp_floaters=True)
                    image = render_results['images'][0]
                    save_arr.append(image.unsqueeze(0))
            rendered_grid = make_grid(torch.cat(save_arr, dim=0), nrow=5)
            save_image(rendered_grid, self.get_save_path(f"controlnet_out/it{self.true_global_step}ren.png"))


        for k, v in render_results['loss'].items():
            self.log(f"retrain/{k}", v)

        gs_loss = render_results['loss']['loss']
        self.log("retrain/gs_loss", gs_loss)

        ctrl_loss = 0.
        batch = self.get_random_view_batch(batch)
        if self.ctrl_loss_ratio > 0.0 and batch['index'][0] != self.inpainting_index:
            

            render_results = self.render_gs(batch, renderbackground=torch.rand(3, dtype=torch.float32, device=self.gaussian.device), need_loss=False, supp_floaters=True)
            images = render_results['images']

            controlnet_outs = []
            sds_ws = []
            alphas = []
            if self.global_step < self.cfg.around_gt_steps:
                for idx, image in enumerate(images):
                    cached_controlnet_out = self.controlnet_outs[batch['index'][idx]]
                    if cached_controlnet_out is not None:
                        controlnet_out = cached_controlnet_out
                        sds_w = self.sds_ws[batch['index'][idx]]
                        alphas.append(self.visibility[batch['index'][idx]])
                    controlnet_outs.append(controlnet_out)
                    sds_ws.append(sds_w)

            distances = self.get_dis_from_ts(batch['T'])
            distance_weight = min(1., 2 * distances[0].cpu().item() / self.max_cam_dis)
            self.log("train/distance_weight", distance_weight)

            # TODO: only works for batch size 1
            controlnet_outs = torch.cat(controlnet_outs, dim=0)
            # alphas = torch.cat(alphas, dim=0) if self.inpaint_counter <= 8 else torch.ones_like(torch.cat(alphas, dim=0))
            alphas = torch.ones_like(torch.cat(alphas, dim=0))
            sds_ws = sds_ws[0].cpu().item()
            self.log("train/sds_ws", sds_ws)

            loss_l1 = torch.nan_to_num(l1_loss(controlnet_outs, images))
            self.log("train/loss_l1", loss_l1)
            ctrl_loss += sds_ws * loss_l1 * self.C(self.cfg.loss['lambda_l1']) * distance_weight

            # loss_l2 = torch.nan_to_num(l2_loss(alphas * controlnet_outs, alphas * images))
            # self.log("train/loss_l2", loss_l2)
            # ctrl_loss += sds_ws * loss_l2 * self.C(self.cfg.loss['lambda_l2']) * distance_weight
            self.lpips_loss = self.lpips_loss.to(alphas.device)
            loss_lpips = torch.nan_to_num(self.lpips_loss(alphas * controlnet_outs, alphas * images))
            self.log("train/loss_lpips", loss_lpips)
            ctrl_loss += sds_ws * loss_lpips * self.C(self.cfg.loss['lambda_lpips']) * distance_weight

            loss_tv = torch.nan_to_num(compute_tv_norm(render_results['depths'], losstype='l2').sqrt().mean())
            self.log("train/loss_tv", loss_tv)
            ctrl_loss += sds_ws * loss_tv * self.C(self.cfg.loss['lambda_tv']) * distance_weight

            self.log("train/loss", ctrl_loss)
        else:
            ctrl_loss = torch.tensor([0]).to(self.gaussian.device)
            # continue
        self.update_learning_rate()
        # print(gs_loss.device, ctrl_loss.device)
        loss = gs_loss * (1.0 - self.ctrl_loss_ratio) + ctrl_loss * self.ctrl_loss_ratio

        if self.inpaint_batch:
            inp_batch = random.choice(self.inpaint_batch)

            render_results = self.render_gs(inp_batch['batch'], renderbackground=self.background_tensor, need_loss=False, supp_floaters=True)
            images = render_results['images']
            mask = torch.tensor(inp_batch['mask']).to(self.gaussian.device)[None, None].float()
            gt = self.pil_to_tensor(inp_batch['gt']).to(self.gaussian.device)[None].float()
            loss += self.C(self.cfg.loss['lambda_lpips']) * torch.nan_to_num(self.lpips_loss(mask * gt, mask * images))
            
            # Ll1 = torch.nan_to_num(l1_loss(mask * gt, mask * images))
            # loss += (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(mask * gt, mask * images))

            # loss += torch.nan_to_num(l2_loss(render_results['alphas'] * mask, mask))


        # loss = ctrl_loss

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}


    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if self.true_global_step < self.opt.densify_until_iter:
                N = self.gaussian.get_xyz.shape[0]
                viewspace_point_tensor_grad = torch.zeros(N, 2, device=self.gaussian.device)
                for idx in range(len(self.viewspace_point_list)):
                    vpt = self.viewspace_point_list[idx]
                    grad = getattr(vpt, 'absgrad', None)
                    if grad is None:
                        grad = vpt.grad
                    if grad is not None:
                        if grad.dim() == 3:
                            grad = grad[0]  # [C, N, 2] -> [N, 2]
                        viewspace_point_tensor_grad = viewspace_point_tensor_grad + grad[:N, :2]
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])

                self.gaussian.add_densification_stats_no_grad(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step >= self.opt.densify_from_iter and self.true_global_step % self.opt.densification_interval == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 300 else None # 3000
                    before_num_gauss = len(self.gaussian._xyz)
                    if before_num_gauss < self.opt.max_num_splats:
                        self.gaussian.densify(self.opt.densify_grad_threshold, self.cameras_extent)
                    if before_num_gauss > self.opt.min_num_splats:
                        self.gaussian.prune(self.opt.prune_opacity_threshold, self.cameras_extent, size_threshold)
                    torch.cuda.empty_cache()
                    after_num_gauss = len(self.gaussian._xyz)
                    threestudio.info(f'Run densification at step: {self.true_global_step}, before: {before_num_gauss}, after: {after_num_gauss}')
                    self.log('gaussian/num_gauss', torch.tensor(after_num_gauss, dtype=torch.float32))
                if self.inpaint_counter >= 8 and self.true_global_step % self.opt.opacity_reset_interval == 0:
                    self.gaussian.reset_opacity()
                if self.true_global_step >= 200 and self.true_global_step % (self.opt.opacity_reset_interval//5) == 0:
                    self.gaussian.reduce_opacity(val=0.8)


    def on_train_epoch_end(self):
        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)


    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        self.opt = OptimizationParams(self.parser)
        for k, v in self.cfg.gaussian_opt_params.items():
            self.opt.__setattr__(k, v)
        self.pipe = PipelineParams(self.parser)
        self.gaussian.training_setup(self.opt)
        ret = {
            "optimizer": self.gaussian.optimizer,
        }
        return ret


    def init_pointcloud(self, path):
        if 'output_den' in path:
            
            threestudio.info(f'init ply file from denoising ckpt {path}')
            ply_path = f'{path}/save/last.ply'
        else:
            if path == 'random':
                pcb = self.pcb()
                self.gaussian.create_from_pcd(pcb, self.cameras_extent)
                return self.pcb()
            max_num = 0
            ply_path = ''
            for it in os.listdir(os.path.join(path, 'point_cloud')):
                if not os.path.isdir(os.path.join(path, 'point_cloud', it)):
                    continue
                num = int(it.split('_')[-1])
                if num > max_num:
                    max_num = num
                    ply_path = os.path.join(path, 'point_cloud', it, 'point_cloud.ply')
            threestudio.info(f'init ply file from iter {max_num} {ply_path}')

        self.point_cloud = fetchPly(ply_path)
        self.num_pts = self.point_cloud.points.shape[0]
        self.gaussian.load_ply(ply_path)
        self.gaussian.update_spatial_lr_scale(self.cameras_extent)
        # self.gaussian_ref.load_ply(ply_path.replace('point_cloud.ply', 'point_cloudOrig.ply'))
        return self.point_cloud
