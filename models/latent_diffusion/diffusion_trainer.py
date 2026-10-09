import json
import os
import time

import kornia.losses
import numpy as np
import torch
import torchvision
from kornia.losses import ssim_loss
from torch import optim, nn
import torch.nn.functional as F

from models.generic_trainer import GenericTrainer, create_object, LPLoss, compare_models
from models.latent_diffusion.autoencoder import AutoencoderKL, VQModelInterface
from models.latent_diffusion.encoders import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder, BERTEmbedder
from models.latent_diffusion.openaimodel import UNetModel
from models.latent_diffusion.modified_openai import ModifiedUNetModel
from models.latent_diffusion.regression_openai import UNetModel as RegressionModel
from models.loss_functions import LaplacianLoss, CharbonnierLoss
import yaml

from models.vgg_loss import TVLoss
from util.datasets import OutputType
from util.pcm_atm import PcmcAtm


# image_size = 128,
# in_channels = 3,
# model_channels = 128,
# out_channels = 1,
# num_res_blocks = 3,
# # attention_resolutions=[1, 2, 4, 8],
# attention_resolutions = [2, 4, 8],
# dropout = 0.25,
# channel_mult = (1, 2, 3, 4),
# conv_resample = True,
# dims = 2,
# num_classes = None,
# use_checkpoint = True,
# use_fp16 = False,
# num_heads = 8,
# # num_heads=1,
# # num_head_channels=32,
# num_head_channels = -1,
# num_heads_upsample = -1,
# use_scale_shift_norm = True,
# resblock_updown = True,
# use_new_attention_order = True,  # False uses legacy
# use_spatial_transformer = False,  # This is for context integration
# transformer_depth = 2,
# # context_dim=64,
# # context_dim=768,
# context_dim = None,
# n_embed = None,
# legacy = False,
# use_downsample_block = False,



class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        # so that I can index arrays with t=1...T, since x_0 = x so I ignore t=0
        self.noise_steps = noise_steps + 1
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.pcmc_atm = PcmcAtm(device=device)

        # self.noise_function = self.pcmc_atm.generate_noise
        self.noise_function = torch.randn_like

        self.beta = self.prepare_noise_schedule().to(device)
        self.beta = self.beta.view(-1, 1, 1, 1)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        # self.sqrt_alpha_hat[0] = 1
        # self.sqrt_one_minus_alpha_hat[0] = 0

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def generate_noise(self, x):
        return self.noise_function(x)

    def noise_images(self, x, t):
        # sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        # sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        eps = self.generate_noise(x)
        return self.sqrt_alpha_hat[t] * x + self.sqrt_one_minus_alpha_hat[t] * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def generate_timesteps_array(self, x, t):
        return torch.ones(x.shape[0]) * t

    def denoising_step(self, x_t, t, t_next, pred):
        t = t.long()
        t_next = t_next.long()
        if t_next[0] > 0:
            z = self.generate_noise(x_t)
        else:
            z = torch.zeros_like(x_t)

        x0 = (x_t - self.sqrt_one_minus_alpha_hat[t] * pred) / self.sqrt_alpha_hat[t]
        x_t_next = self.sqrt_alpha_hat[t_next] * x0 + self.sqrt_one_minus_alpha_hat[t_next] * z
        # x_t_next = self.sqrt_alpha_hat[t_next] * x0 + self.sqrt_one_minus_alpha_hat[t_next] * pred
        return x_t_next


def wrap_img(img):
    """
    Wrap an image
    :param img: unwrapped image between [-1, 1]
    :return: wrapped image between [-1, 1]
    """
    ### Proper way
    # img = img * 30
    # img = (img + torch.pi) % (2 * torch.pi) - torch.pi
    # img = (img + torch.pi) / (2 * torch.pi)
    # img = img * 2 - 1

    ### Simplified way
    img = (img * 30 + torch.pi) % (2 * torch.pi)
    img = img / torch.pi - 1
    return img


class RegressionLoss(nn.Module):
    def __init__(self, device='cuda', wrap=False):
        super(RegressionLoss, self).__init__()
        self.wrap = wrap
        self.device = device
        with open('results/etna_wrap_256/Regression_network_150.json', "r") as infile:
            self.args = json.load(infile)

        self.model = RegressionModel(**self.args)
        state_dict = torch.load('results/etna_wrap_256/Regression_network_150.pt', weights_only=True)
        self.model.load_state_dict(state_dict)

        self.model.eval()
        self.model.requires_grad = False
        # for p in self.model.parameters():
        #     p.requires_grad_(False)

        self.model.to(device)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, output, target):
        if not self.wrap:
            output = wrap_img(output)
            target = wrap_img(target)

        o = self.model(output.to(self.device))
        with torch.no_grad():
            t = self.model(target.to(self.device))

        # t = target.to(self.device)
        loss = self.mse(o, t)
        # print(o.cpu(), t.cpu(), loss.cpu())
        # print(loss)
        return loss.to(output.device)


class VGG(nn.Module):
    """VGG/Perceptual Loss

    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """

    def __init__(self, conv_index: str = '22'):

        super(VGG, self).__init__()
        vgg_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT).features
        modules = [m for m in vgg_features]

        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229, 0.224, 0.225)
        # self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)

        self.vgg.eval()
        self.vgg.requires_grad = False
        self.vgg.requires_grad_(False)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """

        if sr.shape[1] == 1:
            s = torch.cat((sr, sr, sr), dim=1)
            h = torch.cat((hr, hr, hr), dim=1)
            sr = s
            hr = h

        def _forward(x):
            # x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss


# class CombinedLoss(torch.nn.Module):
#     def __init__(self, device):
#         super(CombinedLoss, self).__init__()
#         self.device = device
#         self.mse = torch.nn.MSELoss()
#         # self.charbonnier = CharbonnierLoss()
#         # self.laplacian = LaplacianLoss(device=device)
#         self.l1 = nn.L1Loss()
#         self.vgg = VGG().to(self.device)
#
#     def forward(self, output, target):
#         return 1 * self.mse(output, target) + 1 * self.l1(output, target) + 1 * self.vgg(output, target)
#         # return 1 * self.mse(x, y) + 1 * self.l1(x, y)
#         # return 1 * self.mse(x, y) + 1 * self.charbonnier(x, y) + 1 * self.laplacian(x, y)


class ImageEmbedderKL(nn.Module):
    def __init__(self, model_folder='models/latent_diffusion/first_stage_models/kl-f32/', device="cpu"):
        super(ImageEmbedderKL, self).__init__()
        self.device = device

        with open(model_folder + 'config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        data['model']['params']['lossconfig'] = '__is_first_stage__'

        self.encoder = AutoencoderKL(**data['model']['params'], ckpt_path=model_folder + 'model.ckpt')
        self.encoder = self.encoder.to(device)

        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def forward(self, x):
        posterior = self.encoder.encode(x)
        return posterior.sample()

    def decode(self, x):
        dec = self.encoder.decode(x)
        return dec


class ImageEmbedderVQ(nn.Module):
    def __init__(self, model_folder='models/latent_diffusion/first_stage_models/vq-f4/', device="cpu"):
        super(ImageEmbedderVQ, self).__init__()
        self.device = device

        with open(model_folder + 'config.yaml', 'r') as file:
            data = yaml.safe_load(file)

        data['model']['params']['lossconfig'] = '__is_first_stage__'

        self.encoder = VQModelInterface(**data['model']['params'], ckpt_path=model_folder + 'model.ckpt')
        self.encoder = self.encoder.to(device)

        self.encoder.eval()
        self.encoder.requires_grad_(False)

    def forward(self, x):
        h = self.encoder.encode(x)
        return h


class ExpLoss(torch.nn.Module):
    def __init__(self):
        super(ExpLoss, self).__init__()
        self.unreduced_mse = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        return torch.mean(torch.exp(self.unreduced_mse(input, target)) - 1)


class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l3 = LPLoss(3)
        self.charb = CharbonnierLoss(eps=1e-9)

    def forward(self, input, target):
        return self.mse(input, target) + self.l3(input, target) + self.charb(input, target)


def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)


def masked_ssim_loss(pred, target, mask, window_size=5, eps=1e-8):
    """
    pred, target: (B, C, H, W) in [0, 1]
    mask: (B, 1, H, W) or broadcastable
    """
    pred = (pred + 1) / 2
    target = (target + 1) / 2

    # Compute SSIM map (NOT reduced)
    ssim_map = kornia.metrics.ssim(
        pred, target,
        window_size=window_size,
    )  # shape: (B, C, H, W)

    # Convert to loss
    loss_map = 1.0 - ssim_map

    # Ensure mask shape matches
    if mask.shape[1] == 1 and loss_map.shape[1] > 1:
        mask = mask.expand_as(loss_map)

    # Apply mask + normalize
    masked_loss = (loss_map * mask).sum() / (mask.sum() + eps)

    return masked_loss


class DiffusionTrainer(GenericTrainer):
    def __init__(self, args, train_loader, test_loader, summary_writer, logger=None):
        super().__init__(args, train_loader, test_loader, summary_writer, logger)
        self.name = 'Diffusion'

        in_channels = 2
        if 'dem' in args.model_type:
            self.use_dem = True
            self.name += 'DEM'
            in_channels += 1
        else:
            self.use_dem = False

        self.model = create_object(
            UNetModel,
            # ModifiedUNetModel,
            # image_size=self.image_size / 2 if args.downsample else self.image_size,
            # image_size=4,
            image_size=self.image_size,
            in_channels=in_channels,
            model_channels=64,
            # model_channels=128,
            out_channels=1,
            # num_res_blocks=4,
            num_res_blocks=2,
            # attention_resolutions=[1, 2, 4, 8],
            # attention_resolutions=[2, 4, 8],
            attention_resolutions=[2, 4, 8],
            dropout=0,
            # channel_mult=(1, 2, 3, 4),
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=True,
            use_fp16=False,
            # num_heads=16,
            num_heads=8,
            # num_heads=-1,
            # num_head_channels=32,
            # num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,  # False uses legacy
            use_spatial_transformer=False,  # This is for context integration
            transformer_depth=2,
            # context_dim=44,
            # context_dim=2816,
            # context_dim= 1024,
            # context_dim=768 * 2,
            context_dim=None,
            n_embed=None,
            legacy=False,
            use_downsample_block=False,
        )
        print(f'Model with params: {self.model.call_args}')
        # self.save_parameters()
        # self.model_state = self.model.state_dict().__str__()


        self.embedders_device = 'cuda'
        self.image_embedder = ImageEmbedderKL(device=self.embedders_device)
        # self.image_embedder = FrozenClipImageEmbedder(device='cpu')

        # self.text_embedder = BERTEmbedder(64, 16, device='cuda')
        self.text_embedder = FrozenCLIPTextEmbedder(device=self.embedders_device)

        # self.image_embedder = ImageEmbedderVQ()

        self.noise_steps = 1000
        # self.denoising_steps = 100
        self.train_denoising_steps = 10

        self.diffusion = Diffusion(noise_steps=self.noise_steps, img_size=self.image_size, device=self.device)

        # self.optimizer = optim.AdamW(self.model.parameters(), 1e-4, weight_decay=25 * 1e-5)
        self.optimizer = optim.AdamW(self.model.parameters(), 1e-4, weight_decay=25 * 1e-5)
        # self.optimizer = optim.AdamW(self.model.parameters(), 1e-5, weight_decay=0)
        # self.optimizer = optim.Adam(self.model.parameters())

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.loss1 = nn.MSELoss()
        # self.loss2 = CombinedLoss()
        self.tv = TVLoss(p=2)
        self.vgg = VGG().to(self.device)
        self.ssim_loss = kornia.losses.SSIMLoss(window_size=5)

        self.unreduced_mse = nn.MSELoss(reduction='none')
        self.unreduced_l1 = nn.L1Loss(reduction='none')
        self.exp_loss = ExpLoss()
        # self.regression_loss = RegressionLoss(device='gpu')
        self.l3 = LPLoss(3)
        # self.combined_loss = CombinedLoss(device=self.device)
        # self.cel = nn.CrossEntropyLoss()

        self.downsample = torchvision.transforms.Resize(128)
        self.upsample = torchvision.transforms.Resize(self.image_size)

        self.save_count = 0
        self.saved_values = []
        self.buffer = 0

    def prepare_train(self):
        pass
        # self.image_embedder = self.image_embedder.to(self.device)
        # self.text_embedder = self.text_embedder.to(self.device)

    def train_step(self, data_dict) -> torch.Tensor:
        # return self.train_step_no_context(batch, dem, ground_truth, deform_chars)
        return self.train_step_with_context(data_dict)

    def train_step_with_context(self, data_dict):
        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        orig_img = data_dict[OutputType.ORIG_IMG]
        deform = data_dict[OutputType.DEFORMATION]
        annotations = data_dict[OutputType.ANNOTATION]
        image_time_length = data_dict[OutputType.TIME_LENGTH]
        mask = data_dict[OutputType.WATER_MASK]

        # deform = (deform + 1) / 2
        # deform = deform.to(self.image_embedder.device)
        # deform_embed = self.image_embedder(torch.cat([deform, deform, deform], dim=1))
        # deform_embed = deform_embed.to(self.device)
        #
        # start_time = time.time()
        # t = self.diffusion.sample_timesteps(deform_embed.shape[0]).to(self.device)
        # x_t, noise = self.diffusion.noise_images(deform_embed, t)
        #
        # image = image.to(self.image_embedder.device)
        # image = (image + 1) / 2
        # dem = dem.to(self.image_embedder.device)
        # dem = (dem + 1) / 2
        #
        # image_context = self.image_embedder(torch.cat([image, image, image], dim=1))
        # image_context = image_context.view(image_context.shape[0], 1, -1)
        #
        # dem_context = self.image_embedder(torch.cat([dem, dem, dem], dim=1))
        # dem_context = dem_context.view(dem_context.shape[0], 1, -1)
        #
        # annotations_context = self.text_embedder(annotations)
        # annotations_context = annotations_context.view(annotations_context.shape[0], 1, -1)
        #
        # time_length_context = self.text_embedder(image_time_length)
        # time_length_context = annotations_context.view(time_length_context.shape[0], 1, -1)
        #
        # context = torch.cat((image_context, time_length_context, dem_context, annotations_context), dim=2).to(self.device)


        start_time = time.time()
        t = self.diffusion.sample_timesteps(deform.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(deform, t)

        if self.use_dem:
            input = torch.cat([image, x_t, dem], dim=1)
        else:
            input = torch.cat([image, x_t], dim=1)


        predicted_noise = self.model(input, t)
        # prediction_loss = self.loss1(predicted_noise, noise)
        prediction_loss = self.unreduced_mse(predicted_noise, noise)
        prediction_loss = masked_mean(prediction_loss, mask)
        # prediction_loss = self.exp_loss(predicted_noise, noise)

        # x_t = self._sample_random_steps(deform_embed, context)
        x_t = self._sample_random_steps_v2(deform, image, dem)
        # output_loss = self.loss2(x_t, deform)
        # output_loss = self.l1(x_t, deform) + self.ssim_loss(x_t, deform)
        l1_loss = self.unreduced_l1(x_t, deform)
        l1_loss = masked_mean(l1_loss, mask)
        output_loss = l1_loss + masked_ssim_loss(x_t, deform, mask)

        # decoded_x_t = self.image_embedder.decode(x_t.to(self.image_embedder.device))
        # decoded_x_t = decoded_x_t.to(self.device)
        # decoded_x_t = decoded_x_t[:, 0] # has 3 channels, but they are all the same
        # decoded_x_t = decoded_x_t.view(decoded_x_t.shape[0], 1, *decoded_x_t.shape[1:])
        # deform = deform.to(self.device)
        # reconstruction_loss = self.loss2(decoded_x_t, deform)

        # reconstruction_l3 = self.l3(x_t, ground_truth)
        # unreduced_reconstruction_loss = self.unreduced_l1(x_t, ground_truth)
        # weight = 2 * torch.abs(ground_truth)
        # unreduced_reconstruction_loss = weight * unreduced_reconstruction_loss
        # reconstruction_loss = unreduced_reconstruction_loss.mean() + reconstruction_l3


        loss = prediction_loss + output_loss
        # print(loss)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # if self.image_embedder.encoder.state_dict().__str__() == self.image_embedder.model_state:
        #     print("Image embedder not modified")
        # if self.text_embedder.model.state_dict().__str__() == self.text_embedder.model_state:
        #     print("Text embedder not modified")
        # if self.model_state != self.model.state_dict().__str__():
        #     print("Model modified")

        time_elapsed = time.time() - start_time
        # print(time_elapsed)
        return loss

    def train_step_no_context(self, batch, dem, ground_truth, deform_chars):
        # batch = self.image_embedder(torch.cat([batch, batch, batch], dim=1))
        # dem = self.image_embedder(torch.cat([dem, dem, dem], dim=1))
        # ground_truth = self.image_embedder(torch.cat([ground_truth, ground_truth, ground_truth], dim=1))
        # down_batch = self.downsample(batch)
        # down_dem = self.downsample(dem)

        start_time = time.time()
        t = self.diffusion.sample_timesteps(ground_truth.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(ground_truth, t)
        # x = torch.cat([self.downsample(x_t), down_batch, down_dem], dim=1)
        x = torch.cat([x_t, batch, dem], dim=1)

        # downsampled_x = self.downsample(x)
        # downsampled_predicted_noise = self.model(x, t)
        # predicted_noise = self.upsample(downsampled_predicted_noise)
        predicted_noise = self.model(x, t)


        # prediction_loss = self.mse(predicted_noise, noise) + self.l1(predicted_noise, noise)
        # unreduced_prediction_loss = self.unreduced_l1(predicted_noise, noise)

        prediction_loss = self.exp_loss(predicted_noise, noise)
        # prediction_loss = self.mse(predicted_noise, noise)

        # downsampled_x_t = self._sample_random_steps_no_context_updown(down_batch, down_dem)
        # x_t = self.upsample(downsampled_x_t)
        x_t = self._sample_random_steps_no_context(batch, dem)

        # reconstruction_loss = self.combined_loss(x_t, ground_truth)
        # reconstruction_loss = self.regression_loss(x_t, ground_truth) + self.mse(x_t, ground_truth) + self.l1(x_t, ground_truth)
        # reconstruction_loss =  self.mse(x_t, ground_truth) + self.l1(x_t, ground_truth)

        # unreduced_reconstruction_loss = self.unreduced_mse(x_t, ground_truth) + self.unreduced_l1(x_t, ground_truth)
        reconstruction_l3 = self.l3(x_t, ground_truth)

        unreduced_reconstruction_loss = self.unreduced_l1(x_t, ground_truth)
        weight = 2 * torch.abs(ground_truth)
        unreduced_reconstruction_loss = weight * unreduced_reconstruction_loss
        reconstruction_loss = unreduced_reconstruction_loss.mean() + reconstruction_l3

        # print(self.regression_loss(x_t, ground_truth), self.mse(x_t, ground_truth), self.l1(x_t, ground_truth))
        # unreduced_loss = unreduced_prediction_loss + unreduced_reconstruction_loss
        loss = prediction_loss + reconstruction_loss
        # loss = prediction_loss + unreduced_reconstruction_loss.mean()

        # print(prediction_loss, reconstruction_loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        time_elapsed = time.time() - start_time

        # self.buffer += 1
        # if self.buffer > 100:
        #     if unreduced_prediction_loss.shape[0] == self.batch_size:
        #         self.save_count += 1
        #         if self.save_count < 400:
        #             unreduced_loss = unreduced_prediction_loss + unreduced_reconstruction_loss
        #             if self.save_count < 5:
        #                 print(unreduced_loss.shape)
        #             self.saved_values.append((ground_truth.cpu(), unreduced_loss.cpu()))
        #         if self.save_count % 20 == 0:
        #             stats_obj = {'res': self.saved_values}
        #             np.savez(f"loss_function_stats_{self.save_count}",**stats_obj)
        #             self.saved_values = []

        return loss

    def train_step_text_context(self, batch, dem, ground_truth, annotations):
        with torch.no_grad():
            annotations_context = self.text_embedder(annotations)

            context = annotations_context.view(annotations_context.size(0), 1, -1)
            context = context.to(self.device)

        t = self.diffusion.sample_timesteps(ground_truth.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(ground_truth, t)
        x = torch.cat([x_t, batch, dem], dim=1)

        predicted_noise = self.model(x, t, context)
        prediction_loss = self.combined_loss(noise, predicted_noise)

        x_t = self._sample_random_steps_text_context(batch, dem, context)

        reconstruction_loss = self.combined_loss(ground_truth, x_t)

        loss = prediction_loss + 2 * reconstruction_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return loss

    def _sample_random_steps_no_context(self, batch, dem):
        steps = torch.randint(low=1, high=self.noise_steps, size=(self.train_denoising_steps,))
        steps = torch.sort(steps, descending=True).values

        x_t = self.diffusion.generate_noise(batch).to(self.device)

        for ind, t_val in enumerate(steps):
            t = self.diffusion.generate_timesteps_array(x_t, t_val).to(self.device)
            t_next_val = steps[ind + 1] if ind < self.train_denoising_steps - 1 else 0
            t_next = self.diffusion.generate_timesteps_array(x_t, t_next_val).to(self.device)
            x = torch.cat([x_t, batch, dem], dim=1)
            pred = self.model(x, t)
            x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        return x_t

    def _sample_random_steps_no_context_updown(self, batch, dem):
        steps = torch.randint(low=1, high=self.noise_steps, size=(self.train_denoising_steps,))
        steps = torch.sort(steps, descending=True).values

        x_t = self.diffusion.generate_noise(batch).to(self.device)

        for ind, t_val in enumerate(steps):
            t = self.diffusion.generate_timesteps_array(x_t, t_val).to(self.device)
            t_next_val = steps[ind + 1] if ind < self.train_denoising_steps - 1 else 0
            t_next = self.diffusion.generate_timesteps_array(x_t, t_next_val).to(self.device)
            x = torch.cat([x_t, batch, dem], dim=1)
            # downsampled_x = self.downsample(x)
            pred = self.model(x, t)
            # pred = self.upsample(downsampled_pred)
            x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        return x_t

    def _sample_random_steps_text_context(self, batch, dem, context):
        steps = torch.randint(low=1, high=self.noise_steps, size=(self.train_denoising_steps,))
        steps = torch.sort(steps, descending=True).values

        x_t = self.diffusion.generate_noise(batch).to(self.device)

        for ind, t_val in enumerate(steps):
            t = self.diffusion.generate_timesteps_array(x_t, t_val).to(self.device)
            t_next_val = steps[ind + 1] if ind < self.train_denoising_steps - 1 else 0
            t_next = self.diffusion.generate_timesteps_array(x_t, t_next_val).to(self.device)
            x = torch.cat([x_t, batch, dem], dim=1)
            pred = self.model(x, t, context)
            x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        return x_t

    def _sample_random_steps(self, batch, context):
        steps = torch.randint(low=1, high=self.noise_steps, size=(self.train_denoising_steps,))
        steps = torch.sort(steps, descending=True).values

        x_t = self.diffusion.generate_noise(batch).to(self.device)

        for ind, t_val in enumerate(steps):
            t = self.diffusion.generate_timesteps_array(x_t, t_val).to(self.device)
            t_next_val = steps[ind + 1] if ind < self.train_denoising_steps - 1 else 0
            t_next = self.diffusion.generate_timesteps_array(x_t, t_next_val).to(self.device)
            pred = self.model(x_t, t, context)
            x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        return x_t

    def _sample_random_steps_v2(self, batch, context, dem=None):
        steps = torch.randint(low=1, high=self.noise_steps, size=(self.train_denoising_steps,))
        steps = torch.sort(steps, descending=True).values

        x_t = self.diffusion.generate_noise(batch).to(self.device)

        for ind, t_val in enumerate(steps):
            t = self.diffusion.generate_timesteps_array(x_t, t_val).to(self.device)
            t_next_val = steps[ind + 1] if ind < self.train_denoising_steps - 1 else 0
            t_next = self.diffusion.generate_timesteps_array(x_t, t_next_val).to(self.device)
            if self.use_dem:
                pred = self.model(torch.cat([context, x_t, dem], dim=1), t)
            else:
                pred = self.model(torch.cat([context, x_t], dim=1), t)
            x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        return x_t

    def _sample_image(self, batch, context, steps=-1, training=False):
        if steps < 1:
            steps = self.train_denoising_steps

        x_t = self.diffusion.generate_noise(batch).to(self.device)
        for i in reversed(range(1, steps)):
            t_ind = (i - 1) * self.noise_steps / steps + 1
            t = self.diffusion.generate_timesteps_array(x_t, t_ind).to(self.device)
            t_next_ind = (i - 2) * self.noise_steps / steps + 1 if i > 1 else 0
            t_next = self.diffusion.generate_timesteps_array(x_t, t_next_ind).to(self.device)
            pred = self.model(x_t, t, context)
            x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        return x_t

    def denoise_image(self, batch, annotations, dem, steps=-1) -> torch.Tensor:
        # return self.denoise_image_no_context(batch, dem, steps)
        return self.denoise_image_context(batch, dem, annotations, steps)

    def denoise_image_no_context(self, batch, dem, steps):
        self.model.eval()
        with torch.no_grad():
            x_t = self.diffusion.generate_noise(batch).to(self.device)
            for i in reversed(range(1, steps)):
                t_ind = (i - 1) * self.noise_steps / steps + 1
                t = self.diffusion.generate_timesteps_array(x_t, t_ind).to(self.device)
                t_next_ind = (i - 2) * self.noise_steps / steps + 1 if i > 1 else 0
                t_next = self.diffusion.generate_timesteps_array(x_t, t_next_ind).to(self.device)
                x = torch.cat([x_t, batch, dem], dim=1)
                pred = self.model(x, t)
                x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)
        return x_t

    def denoise_image_context(self, image, dem, annotations, time_length, steps):
        self.model.eval()
        with torch.no_grad():
            # batch = batch.to(self.image_embedder.device)
            # dem = dem.to(self.image_embedder.device)
            #
            # image_context = self.image_embedder(
            #     torch.cat([(batch + 1) / 2, torch.zeros_like(batch), torch.zeros_like(batch)], dim=1))
            # image_context = image_context.view(image_context.shape[0], 1, -1)
            #
            # dem_context = self.image_embedder(
            #     torch.cat([(dem + 1) / 2, torch.zeros_like(dem), torch.zeros_like(dem)], dim=1))
            # dem_context = dem_context.view(dem_context.shape[0], 1, -1)
            #
            # annotations_context = self.text_embedder(annotations)
            # annotations_context = annotations_context.view(annotations_context.shape[0], 1, -1)
            #
            # context = torch.cat((image_context, dem_context, annotations_context), dim=2).to(self.device)
            # context = image_context.to(self.device)
            image = image.to(self.image_embedder.device)
            image = (image + 1) / 2
            dem = dem.to(self.image_embedder.device)
            dem = (dem + 1) / 2

            image_context = self.image_embedder(torch.cat([image, image, image], dim=1))
            dummy = torch.zeros_like(image_context)
            image_context = image_context.view(image_context.shape[0], 1, -1)

            dem_context = self.image_embedder(torch.cat([dem, dem, dem], dim=1))
            dem_context = dem_context.view(dem_context.shape[0], 1, -1)

            annotations_context = self.text_embedder(annotations)
            annotations_context = annotations_context.view(annotations_context.shape[0], 1, -1)

            time_length_context = self.text_embedder(time_length)
            time_length_context = annotations_context.view(time_length_context.shape[0], 1, -1)

            context = torch.cat((image_context, time_length_context, dem_context, annotations_context), dim=2).to(
                self.device)


            x_t = self.diffusion.generate_noise(dummy).to(self.device)
            for i in reversed(range(1, steps)):
                t_ind = (i - 1) * self.noise_steps / steps + 1
                t = self.diffusion.generate_timesteps_array(x_t, t_ind).to(self.device)
                t_next_ind = (i - 2) * self.noise_steps / steps + 1 if i > 1 else 0
                t_next = self.diffusion.generate_timesteps_array(x_t, t_next_ind).to(self.device)
                pred = self.model(x_t, t, context)
                x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        decoded_x_t = self.image_embedder.decode(x_t.to(self.image_embedder.device))
        decoded_x_t = decoded_x_t.to(self.device)
        decoded_x_t = decoded_x_t[:, 0] # has 3 channels, but they are all the same
        decoded_x_t = decoded_x_t.view(decoded_x_t.shape[0], 1, *decoded_x_t.shape[1:])
        decoded_x_t = decoded_x_t * 2 - 1 # trained on (deform + 1) / 2
        return decoded_x_t

    def denoise_image_context_v2(self, image, dem, annotations, time_length, steps):
        self.model.eval()
        with torch.no_grad():
            context = image
            dummy = torch.zeros_like(image)

            x_t = self.diffusion.generate_noise(dummy).to(self.device)
            for i in reversed(range(1, steps)):
                t_ind = (i - 1) * self.noise_steps / steps + 1
                t = self.diffusion.generate_timesteps_array(x_t, t_ind).to(self.device)
                t_next_ind = (i - 2) * self.noise_steps / steps + 1 if i > 1 else 0
                t_next = self.diffusion.generate_timesteps_array(x_t, t_next_ind).to(self.device)

                if self.use_dem:
                    pred = self.model(torch.cat([context, x_t, dem], dim=1), t)
                else:
                    pred = self.model(torch.cat([context, x_t], dim=1), t)
                x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

            return x_t
    
    def test_step(self, data_dict) -> torch.Tensor:
        # return self.denoise_image_no_context(batch, dem, 10)
        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        annotations = data_dict[OutputType.ANNOTATION]
        time_length = data_dict[OutputType.TIME_LENGTH]
        return self.denoise_image_context_v2(image, dem, annotations, time_length, 30)


class DiffusionContextTrainer(GenericTrainer):
    def __init__(self, args, train_loader, test_loader, summary_writer, logger=None):
        super().__init__(args, train_loader, test_loader, summary_writer, logger)
        self.name = 'DiffusionContext'

        self.model = create_object(
            UNetModel,
            # ModifiedUNetModel,
            # image_size=self.image_size / 2 if args.downsample else self.image_size,
            # image_size=4,
            image_size=self.image_size,
            in_channels=1,
            model_channels=64,
            # model_channels=128,
            out_channels=1,
            # num_res_blocks=4,
            num_res_blocks=2,
            # attention_resolutions=[1, 2, 4, 8],
            # attention_resolutions=[2, 4, 8],
            attention_resolutions=[2, 4, 8],
            dropout=0,
            # channel_mult=(1, 2, 3, 4),
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=True,
            use_fp16=False,
            # num_heads=16,
            num_heads=8,
            # num_heads=-1,
            # num_head_channels=32,
            # num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_new_attention_order=True,  # False uses legacy
            use_spatial_transformer=True,  # This is for context integration
            transformer_depth=2,
            context_dim=1024 * 2 + 768 * 2,
            n_embed=None,
            legacy=False,
            use_downsample_block=False,
        )
        print(f'Model with params: {self.model.call_args}')
        # self.save_parameters()
        # self.model_state = self.model.state_dict().__str__()

        self.embedders_device = 'cuda'
        self.image_embedder = ImageEmbedderKL(device=self.embedders_device)
        # self.image_embedder = FrozenClipImageEmbedder(device='cpu')

        # self.text_embedder = BERTEmbedder(64, 16, device='cuda')
        self.text_embedder = FrozenCLIPTextEmbedder(device=self.embedders_device)

        # self.image_embedder = ImageEmbedderVQ()

        self.noise_steps = 1000
        # self.denoising_steps = 100
        self.train_denoising_steps = 10

        self.diffusion = Diffusion(noise_steps=self.noise_steps, img_size=self.image_size, device=self.device)

        # self.optimizer = optim.AdamW(self.model.parameters(), 1e-4, weight_decay=25 * 1e-5)
        self.optimizer = optim.AdamW(self.model.parameters(), 1e-4, weight_decay=25 * 1e-5)
        # self.optimizer = optim.AdamW(self.model.parameters(), 1e-5, weight_decay=0)
        # self.optimizer = optim.Adam(self.model.parameters())

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.loss1 = nn.MSELoss()
        # self.loss2 = CombinedLoss()
        self.tv = TVLoss(p=2)
        self.vgg = VGG().to(self.device)
        self.ssim_loss = kornia.losses.SSIMLoss(window_size=5)

        self.unreduced_mse = nn.MSELoss(reduction='none')
        self.unreduced_l1 = nn.L1Loss(reduction='none')
        self.exp_loss = ExpLoss()
        # self.regression_loss = RegressionLoss(device='gpu')
        self.l3 = LPLoss(3)
        # self.combined_loss = CombinedLoss(device=self.device)
        # self.cel = nn.CrossEntropyLoss()

        self.save_count = 0
        self.saved_values = []
        self.buffer = 0

    def train_step(self, data_dict) -> torch.Tensor:
        return self.train_step_with_context(data_dict)

    def train_step_with_context(self, data_dict):
        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        orig_img = data_dict[OutputType.ORIG_IMG]
        deform = data_dict[OutputType.DEFORMATION]
        annotations = data_dict[OutputType.ANNOTATION]
        image_time_length = data_dict[OutputType.TIME_LENGTH]

        image = (image + 1) / 2
        image = image.to(self.image_embedder.device)
        image_context = self.image_embedder(torch.cat([image, image, image], dim=1))
        image_context = image_context.to(self.device)
        image_context = image_context.view(image_context.shape[0], 1, -1)

        dem = (dem + 1) / 2
        dem = dem.to(self.image_embedder.device)
        dem_context = self.image_embedder(torch.cat([dem, dem, dem], dim=1))
        dem_context = dem_context.to(self.device)
        dem_context = dem_context.view(dem_context.shape[0], 1, -1)

        annotations_context = self.text_embedder(annotations)
        annotations_context = annotations_context.view(annotations_context.shape[0], 1, -1)

        time_length_context = self.text_embedder(image_time_length)
        time_length_context = annotations_context.view(time_length_context.shape[0], 1, -1)

        context = torch.cat((image_context, time_length_context, dem_context, annotations_context), dim=2).to(self.device)

        start_time = time.time()
        t = self.diffusion.sample_timesteps(deform.shape[0]).to(self.device)
        x_t, noise = self.diffusion.noise_images(deform, t)

        predicted_noise = self.model(x_t, t, context)
        prediction_loss = self.loss1(predicted_noise, noise)

        x_t = self._sample_random_steps(deform, context)
        output_loss = self.l1(x_t, deform) + self.ssim_loss(x_t, deform)

        loss = prediction_loss + output_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        time_elapsed = time.time() - start_time
        # print(time_elapsed)
        return loss

    def _sample_random_steps(self, batch, context):
        steps = torch.randint(low=1, high=self.noise_steps, size=(self.train_denoising_steps,))
        steps = torch.sort(steps, descending=True).values

        x_t = self.diffusion.generate_noise(batch).to(self.device)

        for ind, t_val in enumerate(steps):
            t = self.diffusion.generate_timesteps_array(x_t, t_val).to(self.device)
            t_next_val = steps[ind + 1] if ind < self.train_denoising_steps - 1 else 0
            t_next = self.diffusion.generate_timesteps_array(x_t, t_next_val).to(self.device)
            pred = self.model(x_t, t, context)
            x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        return x_t

    def denoise_image_context(self, image, dem, annotations, image_time_length, steps):
        self.model.eval()
        with torch.no_grad():
            image = (image + 1) / 2
            image = image.to(self.image_embedder.device)
            image_context = self.image_embedder(torch.cat([image, image, image], dim=1))
            image_context = image_context.to(self.device)
            image_context = image_context.view(image_context.shape[0], 1, -1)

            dem = (dem + 1) / 2
            dem = dem.to(self.image_embedder.device)
            dem_context = self.image_embedder(torch.cat([dem, dem, dem], dim=1))
            dem_context = dem_context.to(self.device)
            dem_context = dem_context.view(dem_context.shape[0], 1, -1)

            annotations_context = self.text_embedder(annotations)
            annotations_context = annotations_context.view(annotations_context.shape[0], 1, -1)

            time_length_context = self.text_embedder(image_time_length)
            time_length_context = annotations_context.view(time_length_context.shape[0], 1, -1)

            context = torch.cat((image_context, time_length_context, dem_context, annotations_context), dim=2).to(
                self.device)

            dummy = image # shape of output
            x_t = self.diffusion.generate_noise(dummy).to(self.device)
            for ind, t_val in enumerate(steps):
                t = self.diffusion.generate_timesteps_array(x_t, t_val).to(self.device)
                t_next_val = steps[ind + 1] if ind < self.train_denoising_steps - 1 else 0
                t_next = self.diffusion.generate_timesteps_array(x_t, t_next_val).to(self.device)
                pred = self.model(x_t, t, context)
                x_t = self.diffusion.denoising_step(x_t, t, t_next, pred)

        return x_t

    def test_step(self, data_dict) -> torch.Tensor:
        # return self.denoise_image_no_context(batch, dem, 10)
        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        annotations = data_dict[OutputType.ANNOTATION]
        image_time_length = data_dict[OutputType.TIME_LENGTH]
        return self.denoise_image_context(image, dem, annotations, image_time_length, 30)


