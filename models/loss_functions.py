# MIT License
#
# Copyright (c) 2024 Alper Ahmetoglu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import torch
from torch import nn, Tensor
from piqa import SSIM, PSNR
import torch
import scipy.ndimage as ndi


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    return conv_gauss(x_up, 4 * gauss_kernel(channels=x.shape[1], device=x.device))


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out


def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr


class LaplacianLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, device=torch.device('cpu')):
        super(LaplacianLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels, device=device)

    def forward(self, x, y):
        pyr_input = laplacian_pyramid(img=x, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=y, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


def get_outnorm(x: torch.Tensor, out_norm: str = ''):
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


class CharbonnierLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, out_norm: str = 'bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps ** 2))
        return loss * norm


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class WeightedTVLoss(nn.L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


class FFTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.criterion(pred_fft, target_fft)


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


def create_soft_mask(mask, sigma=3):
    mask_np = mask.cpu().numpy()

    soft = np.zeros_like(mask_np)

    for b in range(mask_np.shape[0]):
        soft[b,0] = ndi.gaussian_filter(mask_np[b,0], sigma=sigma)

    soft = np.clip(soft, 0.0, 1.0)

    return torch.from_numpy(soft).float()


def remove_plane(img):
    B, C, H, W = img.shape
    device = img.device

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij"
    )

    A = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)  # H,W,3
    A = A.view(-1, 3)  # (HW,3)

    pinv = torch.linalg.pinv(A)  # (3,HW)

    img_flat = img.view(B*C, -1)  # (BC,HW)

    coeff = (pinv @ img_flat.T).T  # (BC,3)

    a = coeff[:,0].view(B, C, 1, 1)
    b = coeff[:,1].view(B, C, 1, 1)
    c = coeff[:,2].view(B, C, 1, 1)

    plane = a*xx + b*yy + c

    return img - plane


class AtmFFTLoss(torch.nn.Module):

    def __init__(self, loss_weight=0.02):
        super().__init__()
        self.loss_weight = loss_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, mask=None):
        pred = remove_plane(pred)
        target = remove_plane(target)

        if mask is not None:
            soft_mask = create_soft_mask(mask).to(pred.device)
            pred = pred * soft_mask
            target = target * soft_mask

        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        B,C,H,W = pred.shape

        fy = torch.fft.fftfreq(H, device=pred.device)
        fx = torch.fft.rfftfreq(W, device=pred.device)

        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(xx**2 + yy**2)

        # weight = r / (1 + (r / 0.15)**2)
        weight = 1 / (1 + (r / 0.2) ** 2)

        weight = weight.unsqueeze(0).unsqueeze(0)
        pred_fft = pred_fft * weight
        target_fft = target_fft * weight

        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)

        return self.loss_weight * self.l1(pred_fft, target_fft)


class SimpleFFTLoss(nn.Module):

    def __init__(self, loss_weight=0.01):
        super().__init__()
        self.loss_weight = loss_weight
        self.l1 = nn.L1Loss()

    def forward(self, pred, target, mask=None):

        if mask is not None:
            pred = pred * mask
            target = target * mask

        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        loss = self.l1(pred_amp, target_amp)

        return self.loss_weight * loss
