import math
from collections import Counter

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import _LRScheduler

from models.generic_trainer import GenericTrainer, create_object
from models.loss_functions import FFTLoss
from models.wavemamba.wavemamba import WaveMamba
from util.datasets import OutputType
import torch.nn.functional as F


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The mimimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights,
                 eta_mins=[0.0003, 0.000001],
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        # print('idx',idx)
        # print('self.cumulative_period', self.cumulative_period)
        # print('self.restart_weights', self.restart_weights)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                    (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target)**2 + eps)


def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)


def gradient_loss(pred, target, mask, coh):

    # compute gradients
    dx_pred = pred[:,:, :,1:] - pred[:,:,:,:-1]
    dy_pred = pred[:,:,1:,:] - pred[:,:,:-1,:]

    dx_target = target[:,:, :,1:] - target[:,:,:,:-1]
    dy_target = target[:,:,1:,:] - target[:,:,:-1,:]

    # match mask sizes
    mask_x = mask[:,:,:,1:] * mask[:,:,:,:-1]
    mask_y = mask[:,:,1:,:] * mask[:,:,:-1,:]

    coh_x = coh[:,:,:,1:] * coh[:,:,:,:-1]
    coh_y = coh[:,:,1:,:] * coh[:,:,:-1,:]

    loss_x = torch.abs(dx_pred - dx_target) * mask_x * coh_x
    loss_y = torch.abs(dy_pred - dy_target) * mask_y * coh_y

    loss = (
        loss_x.sum() / (mask_x * coh_x).sum().clamp_min(1e-8)
        + loss_y.sum() / (mask_y * coh_y).sum().clamp_min(1e-8)
    )

    return loss


def local_max_loss(pred, target, mask, coh_w):

    pred_pool = torch.nn.functional.max_pool2d(pred, 9, stride=1, padding=4)
    target_pool = torch.nn.functional.max_pool2d(target, 9, stride=1, padding=4)

    loss = torch.abs(pred_pool - target_pool) * mask * coh_w
    return loss.sum() / (mask * coh_w).sum()


def dem_features_sobel(dem, add_mag=True, add_lap=False):
    B, C, H, W = dem.shape
    assert C == 1

    # Sobel kernels normalized so output roughly in [-1,1] for dem in [-1,1]
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=dem.device, dtype=dem.dtype).view(1,1,3,3) / 8.0
    sobel_y = sobel_x.transpose(2, 3)

    dx = F.conv2d(dem, sobel_x, padding=1)
    dy = F.conv2d(dem, sobel_y, padding=1)

    feats = [dem, dx, dy]

    if add_mag:
        mag = torch.tanh(torch.sqrt(dx * dx + dy * dy + 1e-12))
        feats.append(mag)

    if add_lap:
        lap_k = torch.tensor([[0, 1, 0],
                              [1,-4, 1],
                              [0, 1, 0]], device=dem.device, dtype=dem.dtype).view(1,1,3,3) / 4.0
        lap = F.conv2d(dem, lap_k, padding=1)
        lap = torch.tanh(lap)
        feats.append(lap)

    return torch.cat(feats, dim=1)


class WaveMambaTrainer(GenericTrainer):
    def __init__(self, args, train_loader, test_loader, summary_writer, logger=None):
        super(WaveMambaTrainer, self).__init__(args, train_loader, test_loader, summary_writer, logger)
        self.name = 'WaveMamba'
        self.use_dem = False

        self.model = self.build_model(args)

        print(f'Model with params: {self.model.call_args}')

        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-3, betas=(0.9, 0.99))
        self.l1 = nn.L1Loss()
        self.cri_fft = FFTLoss(loss_weight=0.1)

        self.unreduced_l1 = nn.L1Loss(reduction='none')

        self.scheduler = CosineAnnealingRestartCyclicLR(self.optimizer, [100, 100000],
                                                        [1, 1], [0.0005, 0.0000001])

        self.optimizers = [self.optimizer]
        self.schedulers = [self.scheduler]

        self.use_fft = True

    def build_model(self, args):
        # WaveDinoTrainer overrides this (and sets its name and flags there)
        return create_object(
            WaveMamba,
            in_chn=1,
            out_chn=1,
            wf=32,
            n_l_blocks=[1, 2, 4],
            n_h_blocks=[1, 1, 2],
            ffn_scale=2.0,
        )

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def prepare_train(self):
        pass

    def get_dem_input(self, dem):
        if self.use_dem:
            dx = dem[:, :, :, 1:] - dem[:, :, :, :-1]
            dy = dem[:, :, 1:, :] - dem[:, :, :-1, :]

            dx = F.pad(dx, (0, 1, 0, 0))
            dy = F.pad(dy, (0, 0, 0, 1))

            dem_input = torch.cat([dem, dx, dy], dim=1)
        else:
            dem_input = None
        return dem_input

    def train_step(self, data_dict) -> torch.Tensor:
        self.update_learning_rate(self.iterations, -1)

        self.optimizer.zero_grad()

        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        deform = data_dict[OutputType.DEFORMATION]
        mask = data_dict[OutputType.WATER_MASK]

        dem_input = self.get_dem_input(dem)
        # dem_input = dem_features_sobel(dem, add_mag=False, add_lap=False)

        deform_pred = self.model(image, dem_input)
        # loss = self.l1(deform_pred, deform)
        l1 = charbonnier(deform_pred, deform)
        l1 = masked_mean(l1, mask)
        loss = l1 + self.cri_fft(deform_pred * mask, deform * mask)

        loss.backward()
        self.optimizer.step()

        return loss

    def test_step(self, data_dict) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            image = data_dict[OutputType.IMAGE]
            dem = data_dict[OutputType.DEM]

            dem_input = self.get_dem_input(dem)
            # dem_input = dem_features_sobel(dem, add_mag=False, add_lap=False)

            out = self.model(image, dem_input)
        return out
