import random

import torch
from torch import optim, nn
from torch.nn import L1Loss

from models.generic_trainer import GenericTrainer, create_object
from models.restormer.restormer import Restormer
from util.datasets import OutputType


class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)


class RestormerTrainer(GenericTrainer):
    def __init__(self, args, train_loader, test_loader, summary_writer, logger=None):
        super().__init__(args, train_loader, test_loader, summary_writer, logger)
        self.name = 'Restormer'

        self.use_grad_clipping = True
        self.grad_clip = 0.01
        self.mixing_flag = True
        self.mixing_augmentation = Mixing_Augment(1.2, True, self.device)

        # gray restormer
        self.model = create_object(
            Restormer,
            inp_channels=1,
            out_channels=1,
            dim=48,
            num_blocks=[4,6,6,8],
            num_refinement_blocks=4,
            heads=[1,2,4,8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='BiasFree',  ## Other option 'BiasFree'
            dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        )
        print(f'Model with params: {self.model.call_args}')

        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999))
        self.loss_function = L1Loss()
        self.unreduced_loss_function = nn.L1Loss(reduction='none')

    def prepare_train(self):
        pass

    def train_step(self, data_dict) -> torch.Tensor:
        self.optimizer.zero_grad()

        image = data_dict[OutputType.IMAGE]
        deform = data_dict[OutputType.DEFORMATION]
        mask = data_dict[OutputType.WATER_MASK]

        if self.mixing_flag:
            deform, image = self.mixing_augmentation(deform, image)

        preds = self.model(image)

        loss = self.unreduced_loss_function(preds, deform)
        loss = masked_mean(loss, mask)

        loss.backward()
        if self.use_grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss

    def test_step(self, data_dict) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            out = self.model(data_dict[OutputType.IMAGE])
        return out
