import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from models.generic_trainer import GenericTrainer, create_object
from models.loss_functions import CharbonnierLoss
from .network_swinir import SwinIR

from util.datasets import OutputType


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target)**2 + eps)


def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)


class SwinIRTrainer(GenericTrainer):
    def __init__(self, args, train_loader, test_loader, summary_writer, logger=None):
        super(SwinIRTrainer, self).__init__(args, train_loader, test_loader, summary_writer, logger)
        self.name = 'SwinIR'

        # gray image
        self.model = create_object(
            SwinIR,
            upscale=1,
            in_chans=1,
            out_chans=1,
            img_size=self.image_size,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv',
            use_checkpoint=True,
        )
        print(f'Model with params: {self.model.call_args}')

        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4, weight_decay=0)
        self.loss_function = CharbonnierLoss(eps=1e-9)
        self.unreduced_loss_function = charbonnier

        self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                     milestones=[15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000], gamma=0.5)

    def prepare_train(self):
        pass

    def train_step(self, data_dict) -> torch.Tensor:
        self.optimizer.zero_grad()

        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        orig_img = data_dict[OutputType.ORIG_IMG]
        deform = data_dict[OutputType.DEFORMATION]
        mask = data_dict[OutputType.WATER_MASK]

        # x = torch.cat([image, dem], dim=1)
        x = image

        out = self.model(x, add_x=True)
        # out_deform, out_ts = torch.split(out, 1, dim=1)

        # loss1 = self.loss_function(out, deform)
        loss1 = self.unreduced_loss_function(out, deform)
        loss1 = masked_mean(loss1, mask)
        # loss2 = self.loss_function(out_ts, orig_img)
        # loss3 = self.loss_function(image, out_deform + out_ts)
        # loss4 = self.loss_function(image - out_ts, deform)
        #

        loss = loss1

        # unreduced loss
        # unreduced_loss = self.unreduced_loss_function(preds, out)
        # weight = 2 * torch.abs(ground_truth) + 0.001
        # unreduced_loss = weight * unreduced_loss
        # loss = unreduced_loss.mean()

        # pixel loss

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def denoise_image(self, image, annotations, dem, steps=-1) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            # x = torch.cat([image, dem], dim=1)
            x = image
            out = self.model(x, add_x=True)
            # out_deform, out_ts = torch.split(out, 1, dim=1)
            out_deform = out
        return out_deform

    def prepare_test(self):
        pass

    def test_step(self, data_dict) -> torch.Tensor:
        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        annotations = data_dict[OutputType.ANNOTATION]
        return self.denoise_image(image, annotations, dem)



class SwinIR2ChannelsTrainer(GenericTrainer):
    def __init__(self, args, train_loader, test_loader, summary_writer, logger=None):
        super(SwinIR2ChannelsTrainer, self).__init__(args, train_loader, test_loader, summary_writer, logger)
        self.name = 'SwinIR2Channels'

        # gray image
        self.model = create_object(
            SwinIR,
            upscale=1,
            in_chans=1,
            out_chans=2,
            img_size=self.image_size,
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='',
            resi_connection='1conv',
            use_checkpoint=True,
        )
        print(f'Model with params: {self.model.call_args}')

        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-4, weight_decay=0)
        self.loss_function = CharbonnierLoss(eps=1e-9)
        self.unreduced_loss_function = None

        self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                     milestones=[15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000], gamma=0.5)

    def prepare_train(self):
        pass

    def train_step(self, data_dict) -> torch.Tensor:
        self.optimizer.zero_grad()

        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        orig_img = data_dict[OutputType.ORIG_IMG]
        deform = data_dict[OutputType.DEFORMATION]

        # x = torch.cat([image, dem], dim=1)
        x = image

        out = self.model(x, add_x=False)
        out_deform, out_ts = torch.split(out, 1, dim=1)

        loss1 = self.loss_function(out_ts, orig_img)
        loss2 = self.loss_function(deform, out_deform)
        loss3 = self.loss_function(image, out_deform + out_ts)

        loss = loss1 + loss2 + loss3

        # unreduced loss
        # unreduced_loss = self.unreduced_loss_function(preds, out)
        # weight = 2 * torch.abs(ground_truth) + 0.001
        # unreduced_loss = weight * unreduced_loss
        # loss = unreduced_loss.mean()

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss

    def test_step(self, data_dict) -> torch.Tensor:
        image = data_dict[OutputType.IMAGE]

        self.model.eval()
        with torch.no_grad():
            # x = torch.cat([image, dem], dim=1)
            x = image
            out = self.model(x, add_x=False)
            out_deform, out_ts = torch.split(out, 1, dim=1)
        return out_deform

