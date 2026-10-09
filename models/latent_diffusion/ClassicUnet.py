import torch
from torch import optim, nn

from models.generic_trainer import GenericTrainer, create_object
from models.latent_diffusion.openaimodel import UNetModel
from models.loss_functions import CharbonnierLoss
from util.datasets import OutputType


def charbonnier(pred, target, eps=1e-3):
    return torch.sqrt((pred - target)**2 + eps)


def masked_mean(x, mask, eps=1e-8):
    return (x * mask).sum() / (mask.sum() + eps)


class ClassicUnetTrainer(GenericTrainer):
    def __init__(self, args, train_loader, test_loader, summary_writer, logger=None):
        super().__init__(args, train_loader, test_loader, summary_writer, logger)
        self.name = 'ClassicUnet'

        self.model = create_object(
            UNetModel,
            image_size=self.image_size,
            in_channels=1,
            model_channels=128,
            out_channels=1,
            # num_res_blocks=4,
            num_res_blocks=3,
            # attention_resolutions=[1, 2, 4, 8],
            attention_resolutions=[2, 4, 8],
            dropout=0,
            # channel_mult=(1, 2, 3, 4),
            channel_mult=(1, 2, 3, 4),
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
            context_dim=None,
            n_embed=None,
            legacy=False,
            use_downsample_block=False,
        )
        print(f'Model with params: {self.model.call_args}')


        self.optimizer = optim.AdamW(self.model.parameters(), 1e-4, weight_decay=25 * 1e-5)

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.charb = CharbonnierLoss()

        self.unreduced_l1 = nn.L1Loss(reduction='none')
        self.unreduced_mse = nn.MSELoss(reduction='none')
        self.unreduced_charb = charbonnier

    def train_step(self, data_dict):
        self.optimizer.zero_grad()

        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        orig_img = data_dict[OutputType.ORIG_IMG]
        deform = data_dict[OutputType.DEFORMATION]
        annotations = data_dict[OutputType.ANNOTATION]
        image_time_length = data_dict[OutputType.TIME_LENGTH]
        mask = data_dict[OutputType.WATER_MASK]

        ts = torch.randint(low=1, high=10, size=(image.shape[0],)).to(self.device)
        for i in range(image.shape[0]):
            ts[i] = int(image_time_length[i])

        out = self.model(image, ts)

        out_deform = image - out

        # loss = self.mse(out_deform, deform) + self.l1(out_deform, deform) + self.charb(out_deform, deform)

        mse_loss = self.unreduced_mse(out_deform, deform)
        mse_loss = masked_mean(mse_loss, mask)

        l1_loss = self.unreduced_l1(out_deform, deform)
        l1_loss = masked_mean(l1_loss, mask)

        charb_loss = self.unreduced_charb(out_deform, deform)
        charb_loss = masked_mean(charb_loss, mask)

        loss = mse_loss + l1_loss + charb_loss

        loss.backward()
        self.optimizer.step()
        return loss

    def test_step(self, data_dict) -> torch.Tensor:
        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        annotations = data_dict[OutputType.ANNOTATION]
        image_time_length = data_dict[OutputType.TIME_LENGTH]

        self.model.eval()
        with torch.no_grad():
            ts = torch.randint(low=1, high=10, size=(image.shape[0],)).to(self.device)
            for i in range(image.shape[0]):
                ts[i] = int(image_time_length[i])

            out = self.model(image, ts)
            out_deform = image - out
        return out_deform

