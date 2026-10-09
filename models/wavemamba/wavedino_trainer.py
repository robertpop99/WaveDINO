"""
WaveDino trainer: WaveMamba with frozen DINOv3 features, DEM gradients, water mask / distance input channels and a
low-order ramp head (phase ramp head, PRH). The network predicts the atmosphere; deformation = image - atmosphere.

Model-type suffixes remove one component for the ablation study (Table III): -water, -dem, -ramp, -dino.
The paper model is wavedino-4layers (-4layers = the 3-level variant).
"""
import torch

from models.dino import Dino
from models.generic_trainer import create_object
from models.loss_functions import SimpleFFTLoss
from models.wavemamba.mamba_trainer import WaveMambaTrainer, charbonnier, masked_mean, gradient_loss, local_max_loss
from models.wavemamba.wavemamba import WaveDino
from util.datasets import OutputType, get_day_interval, DEFAULT_DAY_INTERVAL


class WaveDinoTrainer(WaveMambaTrainer):
    def __init__(self, args, train_loader, test_loader, summary_writer, logger=None):
        super(WaveDinoTrainer, self).__init__(args, train_loader, test_loader, summary_writer, logger)
        # self.atm_fft = AtmFFTLoss(loss_weight=0.01)
        self.atm_fft = SimpleFFTLoss(loss_weight=0.1)

    def build_model(self, args):
        self.use_dino = True
        self.use_dem = True
        self.water_input = True
        self.use_ramp_head = True
        self.use_4_layers = True

        self.name = 'WaveDino'

        self.dino = Dino()
        self.dino.to(self.device)
        in_channels = 3

        if '-dino' in args.model_type:
            self.use_dino = False
            self.name += '-dino'
        if '-dem' in args.model_type:
            self.use_dem = False
            self.name += '-dem'
        if '-water' in args.model_type:
            self.water_input = False
            in_channels = 1
            self.name += '-water'
        if '-ramp' in args.model_type:
            self.use_ramp_head = False
            self.name += '-ramp'
        if '-4layers' in args.model_type:
            self.use_4_layers = False
            self.name += '-4layers'
        if get_day_interval(args) != DEFAULT_DAY_INTERVAL:
            # stricter temporal-baseline cutoff, applied in util/datasets.py
            self.name += f'-{get_day_interval(args) - 1}d'

        return create_object(
            WaveDino,
            in_chn=in_channels,
            out_chn=1,
            wf=32,
            n_l_blocks=[1, 2, 4],
            n_h_blocks=[1, 1, 2],
            ffn_scale=2.0,
            use_dem=self.use_dem,
            use_dino=self.use_dino,
            use_ramp_head=self.use_ramp_head,
            use_4_layers=self.use_4_layers,
        )

    def ramp_loss(self, ramp):
        lap = (
                ramp[:, :, 2:, 1:-1] +
                ramp[:, :, :-2, 1:-1] +
                ramp[:, :, 1:-1, 2:] +
                ramp[:, :, 1:-1, :-2]
                - 4 * ramp[:, :, 1:-1, 1:-1]
        )
        loss_lap = torch.mean(lap ** 2)
        return loss_lap

    def dino_features(self, image, mask):
        return self.dino(image)

    def train_step(self, data_dict) -> torch.Tensor:
        self.update_learning_rate(self.iterations, -1)

        self.optimizer.zero_grad()

        image = data_dict[OutputType.IMAGE]
        dem = data_dict[OutputType.DEM]
        orig_img = data_dict[OutputType.ORIG_IMG]
        deform = data_dict[OutputType.DEFORMATION]
        mask = data_dict[OutputType.WATER_MASK]
        dist = data_dict[OutputType.WATER_DISTANCE]
        coh = data_dict[OutputType.COHERENCE]

        dem_input = self.get_dem_input(dem)
        # dem_input = dem_features_sobel(dem, add_mag=False, add_lap=False)

        if self.use_dino:
            hidden = self.dino_features(image, mask)
        else:
            hidden = None

        if self.water_input:
            image_input = torch.cat([image, mask, dist], dim=1)
        else:
            image_input = image
        atm_pred, ramp = self.model(image_input, hidden, dem_input)
        deform_pred = image - atm_pred

        # coh_w = 0.4 + 0.6 * coh
        water_w = mask
        coh_w = torch.ones_like(water_w)
        # water_w = torch.ones_like(water_w)

        valid_w = water_w * coh_w

        error = torch.abs(deform_pred - deform)
        weight = ((1 + torch.abs(deform)) ** 3) * ((1 + error) ** 2 - 1) + 1e-8

        l1 = charbonnier(deform_pred, deform)
        l1 = l1 * weight
        l1 = masked_mean(l1 * coh_w, water_w)

        l2 = (deform_pred - deform) ** 2
        l2 = l2 * weight
        l2 = masked_mean(l2 * coh_w, water_w)

        grad_loss = gradient_loss(deform_pred, deform, water_w, coh_w)
        loc_max_loss = local_max_loss(deform_pred, deform, water_w, coh_w)

        lap_loss = 0.
        if self.checkpoint_loaded:
            if self.use_ramp_head:
                lap_loss = self.ramp_loss(ramp)
            self.atm_fft.loss_weight = 0.1
            fft_loss = self.atm_fft(deform_pred, deform, valid_w)
        else:
            if self.iterations > 200:
                if self.iterations < 300:
                    self.atm_fft.loss_weight = 0.01
                elif self.iterations < 500:
                    self.atm_fft.loss_weight = 0.05
                else:
                    self.atm_fft.loss_weight = 0.1

                fft_loss = self.atm_fft(deform_pred, deform, valid_w)
                if self.use_ramp_head:
                    lap_loss = self.ramp_loss(ramp)
            else:
                fft_loss = 0.
                lap_loss = 0.

        loss = (
            1. * l1 +
            1. * l2 +
            # 0.5 * grad_loss +
            # 0.2 * loc_max_loss +
            1 * grad_loss +
            1 * loc_max_loss +
            0.05 * lap_loss +
            1 * fft_loss
        )

        # if self.iterations % 10 == 0:
        #     ramp_energy = (ramp ** 2).mean()
        #     atm_energy = ((atm_pred - ramp) ** 2).mean()
        #
        #     ratio_energy = ramp_energy / (atm_energy + 1e-6)
        #     print(f'Energy ratio (ideal 0.1-0.3): {ratio_energy}')

        loss.backward()
        self.optimizer.step()

        return loss

    def test_step(self, data_dict) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            image = data_dict[OutputType.IMAGE]
            dem = data_dict[OutputType.DEM]
            mask = data_dict[OutputType.WATER_MASK]
            dist = data_dict[OutputType.WATER_DISTANCE]
            coh = data_dict[OutputType.COHERENCE]

            dem_input = self.get_dem_input(dem)
            # dem_input = dem_features_sobel(dem, add_mag=False, add_lap=False)

            if self.use_dino:
                hidden = self.dino_features(image, mask)
            else:
                hidden = None

            if self.water_input:
                image_input = torch.cat([image, mask, dist], dim=1)
            else:
                image_input = image

            atm_pred, _ = self.model(image_input, hidden, dem_input)
            out = image - atm_pred
        return out

    def debug_step(self, data_dict):
        self.model.eval()
        with torch.no_grad():
            image = data_dict[OutputType.IMAGE]
            dem = data_dict[OutputType.DEM]
            mask = data_dict[OutputType.WATER_MASK]
            dist = data_dict[OutputType.WATER_DISTANCE]

            dem_input = self.get_dem_input(dem)
            # dem_input = dem_features_sobel(dem, add_mag=True, add_lap=False)

            hidden = self.dino_features(image, mask)
            image_input = torch.cat([image, mask, dist], dim=1)
            atm_pred, ramp = self.model(image_input, hidden, dem_input)
            out = image - atm_pred
            return out, atm_pred, ramp
