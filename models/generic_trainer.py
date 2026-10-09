import json
import math
import os
import random
import time
from typing import Optional, Literal

import numpy as np
import torch
import torchvision.utils
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from util.datasets import OutputType, rad_to_mm
import tifffile as tiff
from piqa import SSIM, PSNR


class LPLoss(nn.Module):
    def __init__(self, p=3):
        super().__init__()
        self.p = p

    def forward(self, img, target):
        dist = torch.dist(img, target, self.p)
        nr = img.numel()
        return dist / nr


class LPIPS(nn.Module):
    def __init__(self, net_type: Literal["alex", "vgg", "squeeze"] = "alex"):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)

    def forward(self, img: torch.Tensor, target: torch.Tensor):
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img * 2 - 1

        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        target = target * 2 - 1

        if img.shape[1] == 1:
            img = torch.cat([img, img, img], dim=1)
            target = torch.cat([target, target, target], dim=1)

        return self.lpips(img, target)


class SSIMGray(SSIM):
    def forward(self, img, target):
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)

        if img.shape[1] == 1:
            img = torch.cat([img, img, img], dim=1)
            target = torch.cat([target, target, target], dim=1)

        return super().forward(img, target)


class PSNRGray(PSNR):
    def forward(self, img, target):
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)

        return super().forward(img, target)


class ImageQualityMetrics(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        self.methods = {
            'MSE': nn.MSELoss(),
            'LPIPS': LPIPS().to(device),
            'SSIM': SSIMGray().to(device),
            'PSNR': PSNRGray().to(device),
            'L3': LPLoss(3),
            # 'L1': LPLoss(1),
            'ABS': lambda x,y: torch.mean(torch.abs(x - y)),
        }
        self.saved_results = {}
        for key in self.methods:
            self.saved_results[key] = torch.empty(0)

    def forward(self, img, target):
        for index in range(img.shape[0]):
            i = img[index].unsqueeze(0)
            t = target[index].unsqueeze(0)

            for key in self.methods:
                res = self.methods[key](i, t)
                res = res.view(1)
                self.saved_results[key] = torch.cat((self.saved_results[key], res.cpu()))

    def return_results(self):
        return self.saved_results

    def print_results(self):
        row = ''
        print('=======> Results for Image Quality Metrics:', flush=True)
        for key in self.saved_results:
            res = self.saved_results[key]
            print(f'{key}: mean {res.mean()} std {res.std()} size {res.shape}', flush=True)
            if 'L3' in key:
                row += f'{res.mean():.4f} '
            else:
                row += f'{res.mean():.2f} '
        print(f'=======> Excel friendly row: {row}', flush=True)

class DistanceQualityMetrics(nn.Module):
    def __init__(self, filters, samples_per_image=100):
        super().__init__()
        self.filters = filters
        self.samples_per_image = samples_per_image

        self.methods = {
            'MSE': nn.MSELoss(),
            'ABS': lambda x,y: torch.abs(x - y),
            # 'L3': LPLoss(3),
            # 'L1': LPLoss(1),
        }
        self.test_results = {}

        for key in self.methods:
            self.test_results[key] = {}

            for filt1, filt2 in filters:
                self.test_results[key][f'{filt1}_{filt2}'] = torch.empty(0)

    def forward(self, img, target):
        samples_i = {}
        samples_t = []
        for key in self.methods:
            samples_i[key] = []

        for index in range(img.shape[0]):
            for j in range(self.samples_per_image):
                x = torch.randint(img.shape[-1], (1,))
                y = torch.randint(img.shape[-2], (1,))

                i_pixel = img[index, 0, y[0], x[0]]
                t_pixel = target[index, 0, y[0], x[0]]
                samples_t.append(t_pixel)

                for key in self.methods:
                    samples_i[key].append(self.methods[key](i_pixel, t_pixel).cpu())

        samples_t = torch.tensor(samples_t)
        samples_t = torch.abs(samples_t)
        for key in samples_i:
            samples_i[key] = torch.tensor(samples_i[key])

        for filt1, filt2 in self.filters:
            mask = (samples_t < filt1) & (samples_t > filt2)
            if mask.sum() == 0:
                continue

            for key in self.test_results:
                self.test_results[key][f'{filt1}_{filt2}'] = torch.cat(
                    (self.test_results[key][f'{filt1}_{filt2}'], samples_i[key][mask].cpu())
                )

    def return_results(self):
        return self.test_results

    def print_results(self):
        print('=======> Results for Distance Quality Metrics:', flush=True)
        for key in self.test_results:
            row = ''
            for filt1, filt2 in self.filters:
                res = self.test_results[key][f'{filt1}_{filt2}']
                print(f'{key} ({filt1}_{filt2}): mean {res.mean()} std {res.std()} size {res.shape}', flush=True)
                row += f'{res.mean():.2f} '
            print(f'=======> Excel friendly row: {row}', flush=True)


class GenericTrainer:
    def __init__(self, args, datasets_dir, test_loader, summary_writer: Optional[SummaryWriter]=None, logger=None):
        if torch.cuda.is_available() and not args.no_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f'Running on device: {self.device}', flush=True)

        self.train_loader = datasets_dir['train']
        self.validate_loader = datasets_dir['validate']
        self.test_loader = datasets_dir['test']
        self.logger = logger if logger is not None else {'info': ''}
        self.summary_writer = summary_writer
        self.name = 'GenericModel'
        self.start_time = time.strftime('%y_%m_%d_%H_%M_%S')

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.in_channels = args.num_channels
        self.out_channels = self.in_channels
        self.log_frequency = args.log_frequency
        self.save_frequency = args.save_frequency
        self.test_frequency = args.test_frequency
        self.result_dir = os.path.join(args.result_dir, f'{args.dataset}_{args.image_size}')
        self.image_size = args.image_size
        self.model_type = args.model_type
        self.no_heatmap = args.no_heatmap
        self.debug = args.debug
        self.metadata = args.metadata
        self.evaluate = getattr(args, 'evaluate', False)

        # subclasses build these; load_model re-points / restores them
        self.optimizer = None
        self.scheduler = None
        self.model = torch.nn.ModuleDict({})

        self.best_average_loss = torch.inf
        self.current_average_loss = torch.inf
        self.best_validation_loss = torch.inf
        self.current_validation_loss = torch.inf
        self.best_validation_epoch = 0
        self.best_validation_iterations = 0
        self.best_model_epoch = 0
        self.best_model_iterations = 0
        # epochs already completed by a previous job in a checkpoint chain, so this
        # job's epochs are numbered after them rather than restarting at 1.
        self.epoch_offset = 0


        self.print_all_epochs = True
        self.end_epochs_to_print = 0


        # norm_dist = torch.distributions.normal.Normal(self.image_size / 2, self.image_size / 6)
        # a = torch.exp(norm_dist.log_prob(torch.arange(self.image_size))).view(-1, 1)
        # b = torch.exp(norm_dist.log_prob(torch.arange(self.image_size))).view(1, -1)
        #
        # wmap = torch.matmul(a, b)
        # wmap = wmap / wmap.sum()
        # self.wmap = wmap.to(self.device)
        norm_dist = torch.distributions.normal.Normal(self.image_size / 2, self.image_size / 6)
        coords = torch.arange(self.image_size, device=self.device)

        a = torch.exp(norm_dist.log_prob(coords)).view(-1, 1)
        b = torch.exp(norm_dist.log_prob(coords)).view(1, -1)

        wmap = torch.matmul(a, b)
        wmap = wmap / wmap.sum()
        wmap = wmap.view(1, self.image_size * self.image_size, 1)  # (1, P, 1)
        self.wmap = wmap.to(self.device)


        self.test_kernel_size = self.image_size
        self.test_stride = int(self.image_size / 4)

        self.threshold_percentage = 0.95

        self.prepared_for_train = False
        self.prepared_for_test = False

        self.test_metrics = []
        self.test_results = {}

        self.last_finished_epoch = 0
        self.iterations = 0

        self.checkpoint_loaded = False
        self.checkpoint_path = None
        self.loaded_best = False

    def move_batch_to_device(self, data_dict):
        for t in OutputType:
            if t in data_dict and type(data_dict[t]) is torch.Tensor:
                data_dict[t] = data_dict[t].to(self.device)

        return data_dict

    def train(self):
        self.model.to(self.device)
        self.prepare_train()

        self.summary_writer.add_scalars("Time/epoch", {'time_elapsed': 0,
                                                       'average_batch_time': 0}, 0)

        if self.checkpoint_loaded and self.loaded_best:
            # The restored best validation loss is carried over (restore_run_state), so this folder only gets a
            # model_best.pt if the job beats it. Seed it with the loaded best so the folder always holds the chain's
            # best, and the next --model-name last (resume or --evaluate) does not fall back to model.pt.
            self.save_model(True)

        # --epochs is "this many more epochs", numbered after whatever a resumed
        # checkpoint had already finished (0 for a fresh run).
        for epoch in range(self.epoch_offset + 1, self.epoch_offset + self.epochs + 1):
            self.model.train()
            torch.cuda.empty_cache()
            epoch_iter = 0
            train_loss = 0

            start_time = time.time()

            for batch_idx, data_dict in enumerate(self.train_loader):
                # print(batch_idx, flush=True)
                data_dict = self.move_batch_to_device(data_dict)
                data_load_end_time = time.time()

                # zero_percentage = ((batch <= 0.01) & (batch >= 0.01)).sum(dim=(1, 2, 3)) / (batch.shape[1] * batch.shape[2] * batch.shape[3])
                # mask = zero_percentage < 0.6
                # if mask.sum() == 0:
                #     continue
                # else:
                #     batch, ground_truth, dem = batch[mask], ground_truth[mask], dem[mask]

                loss = self.train_step(data_dict)

                self.iterations += 1
                epoch_iter += 1
                train_loss += loss.item()
                data_load_time = data_load_end_time - start_time
                step_time = time.time() - data_load_end_time
                self.log_metrics(epoch, loss / self.batch_size, data_load_time, step_time)

                if epoch_iter % self.log_frequency == 0:
                    print(f'==> Batch: {epoch_iter} / {len(self.train_loader)} Average loss: {loss.item() / self.batch_size:.6f}',
                        flush=True)

                if self.iterations % self.save_frequency == 0:
                    self.validate()
                    self.save_model(False)

            self.last_finished_epoch = epoch

            time_elapsed = time.time() - start_time
            print(f'Epoch finished in: {time_elapsed} Average time per batch: {time_elapsed / len(self.train_loader)}', flush=True)
            self.summary_writer.add_scalars("Time/epoch", {'time elapsed': time_elapsed,
                                                          'average batch time': time_elapsed / len(self.train_loader)}, epoch)

            if self.print_all_epochs or epoch >= self.epoch_offset + self.epochs + 1 - self.end_epochs_to_print:
                self.current_average_loss = train_loss / len(self.train_loader.dataset)
                print(f'====> Epoch: {epoch} Average loss: {self.current_average_loss:.6f}', flush=True)
                # print("\033[F\033[K", end="")
                if self.current_average_loss < self.best_average_loss:
                    self.best_average_loss = self.current_average_loss
                    self.best_model_epoch = self.last_finished_epoch
                    self.best_model_iterations = self.iterations

        self.validate()
        self.save_model(False)

    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("Epoch", epoch, self.iterations)
        self.summary_writer.add_scalars("Loss",
                {"train": float(loss.item())},
                self.iterations
        )
        self.summary_writer.add_scalars("Time/data",
                                        {'data_load_time': data_load_time,
                                        'step_time': step_time}, self.iterations
        )

    def log_images(self, input_: torch.Tensor, output: torch.Tensor, target: torch.Tensor):
        if not input_.shape[1] == output.shape[1] == target.shape[1]:
            input_ = input_[:, 0].view(input_.shape[0], 1, *input_.shape[2:])
            output = output[:, 0].view(output.shape[0], 1, *output.shape[2:])
            target = target[:, 0].view(target.shape[0], 1, *target.shape[2:])

        batch_size = 4 if output.shape[0] >= 4 else output.shape[0]
        img_batch = torch.zeros((batch_size * 3, output.shape[1], *output.shape[2:]))
        for i in range(batch_size):
            img_batch[3 * i + 0] = input_[i]
            img_batch[3 * i + 1] = output[i]
            img_batch[3 * i + 2] = target[i]

        self.summary_writer.add_images('Images/samples/train', img_batch, self.iterations)

    def prepare_train(self):
        pass

    def train_step(self, data_dict) -> torch.Tensor:
        pass

    def validate(self):
        self.model.eval()
        validate_limit = self.save_frequency // 5
        counter = 0
        mse = nn.MSELoss()
        average_loss = 0

        print(f'====> Validating for {validate_limit} batches', flush=True)
        with torch.no_grad():
            for batch_inx, data_dict in enumerate(self.validate_loader):
                # print(batch_inx, flush=True)
                data_dict = self.move_batch_to_device(data_dict)

                deform = data_dict[OutputType.DEFORMATION]
                result = self.test_step(data_dict)

                result = result[:, 0, :, :]
                result = result.view(result.shape[0], 1, result.shape[1], result.shape[2])

                average_loss += mse(deform, result).item()

                counter += 1
                if counter >= validate_limit:
                    break

        self.model.train()
        average_loss = average_loss / counter
        self.current_validation_loss = average_loss

        if self.current_validation_loss < self.best_validation_loss:
            self.best_validation_loss = self.current_validation_loss
            self.best_validation_epoch = self.last_finished_epoch + 1 # current epoch
            self.best_validation_iterations = self.iterations
            self.save_model(True)

    def test(self, test_limit=-1, args=None):
        self.model.to(self.device)
        self.model.eval()
        self.prepare_test()

        if test_limit < 0:
            test_limit = len(self.test_loader.dataset)

        filters = [(np.inf, 0), (np.inf, 90), (90, 70), (70, 50), (50, 30), (np.inf, 30), (30, 20), (20, 10), (10, 0)]
        filters.reverse()

        test_results = {'all_time_length': [], 'all_name': [], 'IQM': {}, 'DQM': {}}

        iqm = ImageQualityMetrics(device=self.device)
        dqm = DistanceQualityMetrics(filters=filters)

        test_counter = 0
        save_counter = 0

        torch.manual_seed(10)
        np.random.seed(10)
        random.seed(10)

        print('Testing on synthetic data', flush=True)
        with (torch.no_grad()):
            # print('No grad', flush=True)
            while test_counter < test_limit:
                # print(f'{test_counter}/{test_limit}', flush=True)
                for batch_inx, data_dict in enumerate(self.test_loader):
                    # print(f'Batch {batch_inx} Test counter {test_counter}', flush=True)
                    data_dict = self.move_batch_to_device(data_dict)

                    test_results['all_name'].extend(data_dict[OutputType.NAME])
                    test_results['all_time_length'].extend(data_dict[OutputType.TIME_LENGTH])

                    deform = data_dict[OutputType.DEFORMATION]
                    image = data_dict[OutputType.IMAGE]
                    result = self.test_step(data_dict)

                    result = result[:, 0, :, :]
                    result = result.view(result.shape[0], 1, result.shape[1], result.shape[2])

                    result = result * 100
                    result = rad_to_mm(result)

                    deform = deform * 100
                    deform = rad_to_mm(deform)

                    image = image * 100
                    image = rad_to_mm(image)

                    iqm(result, deform)
                    dqm(result, deform)

                    if args is not None:
                        folder = args.model_name
                        for i in range(data_dict[OutputType.IMAGE].shape[0]):
                            tiff.imwrite(folder + f'_{save_counter}_deform.tif', deform[i, 0].cpu().numpy())
                            tiff.imwrite(folder + f'_{save_counter}_img.tif', image[i, 0].cpu().numpy())
                            tiff.imwrite(folder + f'_{save_counter}_out.tif', result[i, 0].cpu().numpy())
                            save_counter += 1

                    test_counter += data_dict[OutputType.IMAGE].shape[0]
                    if (batch_inx + 1) % self.log_frequency == 0:
                        print(f'==> Batch: {batch_inx + 1} / {len(self.test_loader)} Finished samples: {test_counter}', flush=True)
                    if test_counter >= test_limit:
                        break

        iqm.print_results()
        dqm.print_results()

        print('\n\n\n')
        test_results['IQM'] = iqm.return_results()
        print('\n\n\n')
        test_results['DQM'] = dqm.return_results()
        print('\n\n\n')

        return test_results

    def get_deformation(self, data):

        image = data[OutputType.IMAGE]
        dem = data[OutputType.DEM]
        annotations = data[OutputType.ANNOTATION]
        time_length = data[OutputType.TIME_LENGTH]
        mask = data[OutputType.WATER_MASK]
        dist = data[OutputType.WATER_DISTANCE]
        coherence = data[OutputType.COHERENCE]

        B, C, H, W = image.shape

        patch = self.image_size
        stride = patch // 4
        batch_size = self.batch_size

        assert H >= patch and W >= patch, f'Image {H}x{W} is smaller than the patch size {patch}'

        # -----------------------------
        # Window positions: regular stride, plus a last window flush with the border
        # (nnU-Net style), so every pixel is covered without padding
        # -----------------------------
        def positions(size):
            pos = list(range(0, size - patch + 1, stride))
            if pos[-1] != size - patch:
                pos.append(size - patch)
            return pos

        windows = [(b, y, x) for b in range(B) for y in positions(H) for x in positions(W)]

        wmap = self.wmap.view(1, patch, patch)

        prob_map = torch.zeros(B, 1, H, W, device=image.device)
        norm_map = torch.zeros(B, 1, H, W, device=image.device)

        with torch.no_grad():

            for i in range(0, len(windows), batch_size):
                chunk = windows[i:i + batch_size]

                def extract(x):
                    return torch.stack([x[b, :, y:y + patch, x0:x0 + patch] for b, y, x0 in chunk])

                data_dict = {
                    OutputType.IMAGE: extract(image),
                    OutputType.DEM: extract(dem),
                    OutputType.WATER_MASK: extract(mask),
                    OutputType.WATER_DISTANCE: extract(dist),
                    OutputType.ANNOTATION: [annotations[b] for b, _, _ in chunk],
                    OutputType.TIME_LENGTH: [str(time_length[b]) for b, _, _ in chunk],
                    OutputType.COHERENCE: extract(coherence),
                }

                pred = self.test_step(data_dict)
                pred = pred.reshape(len(chunk), 1, patch, patch)

                # -----------------------------
                # Accumulate Gaussian-weighted patches and their weights
                # -----------------------------
                for (b, y, x), p in zip(chunk, pred):
                    prob_map[b, :, y:y + patch, x:x + patch] += p * wmap
                    norm_map[b, :, y:y + patch, x:x + patch] += wmap

        prob_map = prob_map / norm_map

        return prob_map

    def prepare_test(self):
        pass

    def test_step(self, data_dict) -> torch.Tensor:
        pass

    def save_model(self, save_best=False):
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)

        model_folder = f'{self.name}_network_{self.start_time}'
        if not os.path.isdir(os.path.join(self.result_dir, model_folder)):
            os.mkdir(os.path.join(self.result_dir, model_folder))

        if save_best:
            best_filename = f'model_best.pt'
            save_path = os.path.join(self.result_dir, model_folder, best_filename)
            print('Saving best model at: ' + save_path)
            torch.save(self.model.state_dict(), save_path)
            self.logger['model_name'] = save_path
        else:
            current_filename = f'model.pt'
            save_path = os.path.join(self.result_dir, model_folder, current_filename)
            print('Saving model at: ' + save_path)
            torch.save(self.model.state_dict(), save_path)
            self.logger['model_name'] = save_path

        # Optimizer and scheduler state next to the weights they belong to, so --load-checkpoint resumes AdamW's moments
        # and the LR schedule instead of restarting both cold.
        if self.optimizer is not None:
            opt_name = 'optimizer_best.pt' if save_best else 'optimizer.pt'
            torch.save(self.optimizer.state_dict(), os.path.join(self.result_dir, model_folder, opt_name))
        if self.scheduler is not None:
            sch_name = 'scheduler_best.pt' if save_best else 'scheduler.pt'
            torch.save(self.scheduler.state_dict(), os.path.join(self.result_dir, model_folder, sch_name))

        self.save_parameters()
        self.save_metadata()

    def save_parameters(self, save_path=None):
        if save_path is None:
            model_folder = f'{self.name}_network_{self.start_time}'
            filename = f'model_params.json'
            save_path = os.path.join(self.result_dir, model_folder, filename)

        # Convert and write JSON object to file
        with open(save_path, "w") as outfile:
            json.dump(self.model.call_args, outfile, indent=4, skipkeys=True)

    def save_metadata(self, save_path=None):
        if save_path is None:
            model_folder = f'{self.name}_network_{self.start_time}'
            filename = f'metadata.json'
            save_path = os.path.join(self.result_dir, model_folder, filename)

        metadata_dir = {
            'Using checkpoint': self.checkpoint_loaded,
            'Checkpoint used': self.checkpoint_path,
            'Last finished epoch': self.last_finished_epoch,
            'Last epoch average loss': self.current_average_loss,
            'Last finished iterations': self.iterations,
            'Current validation loss': self.current_validation_loss,
            'Best validation loss': self.best_validation_loss,
            'Best validation epoch': self.best_validation_epoch,
            'Best validation iterations': self.best_validation_iterations,
            'Best epoch': self.best_model_epoch,
            'Best epoch average loss': self.best_average_loss,
            'Best epoch iterations': self.best_model_iterations,
            'Comments': {},
        }

        comments = {}
        metadata_elems = self.metadata.split(';')
        for elem in metadata_elems:
            if elem.strip() == '':
                continue
            key, value = elem.split(':')
            comments[key.strip()] = value.strip()
        metadata_dir['Comments'] = comments

        # Convert and write JSON object to file
        with open(save_path, "w") as outfile:
            json.dump(metadata_dir, outfile, indent=4, skipkeys=True)

    def find_last_saved_model(self):
        folder = self.result_dir
        model_folder = f'{self.name}_network_'

        models = os.listdir(folder)
        models = np.sort(models) # sort by name and date of creation
        models = np.flip(models) # so that the latest model is first

        for model in models:
            if model_folder in model:
                return os.path.join(folder, model)
        raise FileNotFoundError

    def load_model(self, model_name: str, load_best_model=True):
        if model_name == '' or model_name == 'last':
            print(f'==> Loading last model checkpoint')
            model_name = self.find_last_saved_model()

        print(f'==> Loading model checkpoint from {model_name}')
        old_model = self.model
        used_best = load_best_model
        if os.path.isdir(model_name):
            json_file = os.path.join(model_name, 'model_params.json')
            try:
                with open(json_file, "r") as infile:
                    args = json.load(infile)

                self.model = create_object(type(self.model), **args)
            except FileNotFoundError:
                print(f'==> No json file found for {model_name}')

            # fall back to the other checkpoint if the preferred one is absent (a job killed before its first
            # validation has model.pt but no model_best.pt)
            preferred = 'model_best.pt' if load_best_model else 'model.pt'
            fallback = 'model.pt' if load_best_model else 'model_best.pt'
            load_file = os.path.join(model_name, preferred)
            if not os.path.isfile(load_file):
                load_file = os.path.join(model_name, fallback)
                used_best = not load_best_model
                print(f'==> No {preferred} in {model_name}; falling back to {fallback}')
            state_dict = torch.load(load_file, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            folders = model_name.split('/')
            segments = folders[-1].split('_')

            index = 0
            while not segments[index].isnumeric():
                index += 1
            segments.pop(index)
            file_name = '_'.join(segments)

            segments = file_name.split('.')
            segments[-1] = 'json'
            file_name = '.'.join(segments)

            folders[-1] = file_name
            json_file = '/'.join(folders)

            try:
                with open(json_file, "r") as infile:
                    args = json.load(infile)

                self.model = create_object(type(self.model), **args)
            except FileNotFoundError:
                print(f'==> No json file found for {model_name}')

            state_dict = torch.load(model_name, weights_only=True)
            self.model.load_state_dict(state_dict)

        # Must precede the optimizer restore: its state is cast onto each parameter's current device when loaded, and a
        # later .to(self.device) moves the parameters in place without touching the optimizer state.
        self.model.to(self.device)
        self.repoint_optimizers(old_model, self.model)
        if not self.evaluate:
            self.restore_optimizer(model_name, used_best)
            self.restore_run_state(model_name)

        self.checkpoint_loaded = True
        self.checkpoint_path = model_name
        self.loaded_best = used_best
        return model_name

    def restore_optimizer(self, checkpoint_path, load_best_model=True):
        # Best-effort: checkpoints written before optimizer state was saved have none, and a cold AdamW is a transient
        # rather than a correctness problem. The state is keyed by parameter index, which is safe because the model is
        # rebuilt from the same class and model_params.json (same parameter order).
        if self.optimizer is None:
            return
        folder = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
        path = os.path.join(folder, 'optimizer_best.pt' if load_best_model else 'optimizer.pt')
        if not os.path.isfile(path):
            print(f'==> No optimizer state at {path}; the optimizer restarts cold')
            return
        try:
            self.optimizer.load_state_dict(torch.load(path, weights_only=True))
            print(f'==> Restored optimizer state from {path}')
        except Exception as exc:  # never fatal
            print(f'==> Could not restore optimizer state from {path} ({exc}); continuing with a cold optimizer')
            return

        # without the scheduler state every resumed job would restart the LR schedule
        if self.scheduler is None:
            return
        sch_path = os.path.join(folder, 'scheduler_best.pt' if load_best_model else 'scheduler.pt')
        if not os.path.isfile(sch_path):
            print(f'==> No scheduler state at {sch_path}; the LR schedule starts from its initial value')
            return
        try:
            self.scheduler.load_state_dict(torch.load(sch_path, weights_only=False))
            print(f'==> Restored scheduler state from {sch_path} (lr {self.optimizer.param_groups[0]["lr"]:.3e})')
        except Exception as exc:  # never fatal
            print(f'==> Could not restore scheduler state from {sch_path} ({exc}); the LR schedule starts from its '
                  f'initial value')

    def restore_run_state(self, checkpoint_path):
        # Continue the previous job's counters instead of restarting them. The one that matters is best_validation_loss:
        # validate() writes model_best.pt only on an improvement, so with the best reset to inf the first validation of
        # every chained job overwrites "best" whatever it scored. iterations keeps the TensorBoard x-axis continuous and
        # epoch_offset keeps epoch numbering going up. Only comparable while the validation metric is unchanged.
        folder = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
        path = os.path.join(folder, 'metadata.json')
        if not os.path.isfile(path):
            print(f'==> No metadata at {path}; counters start from scratch')
            return
        try:
            with open(path, "r") as infile:
                meta = json.load(infile)
        except (OSError, json.JSONDecodeError) as exc:  # never fatal
            print(f'==> Could not read {path} ({exc}); counters start from scratch')
            return

        self.best_validation_loss = meta.get('Best validation loss', torch.inf)
        self.best_validation_epoch = meta.get('Best validation epoch', 0)
        self.best_validation_iterations = meta.get('Best validation iterations', 0)
        self.best_average_loss = meta.get('Best epoch average loss', torch.inf)
        self.best_model_epoch = meta.get('Best epoch', 0)
        self.best_model_iterations = meta.get('Best epoch iterations', 0)
        self.iterations = meta.get('Last finished iterations', 0)
        self.epoch_offset = meta.get('Last finished epoch', 0)
        print(f'==> Resuming counters: best val loss {self.best_validation_loss:.6f} '
              f'(epoch {self.best_validation_epoch}), {self.iterations} iterations, {self.epoch_offset} epochs done')

    def repoint_optimizers(self, old_model, new_model):
        # load_model rebuilds self.model from model_params.json, but every trainer builds its optimizer on the model
        # created in __init__. Without this the optimizer keeps stepping the discarded model and a --load-checkpoint
        # run never changes the loaded weights. Parameters are matched by name; the (fresh) optimizer state is reset.
        if old_model is new_model:
            return
        old_names = {id(p): name for name, p in old_model.named_parameters()}
        new_params = dict(new_model.named_parameters())

        optimizers = []
        for value in list(vars(self).values()) + list(getattr(self, 'optimizers', [])):
            if isinstance(value, torch.optim.Optimizer) and all(value is not o for o in optimizers):
                optimizers.append(value)

        for optimizer in optimizers:
            covered = set()
            first_model_group = None
            for group in optimizer.param_groups:
                params = []
                for p in group['params']:
                    name = old_names.get(id(p))
                    if name is None:
                        params.append(p)  # not a model parameter, keep it
                    elif name in new_params:
                        params.append(new_params[name])
                        covered.add(name)
                        if first_model_group is None:
                            first_model_group = group
                    else:
                        print(f'==> Optimizer: {name} is not in the loaded model, dropped')
                group['params'] = params

            if first_model_group is not None:
                missing = [name for name in new_params if name not in covered]
                if missing:
                    print(f'==> Optimizer: {len(missing)} loaded-model parameters were not in the old model, added')
                    first_model_group['params'].extend(new_params[name] for name in missing)
            optimizer.state.clear()


def create_object(class_object, **kwargs):
        obj = class_object(**kwargs)
        obj.call_args = kwargs
        return obj


# This method does not work if one of the models is a superset of the other
# because the zip could get rid of the differing excess
def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if key_item_1[0] == key_item_2[0]:
                AssertionError(f'Mismatch found at {key_item_1[0]}')
            else:
                raise ValueError(f"Networks have different layers: model_1 has {key_item_1[0]} but model_2 has {key_item_2[0]}")
    if models_differ == 0:
        return 
        # print('Models match perfectly! :)')


# This works all the time, but it's probably slow
def slow_compare_models(model_1, model_2):
    state_a = model_1.state_dict().__str__()
    state_b = model_2.state_dict().__str__()

    if state_a != state_b:
        raise AssertionError("Weights are different!!!")