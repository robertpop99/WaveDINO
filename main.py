import argparse
import os
import pickle
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.generic_trainer import GenericTrainer
from models.latent_diffusion.ClassicUnet import ClassicUnetTrainer
from models.latent_diffusion.diffusion_trainer import DiffusionTrainer
from models.restormer.restormer_trainer import RestormerTrainer
from models.swin_transformer.swinir_trainer import SwinIRTrainer
from models.wavemamba.mamba_trainer import WaveMambaTrainer
from models.wavemamba.wavedino_trainer import WaveDinoTrainer
from util.datasets import get_datasets, AddCoherence
from util.gps_data_loaders import get_gps_data_loader
from util.parser import parse_args
from util.test import run_test, evaluate_model
from util.test_data_loader import get_test_data_loader


TRAINERS = {
    'classicunet': ClassicUnetTrainer,
    'diffusion': DiffusionTrainer,
    'swinir': SwinIRTrainer,
    'restormer': RestormerTrainer,
    'wavemamba': WaveMambaTrainer,
    'wavedino': WaveDinoTrainer,
}


def get_trainer_class(model_type):
    # ablation suffixes (e.g. wavedino-4layers-dino) are read inside the trainer
    model_name = model_type.split('_')[0]
    model_name = model_name.split('+')[0]
    model_name = model_name.split('-')[0]
    assert model_name in TRAINERS
    return TRAINERS[model_name]


def get_summary_writer_log_dir(args: argparse.Namespace, start_time) -> str:
    if args.debug:
        tb_log_dir = args.log_dir + '/null'
    else:
        tb_log_dir = args.log_dir + '/' + start_time
    return str(tb_log_dir)


def run_on_test_data(args, trainer, folder_path, folder_name, binary=True, name_addon='',
                     patches=True, centers=True, resize=True):
    if name_addon != '' and not name_addon.startswith('_'):
        name_addon = f'_{name_addon}'

    save_folder_center = os.path.join(folder_path, f'{folder_name}{name_addon}')
    save_folder_gps_center = os.path.join(folder_path, f'{folder_name}_gps_center{name_addon}')
    # full-frame sliding window; the old '_patches' folders hold 256 centre-crop outputs
    save_folder_patches = os.path.join(folder_path, f'{folder_name}_patches_full{name_addon}')
    save_folder_resize = os.path.join(folder_path, f'{folder_name}_resize{name_addon}')

    if 'laguna_del_male' in args.dataset or 'laguna_del_maule' in args.dataset:
        path = 'datasets/laguna_del_male/train/'
        data_loader = get_test_data_loader(args.dataset, path)
    else:
        path = 'datasets/campi_flegrei/train/'
        data_loader = get_test_data_loader(args.dataset, path)

    gps_data = get_gps_data_loader(args.dataset)
    coh_extractor = AddCoherence(args.dataset)

    run_test(trainer=trainer,
             data_loader=data_loader,
             gps_data=gps_data,
             dataset_name=args.dataset,
             run_model_on_centers=centers,
             run_model_with_patches=patches,
             run_model_with_resize=resize,
             save_folder_center=save_folder_center,
             save_folder_gps_center=save_folder_gps_center,
             save_folder_resize=save_folder_resize,
             save_folder_patches=save_folder_patches,
             skip_completed=True,
             use_binary_data=False)

    if patches:
        evaluate_model(save_folder_patches, data_loader, gps_data, coherence_extractor=coh_extractor)
    if resize:
        evaluate_model(save_folder_resize, data_loader, gps_data, coherence_extractor=coh_extractor)


def main(args: argparse.Namespace):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger = None

    datasets_dir = get_datasets(args, logger, resolution_mode='adaptive')

    if not os.path.isdir(args.result_dir):
        os.mkdir(args.result_dir)

    start_time = time.strftime('%y_%m_%d_%H_%M_%S')

    log_dir = get_summary_writer_log_dir(args, start_time)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )


    print(f'Running with model {args.model_type }')

    trainer = get_trainer_class(args.model_type)(args, datasets_dir, None, summary_writer, logger)
    assert isinstance(trainer, GenericTrainer)

    trainer.start_time = start_time

    if not args.evaluate:
        if args.load_checkpoint:
            trainer.load_model(args.model_name, load_best_model=True)
        try:
            if args.debug:
                print(f'====> Running model in debug mode')

            trainer.train()

            summary_writer.close()
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            if not args.debug and trainer.iterations >= 50:
                trainer.save_model()
            summary_writer.close()
    else:
        loaded_model_name = trainer.load_model(args.model_name, load_best_model=True)
        args.model_name = loaded_model_name # for autoloading the last model
        torch.cuda.empty_cache()

        folder_name = args.model_name.split('/')
        if len(folder_name[-1]) == 0:
            folder_name = folder_name[-2]
        else:
            folder_name = folder_name[-1]

        folder_path = os.path.join(args.result_dir, f'{args.dataset}_eval')
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        if args.synthetic_test:
            # Table I and the synthetic half of Table II: held-out synthetic deformation on real atmosphere
            test_results = trainer.test(test_limit=args.test_limit)
            with open(os.path.join(folder_path, f'{folder_name}_synthetic.pkl'), 'wb') as f:
                pickle.dump(test_results, f)
            return

        run_on_test_data(args=args, trainer=trainer, folder_path=folder_path, folder_name=folder_name,
                         binary=False, centers=False, patches=not args.no_patches, resize=True)


def evaluate_all_models(args):
    if 'laguna_del_male' in args.dataset or 'laguna_del_maule' in args.dataset:
        path = 'datasets/laguna_del_male/train/'
        data_loader = get_test_data_loader(args.dataset, path)
    else:
        path = 'datasets/campi_flegrei/train/'
        data_loader = get_test_data_loader(args.dataset, path)

    gps_data = get_gps_data_loader(args.dataset)

    folder_path = os.path.join(args.result_dir, f'{args.dataset}_eval')

    for folder in os.listdir(folder_path):
        finished_images_folder = os.path.join(folder_path, folder)
        if os.path.isdir(finished_images_folder):
            print(f'Printing results for {folder}')
            evaluate_model(finished_images_folder, data_loader, gps_data)


if __name__ == '__main__':
    main(parse_args())
    # evaluate_all_models(parse_args())
