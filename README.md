# WaveDINO

Code for **WaveDINO**, a learning-based method that removes atmospheric phase delays from unwrapped Sentinel-1
interferograms while keeping volcanic deformation. WaveDINO is a wavelet encoder-decoder with Mamba state-space
blocks. It is conditioned on frozen DINOv3 ConvNeXt-Base features, on DEM-derived terrain features through FiLM,
and on water mask / distance inputs, and it has a phase ramp head.

The networks are trained on *synthetic* deformation (Mogi-type sources) added to real LiCSAR interferograms. They are
evaluated on held-out synthetic data and, at Laguna del Maule (Chile) and Campi Flegrei (Italy), against GNSS.

This repository contains preprocessing, training, inference, GNSS evaluation and the scripts behind every table and
supplementary analysis of the paper. It covers WaveDINO, the baselines (U-Net, diffusion, SwinIR, Restormer,
WaveMamba) and the ablations.

## Installation

Linux x86-64 with an NVIDIA GPU and a driver for CUDA 12.8 or newer (the Mamba kernels need CUDA). Python 3.12,
PyTorch 2.7.1, `mamba-ssm` 2.2.5 (prebuilt wheel) and `transformers` >= 4.56:

```bash
conda env create -f environment.yml   # creates the env "wavedino"
conda activate wavedino
# CLIP (diffusion baseline only) needs the setuptools installed above to build, so it is not in environment.yml
pip install --no-build-isolation git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
```

[`install.txt`](install.txt) has the same steps without conda.
[`package-list.txt`](package-list.txt) lists every package of the environment the paper's results were produced with.

The diffusion baseline also needs latent-diffusion's pretrained KL-f32 autoencoder (its `config.yaml` is included):

```bash
wget -O kl-f32.zip https://ommer-lab.com/files/latent-diffusion/kl-f32.zip
unzip kl-f32.zip -d models/latent_diffusion/first_stage_models/kl-f32/ && rm kl-f32.zip
```

DINOv3 (`facebook/dinov3-convnext-base-pretrain-lvd1689m`) is a gated model on the Hugging Face Hub. Accept its
licence on the model page and log in once with `huggingface-cli login`, or set `HF_TOKEN`.

## Data

Everything is read from `datasets/` relative to the repository root (run all commands from there). The site names
used in paths and on the command line are `campi_flegrei` and `laguna_del_male`. File names of Laguna del Maule
interferograms start with `laguna_del_maule_`.

| Path | Content | Used by |
|---|---|---|
| `datasets/<site>/train/` | LiCSAR unwrapped interferograms, `<prefix>_<date1>_<date2>.geo.unw.tif`, 500 x 500 clip around the volcano | training (70/10/20 split) and the GNSS test set |
| `datasets/<site>/cc_all/` | coherence, `<prefix>_<date1>_<date2>.geo.cc.tif` | coherence input |
| `datasets/<site>/dem.tif` | DEM on the same grid | DEM input |
| `datasets/<site>/train_all_water/` | the interferogram whose no-data pattern defines the water mask (file name in `GetWaterMaskAndDistance`, `util/datasets.py`) | water mask / distance |
| `datasets/deform_unw_{train,validate,test}/` | synthetic deformation library, `.mat` files with a `los_grid` field (radians), made with the [Synthetic InSAR generator](https://github.com/pui-nantheera/Synthetic_InSAR_image) | synthetic deformation |
| `datasets/campi_flegrei/gps_data/` | INGV-OV weekly GNSS series `<STA>_weekly.txt` and their LOS projection `<STA>_los.csv` (`date,deform`, mm) | GNSS evaluation |
| `datasets/campi_flegrei/gps_data_zenodo_2025/` | INGV-OV daily GNSS series `<STA>_2000_2025.txt` | Supplementary S1 (pairs after 2023-10) |
| `datasets/laguna_del_male/GPS_Data_csv/` | OVDAS-SERNAGEOMIN daily GNSS LOS series `export_<STA>_los.csv` (`fecha,0`, cm) and ENU series `export_<STA>.csv` | GNSS evaluation |
| `datasets/<site>/gacos_all/` | GACOS-corrected interferograms `*.geo.unw_GACOS.tif` | GACOS baseline |

Interferograms are available from the COMET-LiCS portal (<https://comet.nerc.ac.uk>). Station pixel coordinates and
the reference pixels are hard-coded in `util/gps_data_loaders.py` and `util/datasets.py:get_reference`.

GNSS:
- **Campi Flegrei** (public, INGV-OV): weekly series, De Martino et al., *Campi Flegrei cGPS Weekly Positions Time
  Series*, Zenodo, [doi:10.5281/zenodo.10082466](https://doi.org/10.5281/zenodo.10082466); daily series 2000-2025,
  Zenodo, [doi:10.5281/zenodo.22259337](https://doi.org/10.5281/zenodo.22259337).
- **Laguna del Maule** (not public): daily series from the OVDAS-SERNAGEOMIN monitoring network. They are not
  redistributed here; contact OVDAS-SERNAGEOMIN for access.

The LOS series are the ENU displacements projected onto the LiCSAR line-of-sight unit vector (the frame's
`geo.E/N/U` rasters) at each station. Campi Flegrei uses track 124D and Laguna del Maule uses 083D, for every pair.

## Training

```bash
python main.py --dataset campi_flegrei --model-type wavedino-4layers --epochs 1000 --batch-size 35 \
    --image-size 128 --num-workers 5 --result-dir results
```

`--model-type` selects the model:

| Model type | Model |
|---|---|
| `wavedino-4layers` | WaveDINO (the paper model, 3 encoder levels) |
| `wavedino-4layers-water` / `-dem` / `-ramp` / `-dino` | ablations V1-V4: without water inputs / DEM / phase ramp head / DINOv3 |
| `wavedino-4layers-12d` | WaveDINO with only <= 12-day pairs used as whole-scene noise samples (Supplementary S1) |
| `wavemamba`, `restormer`, `swinir`, `classicunet`, `diffusion` | baselines (WaveMamba, Restormer, SwinIR, U-Net, diffusion) |

The batch sizes used for the paper are in `util/generate_jobs.py`. Runs are saved to
`results/<site>_128/<Name>_network_<YY_MM_DD_HH_MM_SS>/`, with `model.pt`, `model_best.pt`, `model_params.json` and
`metadata.json`. TensorBoard logs go to `logs/`. Resume with `--load-checkpoint --model-name last`. `last` is the
newest run of that model type; a run folder path also works.

## Evaluation

```bash
# synthetic test split (Table I, top of Table II) -> results/<site>_eval/<run>_synthetic.pkl
python main.py --dataset campi_flegrei --model-type wavedino-4layers --model-name last --evaluate --synthetic-test

# real interferograms vs GNSS (bottom of Table II): denoises the GNSS test set by resizing the full scene to 128 px
# and by a 128-px sliding window, writes GeoTIFFs to results/<site>_eval/<run>_{resize,patches_full}/ and
# prints the error per GNSS displacement bin; --no-patches skips the sliding window
python main.py --dataset campi_flegrei --model-type wavedino-4layers --model-name last --evaluate
```

The saved GeoTIFFs are in the networks' units. Multiply them by -100 to get back to LiCSAR's unwrapped phase in
radians (28 mm per 2 pi).

## Paper experiments

`python util/generate_jobs.py` writes SLURM scripts for every experiment below to `jobs/` and writes the
`submit_all*.sh` scripts. Edit `SLURM_HEADER` / `ENV_SETUP` for your cluster. The analysis scripts run as modules
from the repository root and write their tables to `results/analysis/` unless noted.

| Result | How |
|---|---|
| Tables I-III (training, synthetic test, GNSS) | `submit_all.sh`, then `submit_all_test.sh` |
| Table II common-pair statistics, Supplementary S5 (resize vs sliding window) | `python -m analysis.gnss_stats --dataset <site> --rebuild-baselines [--with-patches]` -> `results/gnss_stats/` |
| S1 GNSS displacement in short training pairs | `python -m analysis.short_pair_gnss` (and `rate`, `los`) |
| S1 <= 12-day retrain | `submit_all_12d.sh` |
| S2 slope removal, S3 source position, S4 atmospheric strength, S6 gap filling | `submit_all_synth.sh` (`analysis.synth`), then `python -m analysis.synth_summary` |
| S7 time series | `analysis.timeseries_prep` -> `analysis/timeseries_licsbas.sh <variant>` ([LiCSBAS](https://github.com/comet-licsar/LiCSBAS), see below) -> `analysis.timeseries_eval`, `analysis.timeseries_plot` |
| S8 cross-site | `submit_all_cross_site.sh`, then `python -m analysis.cross_site_tables` |
| S9 DINOv3 feature probes | `python -m analysis.dino_lowlevel --dataset <site> --n-images 512` |

The time series were made with LiCSBAS at commit
[`7e5ff25`](https://github.com/comet-licsar/LiCSBAS/commit/7e5ff25) (2025-07-03). The only local changes were for
current matplotlib, which no longer has `plt.register_cmap` / `plt.cm.register_cmap` (removed in 3.9). Colormaps are
registered with `mpl.colormaps.register(...)` instead, in `LiCSBAS_lib/LiCSBAS_plot_lib.py` and
`LiCSBAS_lib/LiCSBAS_tools_lib.py`. Processing is unchanged. Later LiCSBAS versions have not been tested.

## Repository layout

```
main.py                     training / evaluation entry point (trainer registry)
util/datasets.py            data pipeline: transforms, synthetic deformation injection, train/val/test split
util/test.py                full-scene inference and GNSS scoring
util/gps_data_loaders.py    GNSS series and station pixels for both sites
util/generate_jobs.py       SLURM job scripts for the paper experiments
models/generic_trainer.py   training loop, validation, checkpoints, synthetic test, sliding-window inference
models/wavemamba/           WaveMamba and WaveDINO networks (wavemamba.py), terrain / ramp-head blocks
                            (wavedino_blocks.py) and trainers
models/dino.py              frozen DINOv3 backbone and the cross-attention / FiLM blocks
models/restormer/, models/swin_transformer/, models/latent_diffusion/   baselines
analysis/                   scripts for the paper's tables and supplementary analyses
```

## Third-party code

These folders contain modified upstream code and keep its licence (see the `LICENSE` file in each folder):

| Folder | Upstream | Licence |
|---|---|---|
| `models/wavemamba/wavemamba.py` | [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba) | CC BY-NC-SA 4.0 |
| `CrossAttention` in `models/dino.py` | [latent-diffusion](https://github.com/CompVis/latent-diffusion) | MIT |
| `models/restormer/` | [Restormer](https://github.com/swz30/Restormer) | Academic Public License (non-commercial) |
| `models/swin_transformer/` | [SwinIR](https://github.com/JingyunLiang/SwinIR) | Apache-2.0 |
| `models/latent_diffusion/` | [latent-diffusion](https://github.com/CompVis/latent-diffusion) | MIT |
| `models/loss_functions.py` (Laplacian pyramid loss) | Alper Ahmetoglu | MIT (notice in the file) |

## Licence

The code written for this project is Copyright 2026 Robert Gabriel Popescu and released under the Apache License 2.0
(`LICENSE`). Bundled third-party code keeps its own licence (table above). This project's own blocks are Apache-2.0:
the DINOv3 cross-attention / FiLM conditioning (`models/dino.py`) and the terrain modulation and phase ramp heads
(`models/wavemamba/wavedino_blocks.py`). The WaveMamba and WaveDINO networks themselves
(`models/wavemamba/wavemamba.py`) are built from Wave-Mamba blocks. That file, and therefore the assembled WaveDINO
model, is under **CC BY-NC-SA 4.0**: non-commercial use only, and modified versions must be shared under the same
licence. Restormer (`models/restormer/`) is likewise for non-commercial use only. The DINOv3 weights are downloaded
from the Hugging Face Hub under Meta's DINOv3 licence.

## Acknowledgements

The baselines and parts of the network build on code from
[Restormer](https://github.com/swz30/Restormer), [SwinIR](https://github.com/JingyunLiang/SwinIR),
[Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba), [latent-diffusion](https://github.com/CompVis/latent-diffusion)
and [Mamba](https://github.com/state-spaces/mamba). Synthetic deformation is generated with
[Synthetic_InSAR_image](https://github.com/pui-nantheera/Synthetic_InSAR_image), and the interferograms are from
[LiCSAR](https://comet.nerc.ac.uk).
