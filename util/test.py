import os
import pickle
import time
import warnings
from enum import Enum
from pathlib import Path

import h5py
import numpy as np
import torch
import tifffile as tiff
from torch import nn
from torchvision import transforms

from .datasets import Clamp, time_length_filter, BringToInterval, SetRange, generic_annotation_extractor, \
    reference_interferogram, Interpolate, get_time_length, OutputType, rad_to_mm, GetWaterMaskAndDistance, MaskWater, \
    Reference, get_reference, AddDEM, AddCoherence
from .gps_data_loaders import GPSDataLoader, LagunaDelMaleGPSData, get_gps_data_loader
import datetime as dt

from .test_data_loader import TestDataLoader


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# class TimeseriesData(metaclass=Singleton):
#     def __init__(self):
#         self.timeseries_data = None
#
#     def get_data(self):
#         if self.timeseries_data is None:
#             self.timeseries_data = _get_timeseries_data()
#         return self.timeseries_data


class ModelStatsLookupTable(metaclass=Singleton):
    def __init__(self):
        self.lookup_table = {}

    def has(self, folder_path):
        return folder_path in self.lookup_table

    def get(self, folder_path):
        return self.lookup_table[folder_path]

    def add(self, folder_path, stats):
        self.lookup_table[folder_path] = stats

    def reset(self):
        self.lookup_table = {}


def _get_timeseries_data(path='datasets/laguna_del_male/InSAR_TimeSeries/cum_filt.h5'):
    f = h5py.File(path, 'r')

    cum = np.array(f['cum'])
    dates = np.array(f['imdates'])

    timeseries = {}

    for idx in range(len(dates)):
        timeseries[str(dates[idx])] = cum[idx]

    return timeseries


def get_timeseries_data(path='datasets/laguna_del_male/InSAR_TimeSeries/cum_filt.h5'):
    # return TimeseriesData().get_data()
    return _get_timeseries_data(path)


def read_binary_unw(path, ref=(670, 737), crop_center=True, crop_size=500, reverse=True, reference=False):
    img = np.fromfile(path, 'float32')
    img = np.reshape(img, (int(np.sqrt(img.size)), int(np.sqrt(img.size))))

    if reference:
        img = reference_interferogram(img, ref, reverse)

    if crop_center:
        start = (img.shape[0] - crop_size) // 2
        end = img.shape[0] - start
        img = img[start:end, start:end]

    return img


class CustomCrop(torch.nn.Module):
    def __init__(self, size, center):
        super().__init__()
        self.size = int(size)
        self.y = int(center[0] - size / 2)
        self.x = int(center[1] - size / 2)

    def forward(self, img):
        return img[:, :, self.y: self.y + self.size, self.x: self.x + self.size]


def run_test(
            trainer,
            data_loader: TestDataLoader,
            gps_data: GPSDataLoader,
            dataset_name,
            run_model_on_centers=False,
            run_model_with_patches=False,
            run_model_with_resize=True,
            save_folder_center=None,
            save_folder_gps_center=None,
            save_folder_patches=None,
            save_folder_resize=None,
            skip_completed=True,
            use_binary_data=False,
            coherence_extractor=None,
            lower_time_filter=20,
):
    """
    lower_time_filter: pairs shorter than this many days are skipped.
    """

    torch.cuda.empty_cache()
    if run_model_on_centers:
        assert save_folder_center is not None, f"No path provided for save_folder_center"
        if not os.path.isdir(save_folder_center):
            os.mkdir(save_folder_center)

        assert save_folder_gps_center is not None, f"No path provided for save_folder_gps_center"
        if not os.path.isdir(save_folder_gps_center):
            os.mkdir(save_folder_gps_center)

    if run_model_with_patches:
        assert save_folder_patches is not None, f"No path provided for save_folder_patches"
        if not os.path.isdir(save_folder_patches):
            os.mkdir(save_folder_patches)

    if run_model_with_resize:
        assert save_folder_resize is not None, f"No path provided for save_folder_resize"
        if not os.path.isdir(save_folder_resize):
            os.mkdir(save_folder_resize)

    device = trainer.device
    image_size = trainer.image_size
    batch_size = trainer.batch_size

    trainer.model.to(device)
    trainer.model.eval()

    # ----------------------------
    # Load DEM
    # ----------------------------

    dem_obj = AddDEM(f'datasets/{dataset_name}/dem.tif')
    dem = dem_obj.dem.view(1,*dem_obj.dem.shape).type(torch.float).to(device)

    # ----------------------------
    # preprocessing
    # ----------------------------

    preprocessing_transforms = transforms.Compose([
        Reference(ref=get_reference(dataset_name), reverse=True),
        Interpolate(-100, 100, interpolation_type='new'),
        BringToInterval(-100, 100),
        transforms.ConvertImageDtype(torch.float),
    ])

    center_crop = transforms.CenterCrop(128)
    resize_320 = transforms.Resize((128, 128))

    water_mask_obj = GetWaterMaskAndDistance(dataset=dataset_name)
    water_mask = water_mask_obj.mask.view(1,*water_mask_obj.mask.shape).type(torch.float).to(device)
    water_dist = water_mask_obj.dist.view(1,*water_mask_obj.dist.shape).type(torch.float).to(device)
    water_mask_cpu = water_mask.to('cpu').squeeze(dim=0)
    apply_water = MaskWater()

    extract_coherence = AddCoherence(dataset=dataset_name) if coherence_extractor is None else coherence_extractor


    # ----------------------------
    # GPS station crops
    # ----------------------------
    if use_binary_data:
        gps_coords_version = 'coords_off_centered'
    else:
        gps_coords_version = 'coords_centered'

    gps_stations = gps_data.stations()
    station_crops = {}

    for station in gps_stations[:5]:
        if gps_coords_version == 'coords_off_centered':
            assert isinstance(gps_data, LagunaDelMaleGPSData)
            coords = gps_data.station_coords_off_centered(station) - 250 # for images 1000 x 1000
        else:
            coords = gps_data.station_coords(station)

        station_crops[station] = CustomCrop(
            image_size,
            (coords[0], coords[1])
        )

    # ----------------------------
    # batch buffers
    # ----------------------------

    img_batch = []
    coherence_batch = []
    file_batch = []

    def process_batch():

        if len(img_batch) == 0:
            return

        imgs = torch.cat(img_batch, dim=0)
        cohs = torch.cat(coherence_batch, dim=0)

        # ==========================
        # CENTER MODEL
        # ==========================

        if run_model_on_centers:

            B = imgs.shape[0]

            crops = []
            dem_crops = []
            mask_crops = []
            dist_crops = []
            annotations = []
            time_lengths = []

            for i in range(B):

                img = imgs[i:i+1]
                file = file_batch[i]

                ann = generic_annotation_extractor(file)
                tl = str(get_time_length(file))

                crops.append(center_crop(img))
                dem_crops.append(center_crop(dem))
                mask_crops.append(center_crop(water_mask))
                dist_crops.append(center_crop(water_dist))

                annotations.append(ann)
                time_lengths.append(tl)

                for key in station_crops:
                    crops.append(station_crops[key](img))
                    dem_crops.append(station_crops[key](dem))
                    mask_crops.append(station_crops[key](water_mask))
                    dist_crops.append(station_crops[key](water_dist))
                    annotations.append(ann)
                    time_lengths.append(tl)

            img_centers = torch.cat(crops, dim=0)
            dem_centers = torch.cat(dem_crops, dim=0)
            mask_centers = torch.cat(mask_crops, dim=0)
            dist_centers = torch.cat(dist_crops, dim=0)

            data = {
                OutputType.IMAGE: img_centers,
                OutputType.DEM: dem_centers,
                OutputType.ANNOTATION: annotations,
                OutputType.TIME_LENGTH: time_lengths,
                OutputType.WATER_MASK: mask_centers,
                OutputType.WATER_DISTANCE: dist_centers,
            }

            result = trainer.test_step(data)

            r = 0
            for b,file in enumerate(file_batch):

                tiff.imwrite(
                    os.path.join(save_folder_center, f'{file}_1.tif'),
                    result[r].cpu().numpy()
                )
                r += 1

                for key in station_crops:

                    tiff.imwrite(
                        os.path.join(save_folder_gps_center, f'{file}_{key}_3.tif'),
                        result[r].cpu().numpy()
                    )
                    r += 1

        # ==========================
        # PATCH MODEL
        # ==========================

        if run_model_with_patches:

            # full frame: the reference pixel is inside the tiled area and every GPS station is covered
            crop_img = imgs
            crop_cohs = cohs
            crop_dem = dem.repeat(len(img_batch),1,1,1)
            crop_mask = water_mask.repeat(len(img_batch),1,1,1)
            crop_dist = water_dist.repeat(len(img_batch),1,1,1)

            annotations = [
                generic_annotation_extractor(f) for f in file_batch
            ]

            time_lengths = [
                str(get_time_length(f)) for f in file_batch
            ]

            data = {
                OutputType.IMAGE: crop_img,
                OutputType.DEM: crop_dem,
                OutputType.ANNOTATION: annotations,
                OutputType.TIME_LENGTH: time_lengths,
                OutputType.WATER_MASK: crop_mask,
                OutputType.WATER_DISTANCE: crop_dist,
                OutputType.COHERENCE: crop_cohs,
            }

            prob_maps = trainer.get_deformation(data)

            for i,file in enumerate(file_batch):

                tiff.imwrite(
                    os.path.join(save_folder_patches, f'{file}_2.tif'),
                    prob_maps[i].cpu().numpy()
                )

        # ==========================
        # RESIZE MODEL
        # ==========================

        if run_model_with_resize:

            resize_img = resize_320(imgs)
            resize_coh = resize_320(cohs)
            resize_dem = resize_320(dem).repeat(len(img_batch), 1, 1, 1)
            resize_mask = resize_320(water_mask).repeat(len(img_batch), 1, 1, 1)
            resize_dist = resize_320(water_dist).repeat(len(img_batch), 1, 1, 1)

            annotations = [
                generic_annotation_extractor(f) for f in file_batch
            ]

            time_lengths = [
                str(get_time_length(f)) for f in file_batch
            ]

            data = {
                OutputType.IMAGE: resize_img,
                OutputType.DEM: resize_dem,
                OutputType.ANNOTATION: annotations,
                OutputType.TIME_LENGTH: time_lengths,
                OutputType.WATER_MASK: resize_mask,
                OutputType.WATER_DISTANCE: resize_dist,
                OutputType.COHERENCE: resize_coh,
            }

            prob_maps = trainer.test_step(data)

            for i, file in enumerate(file_batch):
                tiff.imwrite(
                    os.path.join(save_folder_resize, f'{file}_resize_128.tif'),
                    prob_maps[i].cpu().numpy()
                )

        img_batch.clear()
        coherence_batch.clear()
        file_batch.clear()

    def skip_file(f):
        if time_length_filter(f, lower_time_filter):
            return True

        if skip_completed:
            if run_model_with_patches:
                if not os.path.isfile(os.path.join(save_folder_patches, f'{f}_2.tif')):
                    return False

            if run_model_with_resize:
                if not os.path.isfile(os.path.join(save_folder_resize, f'{f}_resize_128.tif')):
                    return False

            if run_model_on_centers:
                if not os.path.isfile(os.path.join(save_folder_center, f'{f}_1.tif')):
                    return False

                for key in station_crops:
                    if not os.path.isfile(os.path.join(save_folder_gps_center, f'{f}_{key}_3.tif')):
                        return False
            return True

        return False

    # ----------------------------
    # MAIN LOOP
    # ----------------------------

    time_elapsed = 0.
    average_time = 0.
    samples_done = 0

    for idx, file_path in enumerate(data_loader):
        start_time = time.time()

        file = file_path.split('/')[-1]
        if skip_file(file):
            continue

        if use_binary_data:
            img = read_binary_unw(file_path)
        else:
            img = tiff.imread(file_path)

        img = torch.from_numpy(img).view(1,*img.shape)
        if (torch.isnan(img).sum() + (img == 0).sum()) >= img.shape[1] * img.shape[2] * 0.80:
            # print(f'Skipping {file} for having {torch.isnan(img).sum() / (img.shape[1] * img.shape[2])}% NaN values')
            continue

        img = preprocessing_transforms(img)
        # img = apply_water(img, water_mask_cpu)

        img = img.view(1,*img.shape).to(device)
        coh = extract_coherence(img, file)[OutputType.COHERENCE]
        coh = coh.view(1,*coh.shape).to(device)

        img_batch.append(img)
        coherence_batch.append(coh)
        file_batch.append(file)

        if len(img_batch) >= batch_size:
            process_batch()
            average_time /= samples_done
            time_remaining = average_time * (len(data_loader) - (idx - 1))
            print(f'Finished sample {idx - 1} / {len(data_loader)} in: {time_elapsed:.2f} s  Estimated completion time: {time_remaining / 60:.2f} min', flush=True)
            average_time = 0.
            samples_done = 0

        time_elapsed = time.time() - start_time
        average_time += time_elapsed
        samples_done += 1

    process_batch()


class Stats(Enum):
    tms_diff = 0
    img_diff_1 = 1  # no patches
    img_diff_2 = 2  # patches
    img_diff_3 = 3  # station in the middle of the image
    img_len = 4  # length on interf
    all_gps = 5
    all_data = 6

    data = 7  # actual data of image
    ind = 8  # index of image in all_data

    args = 9 # arguments of the function call


def get_model_stats(images_folder,
                    data_loader,
                    gps_data,
                    lower_time_filter=20,
                    time_filter=180,
                    gps_stations=None,
                    norm_interf=False,
                    denorm_interf=True,
                    offset=None,
                    timeseries_path='datasets/laguna_del_male/InSAR_TimeSeries/cum_filt.h5',
                    use_lookup_table=True,
                    coherence_extractor=None,
                    ):
    lookup_table = ModelStatsLookupTable()

    # if use_lookup_table and lookup_table.has(images_folder):
    if False:
        return lookup_table.get(images_folder)
    else:
        stats = _get_model_stats(images_folder,
                                 data_loader,
                                 gps_data,
                                 lower_time_filter,
                                 time_filter,
                                 gps_stations,
                                 norm_interf,
                                 denorm_interf,
                                 offset,
                                 timeseries_path,
                                 coherence_extractor,)
        lookup_table.add(images_folder, stats)
        return stats


def _get_model_stats(images_folder,
                     data_loader: TestDataLoader,
                     gps_data: GPSDataLoader,
                     lower_time_filter=20,
                     time_filter=180,
                     gps_stations=None,
                     norm_interf=False,
                     denorm_interf=True,
                     offset=None,
                     timeseries_path='datasets/laguna_del_male/InSAR_TimeSeries/cum_filt.h5',
                     coherence_extractor:AddCoherence =None
                    ):
    if gps_stations is None:
        gps_stations = gps_data.stations()
    folder = images_folder
    # timeseries = get_timeseries_data(timeseries_path)

    if coherence_extractor is None:
        coherence_extractor = AddCoherence()

    data = {
        Stats.tms_diff: {},
        Stats.img_diff_1: {},
        Stats.img_diff_2: {},
        Stats.img_diff_3: {},
        Stats.img_len: {},
        Stats.all_gps: {},
        Stats.all_data: {},
    }
    data_with_station = [Stats.tms_diff, Stats.img_diff_1, Stats.img_diff_2, Stats.img_diff_3, Stats.img_len, Stats.all_gps]

    for station in gps_stations:
        for key in data_with_station:
            data[key][station] = {
                Stats.data: [],
                Stats.ind: [],
            }

    if 'binary' in folder or 'IFG_Data' in folder:
        gps_coords_version = 'coords_off_centered'
    else:
        gps_coords_version = 'coords_centered'

    for index, file in enumerate(os.listdir(folder)):
        if time_length_filter(file, lower_time_filter):
            continue

        if not time_length_filter(file, time_filter):
            continue

        name = file.split('.')[0]
        try:
            file_path = data_loader.full_path(name)
        except FileNotFoundError:
            continue
        if 'tif' in file_path:
            img = tiff.imread(file_path)
            img = torch.from_numpy(img)
            if len(img.shape) == 3:
                img = img[0]
        else:
            img = np.fromfile(file_path, 'float32')
            img = np.reshape(img, (int(np.sqrt(img.size)), int(np.sqrt(img.size))))
            img = torch.from_numpy(img)
            img = img[250:750, 250:750]

        nan_percentage = torch.isnan(img).sum() / (img.shape[0] * img.shape[1])
        zero_percentage = (img == 0).sum() / (img.shape[0] * img.shape[1])
        # if nan_percentage >= 0.50:
        #     # print(nan_percentage, zero_percentage)
        #     continue

        segments = file.split('/')[-1].split('.')[0].split('_')
        _index = 0
        while not segments[_index].isnumeric():
            _index += 1
        date1 = segments[_index]
        date2 = segments[_index + 1]

        # if date1 not in timeseries or date2 not in timeseries:
        #     continue

        date1_ = dt.datetime.strptime(str(date1), '%Y%m%d').date()
        date2_ = dt.datetime.strptime(str(date2), '%Y%m%d').date()
        time_interval = date2_ - date1_

        if 'tif' in file:
            img = tiff.imread(os.path.join(folder, file))
            if len(img.shape) == 3:
                img = img[0]
        else:
            # img = read_binary_unw(os.path.join('datasets/laguna_del_male/IFG_Data', name))
            img = read_binary_unw(os.path.join(folder, file))

        if norm_interf:
            if 'laguna_del_maule' in file:
                img = reference_interferogram(img, ref=(213, 460), reverse=True)
            elif 'campi_flegrei' in file:
                img = reference_interferogram(img, ref=(115, 293), reverse=True)
            else:
                raise NotImplementedError

        if denorm_interf:
            img = img * 100

        if offset is not None:
            img = img + offset

        img = rad_to_mm(img)

        avg_dist = 0
        station_nr = 0

        # tms_img = timeseries[date2] - timeseries[date1]

        img_stats = {
            'index': index,
            'file': file,
            'data1': date1,
            'date2': date2,
            'nan_percentage': nan_percentage,
            'zero_percentage': zero_percentage,
            'name': name,
            'time_interval': time_interval.days,
        }


        coh = coherence_extractor(torch.from_numpy(img), file)
        coh = coh[OutputType.COHERENCE]
        coh = coh[0]

        for gps_station in gps_stations:
            coords = gps_data.station_coords(gps_station)

            # if coh[coords[0], coords[1]] < 0.2:
            #     continue

            if gps_coords_version == 'coords_off_centered':
                assert isinstance(gps_data, LagunaDelMaleGPSData)
                coords = gps_data.station_coords_off_centered(gps_station) - 250 # for 1000 * 1000 images

            date1_deform = gps_data.value(gps_station, date1)
            date2_deform = gps_data.value(gps_station, date2)

            if date1_deform is not None and date2_deform is not None:
                real_diff = date2_deform - date1_deform

                img_stats[f'{gps_station}_real'] = real_diff
                img_stats[f'{gps_station}_date1'] = date1_deform
                img_stats[f'{gps_station}_date2'] = date2_deform

                data[Stats.all_gps][gps_station][Stats.data].append(real_diff)
                data[Stats.all_gps][gps_station][Stats.ind].append(index)

                # if tms_img.shape[0] == 1000:
                #     assert isinstance(gps_data, LagunaDelMaleGPSData)
                #     ts_coords = gps_data.station_coords_off_centered(gps_station)
                #     tms_diff = tms_img[ts_coords[0], ts_coords[1]]
                # elif tms_img.shape[0] == 501:
                #     ts_coords = gps_data.station_coords(gps_station)
                #     tms_diff = tms_img[ts_coords[0], ts_coords[1]]
                # else:
                #     tms_diff = np.nan
                #     ValueError(f"Unknown timeseries shape {tms_img.shape}")
                # tms_gps_diff = real_diff - tms_diff

                data[Stats.img_len][gps_station][Stats.data].append(time_interval.days)
                data[Stats.img_len][gps_station][Stats.ind].append(index)

                # data[Stats.tms_diff][gps_station][Stats.data].append(tms_gps_diff)
                # data[Stats.tms_diff][gps_station][Stats.ind].append(index)
                #
                # img_stats[f'{gps_station}_tms_diff'] = tms_diff
                # img_stats[f'{gps_station}_tms_gps_diff'] = tms_gps_diff

                if '_resize' in file:
                    if 'resize_128' in file:
                        coords_resize = 128 / 500
                    elif 'resize_320' in file:
                        coords_resize = 320 / 500

                    coords = (coords * coords_resize).round().astype(int)

                    img = img.reshape(img.shape[-2], img.shape[-1])

                    model_diff = img[coords[0], coords[1]]
                    model_gps_diff = real_diff - model_diff

                    avg_dist += np.abs(model_gps_diff)

                    data[Stats.img_diff_2][gps_station][Stats.data].append(model_gps_diff)
                    data[Stats.img_diff_2][gps_station][Stats.ind].append(index)

                    img_stats[f'{gps_station}_model_diff'] = model_diff
                    img_stats[f'{gps_station}_model_gps_diff'] = model_gps_diff
                    img_stats['type'] = 2

                    station_nr += 1

                elif '_3.tif' in file:
                    if gps_station in file:
                        img = img.reshape(img.shape[-2], img.shape[-1])
                        h, w = img.shape
                        model_diff = img[int(h / 2), int(w / 2)]
                        model_gps_diff = real_diff - model_diff

                        avg_dist += np.abs(model_gps_diff)

                        data[Stats.img_diff_3][gps_station][Stats.data].append(model_gps_diff)
                        data[Stats.img_diff_3][gps_station][Stats.ind].append(index)

                        img_stats[f'{gps_station}_model_diff'] = model_diff
                        img_stats[f'{gps_station}_model_gps_diff'] = model_gps_diff
                        img_stats['type'] = 3

                        station_nr += 1
                    else:
                        data[Stats.all_gps][gps_station][Stats.data].pop()
                        data[Stats.all_gps][gps_station][Stats.ind].pop()

                        data[Stats.img_len][gps_station][Stats.data].pop()
                        data[Stats.img_len][gps_station][Stats.ind].pop()

                elif '_2.tif' in file:
                    h, w = img.shape[-2], img.shape[-1]
                    if (h, w) == (256, 256):
                        coords = coords - 122  # legacy outputs: 256 centre crop of a 500 frame
                    if coords[0] < 0 or coords[1] < 0 or coords[0] >= h or coords[1] >= w:
                        continue

                    model_diff = img[coords[0], coords[1]]
                    model_gps_diff = real_diff - model_diff

                    avg_dist += np.abs(model_gps_diff)

                    data[Stats.img_diff_2][gps_station][Stats.data].append(model_gps_diff)
                    data[Stats.img_diff_2][gps_station][Stats.ind].append(index)

                    img_stats[f'{gps_station}_model_diff'] = model_diff
                    img_stats[f'{gps_station}_model_gps_diff'] = model_gps_diff
                    img_stats['type'] = 2

                    station_nr += 1

                elif '_1.tif' in file:
                    coords = coords - 186
                    if ((coords >= 128) + (coords < 0)).sum() > 0:
                        continue

                    img = img.reshape(img.shape[-2], img.shape[-1])
                    model_diff = img[coords[0], coords[1]]
                    model_gps_diff = real_diff - model_diff

                    avg_dist += np.abs(model_gps_diff)

                    data[Stats.img_diff_1][gps_station][Stats.data].append(model_gps_diff)
                    data[Stats.img_diff_1][gps_station][Stats.ind].append(index)

                    img_stats[f'{gps_station}_model_diff'] = model_diff
                    img_stats[f'{gps_station}_model_gps_diff'] = model_gps_diff
                    img_stats['type'] = 1

                    station_nr += 1

                else:
                    if ((coords > 500) + (coords < 0)).sum() > 0:
                        continue

                    model_diff = img[coords[0], coords[1]]
                    model_gps_diff = real_diff - model_diff

                    avg_dist += np.abs(model_gps_diff)

                    data[Stats.img_diff_2][gps_station][Stats.data].append(model_gps_diff)
                    data[Stats.img_diff_2][gps_station][Stats.ind].append(index)

                    img_stats[f'{gps_station}_model_diff'] = model_diff
                    img_stats[f'{gps_station}_model_gps_diff'] = model_gps_diff
                    img_stats['type'] = 0

                    station_nr += 1

        data[Stats.all_data][index] = img_stats

    for station in gps_stations:
        for key in data_with_station:
            data[key][station][Stats.data] = np.array(data[key][station][Stats.data])
            data[key][station][Stats.ind] = np.array(data[key][station][Stats.ind])

    data[Stats.args] = {
        'lower_time_filter': lower_time_filter,
        'time_filter': time_filter,
        'images_folder': images_folder,
        'norm_interf': norm_interf,
        'denorm_interf': denorm_interf,
        'gps_stations': gps_stations,
    }

    with open(os.path.join(str(Path(images_folder).parent), f'{Path(images_folder).name}_data.pkl'), 'wb') as f:
        print(f'Saving data to {os.path.join(str(Path(images_folder).parent), f'{Path(images_folder).name}_data.pkl')}')
        pickle.dump(data, f)

    return data


class DistanceQualityMetrics(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters

        self.methods = {
            'MSE': nn.MSELoss(),
            'ABS': lambda x, y: torch.abs(x - y),
            # 'L3': LPLoss(3),
            # 'L1': LPLoss(1),
        }
        self.test_results = {}

        for key in self.methods:
            self.test_results[key] = {}

            for filt1, filt2 in filters:
                self.test_results[key][f'{filt1}_{filt2}'] = torch.empty(0)
                self.test_results[key][f'{filt1}_{filt2}'] = torch.empty(0)

    def forward(self, x, target):
        x = torch.tensor(x)
        target = torch.tensor(target)
        for filt1, filt2 in self.filters:
            if (target < filt1) & (target > filt2):
                for key in self.test_results:
                    res = self.methods[key](x, target).cpu()
                    res = res.unsqueeze(0)
                    self.test_results[key][f'{filt1}_{filt2}'] = torch.cat(
                        (self.test_results[key][f'{filt1}_{filt2}'], res.cpu())
                    )

    def return_results(self):
        return self.test_results

    def print_results(self):
        print('=======> Results for Distance Quality Metrics:', flush=True)
        for key in self.test_results:
            row = ''
            for filt1, filt2 in self.filters:
                res = self.test_results[key][f'{filt1}_{filt2}']
                mask = torch.isnan(res)
                res = res[~mask]
                print(f'{key} ({filt1}_{filt2}): mean {res.mean()} std {res.std()} size {res.shape}', flush=True)
                row += f'{res.mean():.2f} '
            print(f'=======> Excel friendly row: {row}', flush=True)


def evaluate_model(folder_path, data_loader, gps_data, coherence_extractor, lower_time_filter=20):
    time_filter = 1000000
    filters = [(np.inf, 0), (np.inf, 30), (30, 20), (20, 10), (10, 0)]
    filters.reverse()
    gps_stations = gps_data.stations()

    data = get_model_stats(folder_path, data_loader, gps_data, lower_time_filter=lower_time_filter, time_filter=time_filter,
                           gps_stations=None, norm_interf=False, denorm_interf=True, coherence_extractor=coherence_extractor)

    main_dqm = DistanceQualityMetrics(filters=filters)
    stations_dqm = {}
    for station in gps_stations:
        stations_dqm[station] = DistanceQualityMetrics(filters=filters)

    for index in data[Stats.all_data]:
        img_stats = data[Stats.all_data][index]

        for gps_station in gps_stations:
            if f'{gps_station}_model_diff' in img_stats:
                x = img_stats[f'{gps_station}_model_diff']
                t = img_stats[f'{gps_station}_real']

                stations_dqm[gps_station](x, t)
                main_dqm(x, t)

    for key in stations_dqm:
        print('\n\n\n')
        print(f'==========> Results for station {key} <==========')
        stations_dqm[key].print_results()

    print('\n\n\n')
    print(f'==========> All results <==========')
    main_dqm.print_results()
    print('\n\n\n')


    # for filt1, filt2 in filters:
    #     print()
    #     print(f'Filter: ({filt1}, {filt2})')
    #     all_diff = np.array([])
    #
    #     for station in gps_stations:
    #         diff = data[Stats.img_diff_1][station][Stats.data]
    #         if len(diff) == 0:
    #             diff = data[Stats.img_diff_2][station][Stats.data]
    #             if len(diff) == 0:
    #                 diff = data[Stats.img_diff_3][station][Stats.data]
    #                 if len(diff) == 0:
    #                     continue
    #
    #         mask = data[Stats.all_gps][station][Stats.data]
    #         mask = np.abs(mask)
    #         mask = (mask < filt1) & (mask > filt2)
    #
    #         diff = diff[mask]
    #         diff = diff[~np.isnan(diff)]
    #
    #         if len(diff) == 0:
    #             continue
    #
    #         diff = np.abs(diff)
    #         all_diff = np.append(all_diff, diff)
    #
    #         print(f'{station} : Mean {diff.mean():.2f} Std {diff.std():.2f}  Min {diff.min():.4f}  Len {diff.shape}')
    #
    #     print(all_diff.mean(), all_diff.std(), all_diff.shape)


