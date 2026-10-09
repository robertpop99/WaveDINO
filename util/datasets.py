import argparse
import numbers
import os
import random
import re
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Callable, List
from enum import Enum

import cv2
import torchvision
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage import morphology
from skimage.restoration import inpaint
from torch import optim
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
import torch.utils.data
import tifffile as tiff
from matplotlib import image as plt_image
import datetime as dt

from scipy.interpolate import griddata
import scipy.ndimage as ndi

printing_logs = []

deform_chars_size = 4
deform_chars_lookup_table = {
    # 'type'      : 0,
    # 'strike'    : 4,
    # 'dip'       : 5,
    # 'rake'      : 3,
    # 'length'    : 4,
    # 'bdepth'    : 0,
    # 'tdepth'    : 0,
    # 'incidence' : 1,
    # 'heading'   : 2,
    # 'opening'   : 9,
    # 'width'     : 10,
    'depth': 0,
    # # 'rotate'    : 12,
    'vol': 1,
    # 'radius'    : 1,
    # 'pressure'  : 14,
}

deform_chars_normalization_table = {
    'type': 6,
    'strike': 360,
    'dip': 90,
    # 'rake'      : 3,
    # 'length'    : 4,
    'bdepth': 10,
    'tdepth': 5,
    # 'incidence' : 7,
    # 'heading'   : 8,
    # 'opening'   : 9,
    # 'width'     : 10,
    'depth': 8,
    # # 'rotate'    : 12,
    # 'vol'       : 12,
    # 'radius'    : 1,
    # 'pressure'  : 14,
}


class OutputType(Enum):
    SAMPLE = 0 # stack of multiple images
    IMAGE = 1 # image used as input
    DEFORM_CHARS = 2
    DEFORM_COORDS = 3
    ORIG_IMG = 4 # stratified + turbulent, original insar
    DEFORMATION = 5
    DEM = 6
    ANNOTATION = 7
    NAME = 8
    TIME_LENGTH = 9 # number of days for interferogram
    WATER_MASK = 10
    WATER_DISTANCE = 11
    COHERENCE = 12
    WD1 = 13
    WD2 = 14
    WD3 = 15
    WD4 = 16
    WD5 = 17

water_distance_keys = [OutputType.WD1, OutputType.WD2, OutputType.WD3, OutputType.WD4, OutputType.WD5]

def get_number(s):
    p = r'[+-]?\d+(\.\d+)?(?:[eE][+-]?\d+(\.\d+)?)?'
    if re.search(p, s) is not None:
        for catch in re.finditer(p, s):
            return catch


def create_deform_chars(s=None) -> torch.Tensor:
    deform_chars = torch.zeros(deform_chars_size, dtype=torch.float)
    if s is None or s == '':
        # deform_chars[0] = 1
        return deform_chars

    s = s.split('/')[-1]

    for frag in s.split('_'):
        catch = get_number(frag)
        nr_s = catch[0]
        stat = frag[:catch.span()[0]].lower()

        if 'e' in nr_s:
            # nr = float(nr_s.split('e')[0]) * 10 ** float(nr_s.split('e')[1])
            nr = float(nr_s.split('e')[1])
        else:
            nr = float(nr_s)

        if stat in deform_chars_lookup_table:
            if deform_chars[deform_chars_lookup_table[stat]] == 0:
                deform_chars[deform_chars_lookup_table[stat]] = nr
            else:
                deform_chars[deform_chars_lookup_table[stat]] = (deform_chars[deform_chars_lookup_table[stat]] + nr) / 2
            # deform_chars[int(nr)] = 1

    return deform_chars




class GenericDataset(datasets.VisionDataset):
    def get_path(self, index) -> str:
        raise NotImplementedError


class SubsetWrapper(Subset, GenericDataset):
    def __init__(self, subset:Subset):
        super().__init__(subset.dataset, subset.indices)
        assert isinstance(self.dataset, GenericDataset)

    def get_path(self, index) -> str:
        return self.dataset.get_path(index)


class LocalDataset(GenericDataset):
    def __init__(self, root,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 ) -> None:
        super().__init__(root)
        self.root = root

        self.samples = np.array(os.listdir(self.root))

        if is_valid_file is not None:
            self.samples = self.samples[np.vectorize(is_valid_file)(self.samples)]

        if self.samples.size == 0:
            raise ValueError("Found no samples in the folder")

        if 'png' in self.samples[0] or 'jpeg' in self.samples[0]:
            self.loader = torchvision.io.read_image
        elif 'tif' in self.samples[0]:
            self.loader = tiff.imread
        else:
            raise ValueError("File type not supported")

        self.len = self.samples.size

    def __getitem__(self, index):
        sample = self.loader(os.path.join(self.root, self.samples[index]))
        return sample

    def __len__(self):
        return self.len

    def get_path(self, index) -> str:
        return os.path.join(self.root, self.samples[index])


class INSAR(datasets.VisionDataset):
    def __init__(self, root,
                 transform: Optional[transforms.Compose] = None,
                 name_sanitizer: Callable = lambda name: name,
                 annotation_extractor: Callable = lambda name: "",
                 dataset: GenericDataset = None,
                 safe_file: Callable = lambda name: True,
                 safe_transform: Optional[transforms.Compose] = None,
                 all_transforms: Optional[List[transforms.Compose]] = None
                 ):
        """
        :param root: root path of the Dataset, used only if dataset is None
        :param transform: the basic transforms Compose object to be used, the first mode only uses this transform
        :param name_sanitizer: function to clean files names
        :param annotation_extractor: function to extract the annotations used for training from files names
        :param dataset: if dataset is None, a default one will be created using root
        :param safe_file: used in second mode, returns true if any of the transforms from all_transforms can be used
        :param safe_transform: transforms Compose object used if safe_file returns False
        :param all_transforms: list of transforms Compose objects, a random one is used on each sample, as long as safe_file returns True

        . note::
            The class has two modes, the second one is used if all_transforms is not None

        """
        super().__init__(root, transform)
        self.dataset = dataset

        if dataset is None:
            self.dataset = LocalDataset(root)
        else:
            self.dataset = dataset
        print(f'Dataset size: {len(self.dataset)}')

        self.name_sanitizer = name_sanitizer
        self.annotation_extractor = annotation_extractor
        self.safe_file = safe_file
        self.safe_transform = safe_transform
        self.all_transforms = all_transforms

        if transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = transform

        if 'tif' in self.dataset.get_path(0):
            self.transform = transforms.Compose([transforms.ToTensor(), *self.transform.transforms])

        self.multiple_transforms = all_transforms is not None
        if self.multiple_transforms:
            if self.safe_transform is None:
                self.safe_transform = transforms.Compose([])

            if 'tif' in self.dataset.get_path(0):
                self.safe_transform = transforms.Compose([transforms.ToTensor(), *self.safe_transform.transforms])
                for i in range(len(self.all_transforms)):
                    self.all_transforms[i] = transforms.Compose([transforms.ToTensor(), *self.all_transforms[i].transforms])


    def add_key(self, key, sample, output, used_keys, t, exceptions=None):
        if key not in output:
            return sample, used_keys
        if exceptions is not None:
            for exception in exceptions:
                if type(t) is exception:
                    return sample, used_keys

        sample = torch.cat([sample, output[key]], dim=0)
        used_keys.append(key)
        return sample, used_keys

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample_path = self.dataset.get_path(index)
        nr_of_nans = np.isnan(sample).sum() / np.size(sample)
        while nr_of_nans > 0.75:
            index += 1
            if index >= len(self.dataset):
                index = 0
            sample = self.dataset[index]
            sample_path = self.dataset.get_path(index)
            nr_of_nans = np.isnan(sample).sum() / np.size(sample)

        if self.multiple_transforms:
            if self.safe_file(sample_path):
                trs = random.choice(self.all_transforms)
            else:
                trs = self.safe_transform
        else:
            trs = self.transform

        output = {
            OutputType.IMAGE: sample,
        }

        for t in trs.transforms:
            sample = output[OutputType.IMAGE]
            used_keys = [OutputType.IMAGE]
            sample, used_keys = self.add_key(key=OutputType.DEFORMATION, sample=sample, output=output,
                                             used_keys=used_keys, t=t)
            sample, used_keys = self.add_key(key=OutputType.DEM, sample=sample, output=output,
                                             used_keys=used_keys, t=t, exceptions=[BringToInterval, MaskWater])
            sample, used_keys = self.add_key(key=OutputType.ORIG_IMG, sample=sample, output=output,
                                             used_keys=used_keys, t=t)
            sample, used_keys = self.add_key(key=OutputType.WATER_MASK, sample=sample, output=output,
                                             used_keys=used_keys, t=t, exceptions=[BringToInterval, MaskWater])
            sample, used_keys = self.add_key(key=OutputType.WATER_DISTANCE, sample=sample, output=output,
                                             used_keys=used_keys, t=t, exceptions=[BringToInterval, MaskWater])
            sample, used_keys = self.add_key(key=OutputType.COHERENCE, sample=sample, output=output,
                                             used_keys=used_keys, t=t, exceptions=[BringToInterval, MaskWater])

            if OutputType.WD1 in output:
                for dc in water_distance_keys:
                    sample, used_keys = self.add_key(key=dc, sample=sample, output=output,
                                                 used_keys=used_keys, t=t, exceptions=[BringToInterval, MaskWater])

            if (
                type(t) is AddDeformation or
                type(t) is Interpolate or
                type(t) is GetWaterMaskAndDistance or
                type(t) is Reference or
                type(t) is AddCoherence
            ):
                sample = output[OutputType.IMAGE]
                used_keys = [OutputType.IMAGE]

            if (
                type(t) is CutMargin or
                type(t) is AddCoherence
            ):
                sample = t(sample, sample_path)
            elif type(t) is MaskWater:
                sample = t(sample, output[OutputType.WATER_MASK])
            else:
                sample = t(sample)

            if type(sample) is dict:
                output.update(sample)
                if OutputType.SAMPLE in sample:
                    sample = sample[OutputType.SAMPLE]

            if type(sample) is not dict:
                out = torch.chunk(sample, len(used_keys), dim=0)

                for ind, key in enumerate(used_keys):
                    output[key] = out[ind]


        if output[OutputType.DEFORM_CHARS].sum() != 0:
            # only if we have deformation
            output[OutputType.DEFORM_CHARS][-2] = output[OutputType.DEFORM_COORDS][0]
            output[OutputType.DEFORM_CHARS][-1] = output[OutputType.DEFORM_COORDS][1]

        if OutputType.WD1 in output:
            dist = torch.cat((
                output[OutputType.WD1],
                output[OutputType.WD2],
                output[OutputType.WD3],
                output[OutputType.WD4],
                output[OutputType.WD5],
            ), dim=0)
            output[OutputType.WATER_DISTANCE] = dist

        # print(f'Returning sample {index}', flush=True)
        # batch, ground_truth, dem = torch.chunk(data, 3, dim=1)
        # torch.cat([img, total_deform, rest, orig_img])
        return_dict = {
            OutputType.IMAGE: output[OutputType.IMAGE],
            OutputType.DEFORMATION: output[OutputType.DEFORMATION],
            OutputType.DEM: output[OutputType.DEM],
            OutputType.ORIG_IMG: output[OutputType.ORIG_IMG],
            OutputType.DEFORM_CHARS: output[OutputType.DEFORM_CHARS],
            OutputType.ANNOTATION: self.annotation_extractor(sample_path),
            OutputType.NAME: self.name_sanitizer(sample_path),
            OutputType.TIME_LENGTH: str(get_time_length(sample_path)),
            OutputType.WATER_MASK: output[OutputType.WATER_MASK],
            OutputType.WATER_DISTANCE: output[OutputType.WATER_DISTANCE],
            OutputType.COHERENCE: output[OutputType.COHERENCE],
        }
        return return_dict
        # return sample, deform_chars, self.annotations[index], self.sanitized_names[index]

    def __len__(self):
        return len(self.dataset)


class RandomScale(torch.nn.Module):
    def __init__(self, high, low, constraints=None):
        super().__init__()
        if constraints is None:
            constraints = [10]
        self.constraints = constraints
        self.high = high - low
        self.low = low

    def forward(self, x):
        mode = torch.randint(low=0, high=2, size=(1,)).item()
        if mode == 0:
            scale = torch.rand(1) * self.high + self.low
            x = x * scale
        elif mode == 1:
            ind = torch.randint(low=0, high=len(self.constraints), size=(1,)).item()
            constraint = self.constraints[ind]
            x = (x / torch.abs(x).max()) * constraint
        return x


class AddDeformation(torch.nn.Module):
    def __init__(self, root, probability, apply_transformation=True, random=True, wrap=False, image_size=128,
                 max_deformation_sources=4):
        super().__init__()
        self.root = root
        self.samples = os.listdir(self.root)
        self.samples = np.vectorize(lambda x: self.root + '/' + x)(self.samples)
        self.max_deformation_sources = max_deformation_sources
        self.image_size = image_size

        mask = np.vectorize(lambda x: 'type4' in x)(self.samples)
        self.samples = self.samples[mask]

        self.size = len(self.samples)

        print_log = f'deformation_{self.size}'
        if print_log not in printing_logs:
            printing_logs.append(print_log)
            print(f'Number of samples in deformation dataset: {self.size}')

        self.probability = probability
        if apply_transformation:
            if image_size == 128:
                translate_size = (0.1, 0.1)
            elif image_size == 256:
                translate_size = (0.2, 0.2)
            else:
                translate_size = (0.4, 0.4)

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=30),
                # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomAffine(degrees=0, translate=translate_size),
                transforms.CenterCrop(image_size),
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        self.scale = RandomScale(high=1, low=0.01, constraints=[5, 8, 10, 15])

        if random:
            self.next_index = self.random_index
        else:
            self.index = -1
            self.next_index = self.iterative_index

        self.wrap = wrap

    def random_index(self):
        return torch.randint(low=0, high=self.size, size=(1,))

    def iterative_index(self):
        self.index = (self.index + 1) % self.size
        return self.index

    def forward(self, img: torch.Tensor,):
        # rest = img[1:]
        # img = img[0:1]
        orig_img = img.clone()

        if self.probability >= torch.rand(1):
            deformation_sources = torch.randint(low=0, high=self.max_deformation_sources, size=(1,)).item() + 1
            total_deform = torch.zeros_like(img)
            for i in range(deformation_sources):
                if self.wrap:
                    sample_name = self.samples[self.next_index()]
                    deform = tiff.imread(sample_name)
                    deform = self.transform(deform)
                    # bring from [0,1] to [-pi,pi]
                    deform = deform * (2 * np.pi) - np.pi
                    total_deform = (total_deform + deform) % (2 * np.pi) - np.pi
                else:
                    sample_name = self.samples[self.next_index()]
                    deform = loadmat(sample_name)['los_grid']
                    deform = self.transform(deform)
                    total_deform = total_deform + deform
            if self.wrap:
                # add deformation to image
                img = (img + total_deform) % (2 * np.pi) - np.pi
            else:
                total_deform = self.scale(total_deform)
                img = img + total_deform
            deform_chars = create_deform_chars(s=None)
        else:
            total_deform = torch.zeros_like(img)
            deform_chars = create_deform_chars(s=None)

        return {
                OutputType.IMAGE: img,
                OutputType.DEFORMATION: total_deform,
                OutputType.ORIG_IMG: orig_img,
                OutputType.DEFORM_CHARS: deform_chars
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class AddDEM(torch.nn.Module):
    def __init__(self, dem_path):
        super().__init__()
        self.dem = tiff.imread(dem_path)
        transform = transforms.Compose([transforms.ToTensor()])
        self.dem = transform(self.dem)
        # self.dem = (self.dem - self.dem.min()) / (self.dem.max() - self.dem.min())
        # self.dem = self.dem * 200 - 100
        self.dem = torch.clamp(self.dem, -500., 4000.) / 4000.

    def forward(self, img):
        return {OutputType.DEM: self.dem}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class CutMargin(torch.nn.Module):
    def __init__(self, day_interval):
        super().__init__()
        self.day_interval = day_interval

    def forward(self, img, img_name):
        if time_length_filter(img_name, self.day_interval):
            return img
        else:
            if 'campi_flegrei' not in img_name:
                cut_region = torch.randint(low=0, high=4, size=(1,))
                if cut_region == 0:
                    return img[:, :128]
                elif cut_region == 1:
                    return img[:, :, :128]
                elif cut_region == 2:
                    return img[:, 372:]
                else:
                    return img[:, :, 372:]
            else:
                cut_region = torch.randint(low=0, high=10, size=(1,))
                if cut_region < 4:
                    return img[:, :200]
                elif cut_region < 8:
                    return img[:, :, 310:]
                elif cut_region == 8:
                    return img[:, 300:]
                else:
                    return img[:, :, :200]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class BringToInterval(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max
        self.range = max - min

    def forward(self, img):
        img =  (img - self.min) / self.range
        img = img * 2 - 1
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"


class Clamp(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, img):
        return torch.clamp(img, min=self.min, max=self.max)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max})"


class NanToNum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return torch.nan_to_num(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SetRange(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img * 2 - 1.

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def remove_small_objects(mask, max_size):
    """Removes objects of <= max_size pixels on scikit-image 0.26 (max_size) and on 0.25 (min_size, the pinned env)."""
    try:
        return morphology.remove_small_objects(mask, max_size=max_size)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=max_size + 1)


class GetWaterMaskAndDistance(torch.nn.Module):
    def __init__(self, max_dist=200, centers=None, sigma=0.25, dataset='laguna_del_male'):
        super().__init__()
        if centers is None:
            centers = [-1, -0.5, 0, 0.5, 1] # normalized
        # Assumes 50m per pixel => 10km for 200
        self.max_dist = max_dist
        self.centers = centers
        self.sigma = sigma # also normalized

        self.mask = None
        self.dist = None

        self.return_dict = None

        if dataset == 'laguna_del_male':
            img = tiff.imread('datasets/laguna_del_male/train_all_water/laguna_del_maule_20160310_20161030.geo.unw.tif')
        elif dataset == 'campi_flegrei':
            img = tiff.imread('datasets/campi_flegrei/train_all_water/campi_flegrei_20160605_20160804.geo.unw.tif')
        else:
            raise NotImplementedError
        self.calculate(img)

    def calculate(self, img: np.ndarray):
        import scipy.ndimage as ndi

        mask = ~(img == 0)

        def encode_distance(dist):
            centers = self.centers  # normalized
            sigma = self.sigma

            # centers = [-200, -100, 0, 100, 200]
            # sigma = 40

            # centers = [-10, -5, 0, 5, 10]  # km
            # sigma = 3

            features = []

            for c in centers:
                f = np.exp(-(dist - c) ** 2 / (2 * sigma ** 2))
                features.append(f)

            return np.stack(features, axis=0)

        def distance_to_coast(mask):
            # mask: 1=land, 0=water
            max_dist = self.max_dist
            # distance from land pixels to nearest water pixel
            land_dist = ndi.distance_transform_edt(mask)
            water_dist = ndi.distance_transform_edt(~mask)

            dist = land_dist - water_dist

            dist = ndi.gaussian_filter(dist, sigma=3)

            dist = np.clip(dist, -max_dist, max_dist)
            dist = dist / max_dist

            # dist = encode_distance(dist)

            return dist

        def clean_water_mask(mask, min_size=200):
            water = mask == 0

            labeled, num = ndi.label(water)

            for i in range(1, num + 1):
                region = labeled == i
                if region.sum() < min_size:
                    mask[region] = 1  # convert to land

            return mask

        # mask = clean_water_mask(mask, min_size=500)
        # mask = ndi.binary_closing(mask, iterations=2)
        water = ~mask
        water = remove_small_objects(water, max_size=500)
        water = morphology.erosion(water, morphology.disk(7))

        # water = morphology.erosion(water, morphology.disk(5))
        # water = morphology.erosion(water, morphology.disk(20))
        water = remove_small_objects(water, max_size=500)
        mask = ~water
        mask = morphology.closing(mask, morphology.disk(2))

        dist = distance_to_coast(mask)

        mask = torch.from_numpy(mask)
        dist = torch.from_numpy(dist)

        mask = mask.unsqueeze(dim=0)
        dist = dist.unsqueeze(dim=0)

        self.mask = mask
        self.dist = dist

        self.return_dict = {
            # OutputType.WATER_MASK: torch.ones_like(mask),
            OutputType.WATER_MASK: mask,
            OutputType.WATER_DISTANCE: dist,
            # OutputType.WD1: dist[0].unsqueeze(dim=0),
            # OutputType.WD2: dist[1].unsqueeze(dim=0),
            # OutputType.WD3: dist[2].unsqueeze(dim=0),
            # OutputType.WD4: dist[3].unsqueeze(dim=0),
            # OutputType.WD5: dist[4].unsqueeze(dim=0),
        }

    def forward(self, img):
        return self.return_dict


class Reference(torch.nn.Module):
    def __init__(self, ref=(487, 420), reverse=True):
        super().__init__()
        self.ref = ref
        self.reverse = reverse

    def forward(self, img):
        assert img.shape[0] == 1
        img = img.view(*img.shape[1:])
        img = reference_interferogram(img, self.ref, self.reverse)
        img = img.view(1, *img.shape)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AddCoherence(torch.nn.Module):
    def __init__(self, dataset='laguna_del_male'):
        super().__init__()
        self.path = f'datasets/{dataset}/cc_all/'
        self.tr = transforms.Compose([transforms.ToTensor()])

    def forward(self, img, img_name):
        name = img_name.split('/')[-1].split('.')[0]
        cc_path = os.path.join(self.path, name + '.geo.cc.tif')
        if os.path.exists(cc_path):
            cc = tiff.imread(cc_path)
            cc = self.tr(cc)
            cc = torch.nan_to_num(cc, nan=0)
        else:
            cc = torch.ones((1, 500, 500))

        return {OutputType.COHERENCE: cc}




class CustomRandomCrop(torch.nn.Module):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = TF.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = transforms.RandomCrop.get_params(img, self.size)

        c1 = (height - 1) / 2 - i
        c2 = (width - 1) / 2 - j

        return {OutputType.SAMPLE: TF.crop(img, i, j, h, w), OutputType.DEFORM_COORDS: torch.tensor([c1 / h, c2 / w])}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class Interpolate(torch.nn.Module):
    def __init__(self, lower_limit=-50., upper_limit=50., interpolation_type='new'):
        super().__init__()
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.interpolation_type = interpolation_type

        print_log = f'interpolation_{interpolation_type}'
        if print_log not in printing_logs:
            printing_logs.append(print_log)
            print(f'Using interpolation type: {interpolation_type}')

        if interpolation_type == 'classic':
            self.interpolate_function = self.interpolate_classic
        elif interpolation_type == 'new':
            self.interpolate_function = self.interpolate_nans_new
        else:
            raise ValueError(f'Invalid interpolation type: {interpolation_type}')
        # self.water = tiff.imread('datasets/taal/taal_032D_07536_111313_mask_water.tif').astype(bool)

    def interpolate_nans_new(self, img):
        c, h, w = img.shape
        image = img.view(h, w).cpu().numpy().copy()

        h, w = image.shape
        yy, xx = np.mgrid[:h, :w]

        mask = ~np.isnan(image)

        points = np.column_stack((yy[mask], xx[mask]))
        values = image[mask]

        image_interp = griddata(points, values, (yy, xx), method='nearest')

        # only smooth interpolated pixels
        interp_mask = np.isnan(image)
        smoothed = ndi.gaussian_filter(image_interp, sigma=1)

        image_interp[interp_mask] = smoothed[interp_mask]

        return torch.from_numpy(image_interp).view(c, h, w)

    def interpolate_classic(self, img):
        c, h, w = img.shape
        image = img.view(h, w).cpu().numpy().copy()

        # Blue area outside the interval
        # imageblur = cv2.medianBlur(image, 5)
        # image[image > self.upper_limit] = imageblur[image > self.upper_limit]
        # image[image < self.lower_limit] = imageblur[image < self.lower_limit]

        # mask_dil = morphology.dilation(np.isnan(image), morphology.disk(2))
        # mask = morphology.closing(mask_dil, morphology.disk(5))
        mask = np.isnan(image)
        mask = morphology.closing(mask, morphology.disk(1))
        mask = mask.astype(bool)

        if mask.sum() > (image.size * 0.75):
            return torch.zeros_like(img)

        # Shift with bias
        # refimg = image.copy()
        # refimg[mask + (image > self.upper_limit) + (image < self.lower_limit)] = np.nan
        # refval = np.nanmean(refimg)
        # image = image - refval

        # image = cv2.inpaint(image, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        image_inp = inpaint.inpaint_biharmonic(image, mask)
        # image[mask_dil] = image_inp[mask_dil]
        image[mask] = image_inp[mask]
        image = np.nan_to_num(image)
        return torch.from_numpy(image).view(c, h, w)

    def forward(self, img):
        # failsafe for bad samples
        if torch.isnan(img).sum() > (torch.numel(img) * 0.75):
            return torch.zeros_like(img)

        # image = self.interpolate_classic(image)
        # image = self.interpolate_nans_new(image)

        return self.interpolate_function(img)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(inpaint_biharmonic)"


class MaskWater(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, mask):
        return img * mask


DEFAULT_DAY_INTERVAL = 20


def get_day_interval(args) -> int:
    """
    Temporal-baseline cutoff (exclusive, days) below which a whole interferogram is treated as deformation-free.
    A '-<N>d' suffix on --model-type (e.g. wavedino-4layers-12d) keeps only baselines <= N days.
    """
    match = re.search(r'-(\d+)d(?![a-z])', args.model_type)
    if match is None:
        return DEFAULT_DAY_INTERVAL
    return int(match.group(1)) + 1


def get_transforms(args, mode, dem_path, deformation_path, logger=None, resize_mode='crop', image_size=None):
    assert mode in ['train', 'test', 'validate']
    assert resize_mode in ['crop', 'resize']
    if image_size is None:
        image_size = args.image_size

    wrap = args.wrap
    # for wrap interval -pi + pi/ rezise / set range
    interval_min = -torch.pi if wrap else -100.
    interval_max = torch.pi if wrap else 100.

    deform_image_size = image_size if resize_mode == 'crop' else 500

    if mode == 'train':
        add_deform = AddDeformation(deformation_path, 0.99, apply_transformation=True, random=True,
                           wrap=wrap, image_size=deform_image_size, max_deformation_sources=4)
    elif mode == 'test':
        add_deform = AddDeformation(deformation_path, 1., apply_transformation=True, random=False,
                               wrap=wrap, image_size=deform_image_size, max_deformation_sources=4)
    elif mode == 'validate':
        add_deform = AddDeformation(deformation_path, 1., apply_transformation=True, random=True,
                               wrap=wrap, image_size=deform_image_size, max_deformation_sources=4)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    transform_obj = [
            GetWaterMaskAndDistance(dataset=args.dataset),
            Reference(ref=get_reference(args.dataset), reverse=True),
            AddCoherence(dataset=args.dataset),
            Interpolate(interval_min, interval_max, interpolation_type=args.interpolation),
            AddDEM(dem_path),
    ]

    if resize_mode == 'resize':
        transform_obj += [
            add_deform,
            transforms.Resize((image_size, image_size)),
        ]
    elif resize_mode == 'crop':
        if image_size == 128:
            # the synthetic test set keeps the default cutoff so it matches the other models' test set
            day_interval = get_day_interval(args) if mode != 'test' else DEFAULT_DAY_INTERVAL
            transform_obj += [
                CutMargin(day_interval=day_interval),
            ]

        transform_obj += [
            CustomRandomCrop(image_size, pad_if_needed=True),
            add_deform,
        ]

    transform_obj += [
        BringToInterval(interval_min, interval_max),
        transforms.ConvertImageDtype(torch.float),
        # MaskWater(),
    ]

    transform_obj = transforms.Compose(transform_obj)

    # if logger is not None:
    #     trs = str(transform_obj).split('\n')[1:-1]
    #     log = ''
    #     for tr in trs:
    #         log += tr.split('    ')[-1]
    #         log += '->'
    #     log = log[:-2]
    #     logger['preprocessing'] = log
    return transform_obj


def get_all_transforms(args, dem_path, mode, deformation_path):
    modes = [('crop'  , 128), ('crop'  , 160), ('crop',   192), ('crop',   224), ('crop',   256), ('crop',   288), ('crop',   320),
             ('resize', 128), ('resize', 160), ('resize', 192), ('resize', 224), ('resize', 256), ('resize', 288), ('resize', 320)]

    trs = []
    for resize_mode, img_size in modes:
        trs.append(get_transforms(args, logger=None, mode=mode, dem_path=dem_path,
                       deformation_path=deformation_path, resize_mode=resize_mode, image_size=img_size))

    return trs


def get_datasets(args, logger=None, load_from_logger=False, resolution_mode = 'adaptive'):
    assert resolution_mode in ['adaptive', 'static']

    dataset_path = str(os.path.join(args.datasets_dir, args.dataset))
    dem_path = os.path.join(dataset_path, "dem.tif")
    train_dir = os.path.join(dataset_path, "train")

    def file_filter(file):
        condition = True
        condition &= time_length_filter(file, 60)

        if 'laguna_del_male' in args.dataset:
            condition &= 'laguna_del_maule' in file
        else:
            condition &= args.dataset in file

        return condition

    def default_file_filter(file):
        condition = True
        if 'laguna_del_male' in args.dataset:
            condition &= 'laguna_del_maule' in file
        else:
            condition &= args.dataset in file

        return condition

    def safe_file_filter(file, day_interval):
        return time_length_filter(file, day_interval)

    if args.wrap:
        deform_path = {
            'train': os.path.join(args.datasets_dir, 'deform_wrap_train'),
            'validate': os.path.join(args.datasets_dir, 'deform_wrap_validate'),
            'test': os.path.join(args.datasets_dir, 'deform_wrap_test'),
        }
    else:
        deform_path = {
            'train': os.path.join(args.datasets_dir, 'deform_unw_train'),
            'validate': os.path.join(args.datasets_dir, 'deform_unw_validate'),
            'test': os.path.join(args.datasets_dir, 'deform_unw_test'),
        }

    dataset_loader = LocalDataset(root=train_dir, is_valid_file=file_filter if resolution_mode == 'adaptive' else default_file_filter)
    train_loader, validate_loader, test_loader = torch.utils.data.random_split(dataset_loader, [0.7, 0.1, 0.2])
    dataset_loader_dict = {
        'train': SubsetWrapper(train_loader),
        'validate': SubsetWrapper(validate_loader),
        'test': SubsetWrapper(test_loader)
    }

    loader_dict = {}

    modes = ['train', 'validate', 'test']
    for mode in modes:
        print(f'{mode.capitalize()} dataset', flush=True)
        if resolution_mode == 'adaptive':
            day_interval = get_day_interval(args) if mode != 'test' else DEFAULT_DAY_INTERVAL
            safe_transform = get_transforms(args, logger=None, mode=mode, dem_path=dem_path,
                                                  deformation_path=deform_path[mode], resize_mode='crop',
                                                  image_size=128)
            all_transforms = get_all_transforms(args, dem_path=dem_path, mode=mode,
                                                      deformation_path=deform_path[mode])
            dataset = INSAR(
                root=train_dir, name_sanitizer=generic_name_sanitizer,
                annotation_extractor=generic_annotation_extractor, dataset=dataset_loader_dict[mode],
                safe_file=lambda file, d=day_interval: safe_file_filter(file, d),
                safe_transform=safe_transform, all_transforms=all_transforms)

            loader = torch.utils.data.DataLoader(persistent_workers=True, pin_memory=True, prefetch_factor=4,
                                                       dataset=dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, shuffle=True,
                                                       collate_fn=collate_fn)
        elif resolution_mode == 'static':
            transform = get_transforms(args, logger=None, mode=mode, dem_path=dem_path,
                                       deformation_path=deform_path[mode], resize_mode='crop')
            dataset = INSAR(
                root=train_dir, transform=transform, name_sanitizer=generic_name_sanitizer,
                annotation_extractor=generic_annotation_extractor, dataset=dataset_loader_dict[mode])
            loader = torch.utils.data.DataLoader(persistent_workers=True, pin_memory=True, prefetch_factor=4,
                                                       dataset=dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, shuffle=True)
        else:
            raise ValueError(f"Invalid resolution mode: {resolution_mode}")
        loader_dict[mode] = loader

    return loader_dict


def collate_fn(batch):
    # sizes: list[int] = [128, 160, 192, 224, 256, 288, 320]
    sizes: list[int] = [128]
    need_interpolation = [OutputType.IMAGE, OutputType.DEFORMATION, OutputType.DEM, OutputType.ORIG_IMG,
                          OutputType.WATER_MASK, OutputType.WATER_DISTANCE, OutputType.COHERENCE]

    size: int = random.choice(sizes)

    output = {}
    for key in batch[0].keys():
        output[key] = []

    for sample in batch:
        for key in sample.keys():
            data = sample[key]
            if key in need_interpolation:
                data = TF.resize(data, [size, size])  # same size per batch
            output[key].append(data)

    for key in output.keys():
        if type(output[key][0]) is torch.Tensor:
            output[key] = torch.stack(output[key], dim=0)

    return output


def generic_name_sanitizer(name):
    return name.split('/')[-1].split('.')[0]


def generic_annotation_extractor(name):
    name = name.split('/')[-1].split('.')[0]

    date1 = name.split('_')[-2]
    date2 = name.split('_')[-1]
    date1 = date1[4:]
    date2 = date2[4:]
    date1 = str(int(date1[:2])) + ' ' + str(int(date1[2:]))
    date2 = str(int(date2[:2])) + ' ' + str(int(date2[2:]))

    annotations = date1 + ' ' + date2
    return annotations


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _get_timedelta(file_name) -> dt.timedelta:
    name = file_name.split('/')[-1].split('.')[0]

    date1 = name.split('_')[-2]
    date2 = name.split('_')[-1]

    date1 = dt.datetime.strptime(str(date1),'%Y%m%d').date()
    date2 = dt.datetime.strptime(str(date2),'%Y%m%d').date()
    diff = date2 - date1

    return diff


def get_time_length(file_name) -> int:
    diff = _get_timedelta(file_name)
    return diff.days


def time_length_filter(file_name, days) -> bool:
    diff = _get_timedelta(file_name)
    return diff < dt.timedelta(days=days)


def reference_interferogram(img, ref=(213, 460), reverse=True):
    # ref = (487, 420)
    ref_y, ref_x = ref

    if reverse:
        img = img * -1  # all images are reversed for some reason (for laguna del maule)

    reference = img[ref_y, ref_x]
    if np.isnan(reference):
        s = 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            while np.isnan(reference) and s < 15:
                reference = np.nanmean(img[ref_y - s:ref_y + s, ref_x - s:ref_x + s])
                s += 1
        if np.isnan(reference):
            reference = 0
    img = img - reference

    return img


def get_reference(dataset_name):
    if 'campi_flegrei' in dataset_name:
        return 115, 293
    elif 'laguna_del_male' in dataset_name or 'laguna_del_maule' in dataset_name:
        return 213, 460
    else:
        raise ValueError('Dataset name {} not recognized'.format(dataset_name))


def rad_to_mm(x):
    return x / (2 * np.pi) * 28


def mm_to_rad(x):
    return x / 28 * (2 * np.pi)
