import os
from typing import Iterator, Union

import numpy as np
import pandas as pd
import torch


def get_test_data_loader(dataset_name, path=None):
    # GNSS evaluation set: a fixed random 5000 interferograms of the dataset folder (seeded in ClassicDataLoader)
    return ClassicDataLoader(path, max_number_files=5000)


class TestDataLoader(Iterator):
    def __init__(self, path: Union[str, os.PathLike], max_number_files=None):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def full_path(self, file):
        raise NotImplementedError


class ClassicDataLoader(TestDataLoader):
    def __init__(self, path: Union[str, os.PathLike], max_number_files=None):
        super().__init__(path, max_number_files)
        self.path = path
        self.max_number_files = max_number_files

        self.files = os.listdir(path)

        self.files = np.array(self.files)

        if self.max_number_files is not None:
            torch.manual_seed(1234)
            perm = torch.randperm(len(self.files))
            self.files = self.files[perm][:self.max_number_files]

    def __len__(self):
         return len(self.files)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.files):
            raise StopIteration

        file = os.path.join(self.path, self.files[self.index])
        self.index += 1
        return file

    def full_path(self, file):
        for sample in self.files:
            if file in sample:
                return os.path.join(self.path, sample)

        raise FileNotFoundError
