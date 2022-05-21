import csv, copy
import torch
import numpy as np
from torch.utils.data import Dataset as AbstractDataset
from torchvision.transforms import Compose
from lib.transforms import Identity


def read_data_csv(path, dtype={}, rename_map={}):
    with open(path) as fp:
        data_list = list(csv.DictReader(fp))

    for data in data_list:
        for k, dt in dtype.items():
            data[k] = dt(data[k])

        for key_old, key_new in rename_map.items():
            data[key_new] = data.pop(key_old)

    return data_list


class CollationFunction:

    def __init__(self, fields=None):
        self.fields = fields

    def __call__(self, batch):
        batch_data = dict.fromkeys(self.fields)

        for key in self.fields:
            batch_data[key] = torch.stack([data[key] for data in batch])

        return batch_data


class Dataset(AbstractDataset):

    def __init__(self, data_list, transforms=None):
        self.data_list  = data_list
        self.transforms = Compose(transforms) if transforms else Identity()

    def __getitem__(self, index):
        data = copy.deepcopy(self.data_list[index])

        return self.transforms(data)

    def __len__(self):
        return len(self.data_list)