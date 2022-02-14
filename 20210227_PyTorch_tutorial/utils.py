from abc import ABC, abstractmethod

import os, torch
import numpy as np
import pandas as pd

from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms


def compute_accuracy(pred, label):
    n_correct = pred == label
    total_correct = torch.sum(n_correct, axis=0).double()
    mean_correct = torch.mean(total_correct)

    return mean_correct / len(pred)

class Transform(ABC):
    @abstractmethod
    def __call__(self, data):
        raise NotImplemented

class RANZCR(Dataset):
    def __init__(self, data_root, data_list, transform=None):
        super(RANZCR, self).__init__()
        self.data_root = data_root
        self.data_list = data_list
        self.transform = transform
        self.file = "{}.jpg"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        data = {
            "patient_id": self.data_list.iloc[i, -1],
            "image": os.path.join(self.data_root, self.file.format(self.data_list.iloc[i, 0])),
            "label": np.array(self.data_list.iloc[i, 1:-1], dtype="float32")
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

class JPGLoader(Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                path = data[key]
                data[key] = Image.open(path)

            else:
                raise KeyError(f"{key} is a key of {data}.")

        return data

class Resize(Transform):
    def __init__(self, keys, size, interpolation=2):
        self.keys = keys
        self.resize = transforms.Resize(size, interpolation=interpolation)

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                image = data[key]
                data[key] = self.resize(image)

            else:
                raise KeyError(f"{key} is a key of {data}.")

        return data

class PILToTensor(Transform):
    def __init__(self, keys):
        self.keys = keys
        self.to_tensor = transforms.ToTensor()

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = self.to_tensor(data[key])

            else:
                raise KeyError(f"{key} is a key of {data}.")

        return data

class NumpyToTensor(Transform):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = torch.Tensor(data[key])

            else:
                raise KeyError(f"{key} is a key of {data}.")

        return data

class ExampleModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExampleModel, self).__init__()
        # input 224 x 224
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=4, padding=1) # 56 x 56
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1) # 14 x 14
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0) # 12 x 12
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=4, padding=1) # 3 x 3
        self.bn4 = nn.BatchNorm2d(64)
        self.flat = nn.Flatten() # 64 x 3 x 3 = 576
        self.linear = nn.Linear(576, out_channels)
        self.bn_out = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flat(x)
        x = torch.sigmoid(self.bn_out(self.linear(x)))

        return x