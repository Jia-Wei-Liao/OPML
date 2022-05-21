import os
from abc import abstractmethod, ABC

import torch
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import functional as F

class Transform(ABC):

    @abstractmethod
    def __call__(self, data):
        pass


class Identity(Transform):

    def __call__(self, data):
        return data


class ReadImage(Transform):

    def __init__(self, keys, data_root=""):
        self.keys      = (keys,) if isinstance(keys, str) else keys
        self.data_root = data_root

    def __call__(self, data):
        for key in self.keys:
            file_path = os.path.join(self.data_root, data[key])
            data[key] = F.convert_image_dtype(read_image(file_path))

        return data


class Resize(Transform):

    def __init__(self, keys, size):
        self.keys = (keys,) if isinstance(keys, str) else keys
        self.size = size

    def __call__(self, data):
        for key in self.keys:
            data[key] = F.resize(data[key], self.size)

        return data


class ToTensor(Transform):

    def __init__(self, keys):
        self.keys = (keys,) if isinstance(keys, str) else keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = torch.tensor(np.array(data[key]))

        return data