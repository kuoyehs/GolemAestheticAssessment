# -*- coding: utf-8

import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data


class PredDataset(data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = self.data
        if self.transform:
            image = self.transform(image)

        return image
