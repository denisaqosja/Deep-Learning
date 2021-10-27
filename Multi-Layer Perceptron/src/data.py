from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import numpy as np

BS = 128

def load_train_dataset():
    data = datasets.CIFAR10("../data/", train=True, download=True, transform=transforms.ToTensor())
    trainLoader = DataLoader(data, batch_size=BS, shuffle=True)
    return trainLoader

def load_test_dataset():
    data = datasets.CIFAR10("../data/", train=False, download=True, transform=transforms.ToTensor())
    trainLoader = DataLoader(data, batch_size=BS, shuffle=False)
    return trainLoader