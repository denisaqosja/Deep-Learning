
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

import numpy as np

BS = 128

def split_data(valid_size = 0.1):
    data = datasets.CIFAR10("../data/", train=True, download=True, transform=transforms.ToTensor())

    data_size = len(data)
    split = int(np.floor(data_size * valid_size))
    indices = list(range(data_size))

    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_datasplit = SubsetRandomSampler(train_indices)
    valid_datasplit = SubsetRandomSampler(val_indices)

    return data, train_datasplit, valid_datasplit

def load_train_dataset():
    data, train_datasplit, valid_datasplit = split_data(0.1)

    trainLoader = DataLoader(data, batch_size=BS, sampler=train_datasplit)
    validLoader = DataLoader(data, batch_size=BS, sampler=valid_datasplit)

    return trainLoader, validLoader

def load_test_dataset():
    data = datasets.CIFAR10("../data/", train=False, download=True, transform=transforms.ToTensor())
    trainLoader = DataLoader(data, batch_size=BS, shuffle=False)
    return trainLoader


"""
rom torchvision import datasets
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
"""