import torch
from torchvision import datasets, transforms

class Dataset():
    def __init__(self):
        self.batch_size = 256
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])

    def train_loader(self):
        train_dataset = datasets.FashionMNIST(root="../data/", train=True, download=True, transform=self.transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True)

        return train_loader

    def test_loader(self):
        test_dataset = datasets.FashionMNIST(root="../data/", train=False, download=True, transform=transforms.ToTensor() )
        test_loader = torch.utils.data.DataLoader(test_dataset, self.batch_size, shuffle=False)

        return test_loader
