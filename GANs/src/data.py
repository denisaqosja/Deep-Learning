"""
Using 3 datasets: STL10, SVHN, CelebA

CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
"""

import torch
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


class STL10Dataset():
    def __init__(self):
        self.batch_size = 256
        self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def train_loader(self):
        train_dataset = datasets.STL10(root="../datasetSTL/", split="train", download=True, transform=self.transforms)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.plot_training_images(train_loader)

        return train_loader

    def test_loader(self):
        test_dataset = datasets.STL10(root="../datasetSTL/", split="test", download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return test_loader

    def plot_training_images(self, train_loader):
        images, _ = next(iter(train_loader))
        images_numpy = images.numpy().transpose(0, 2, 3, 1)
        std = mean = np.array([0.5, 0.5, 0.5])
        data = images_numpy * std + mean

        fig = plt.figure(figsize=(8 * 0.7, 4 * 0.7))
        for i in range(32):
            plt.subplot(4, 8, i + 1)
            plt.imshow(data[i])
            plt.axis("off")
        plt.suptitle("Training images")
        # plt.show()
        fig.savefig("../training_images_STL10.png")

