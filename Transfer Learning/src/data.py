import matplotlib.pyplot as plt
import numpy as np
import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils


class Dataset():
    def __init__(self):
        self.batch_size = 32

        visualize = False
        if(visualize):
            self.visualize()


    def data_loader(self):
        dir = "../hymenoptera_data/"
        data_transforms = {
            "train": transforms.Compose(
                [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
            "val": transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        }

        self.dataset = {
            "train": datasets.ImageFolder(os.path.join(dir, "train"), transform=data_transforms["train"]),
            "val": datasets.ImageFolder(os.path.join(dir, "val"), transform=data_transforms["val"])
        }

        train_loader = torch.utils.data.DataLoader(self.dataset["train"], self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.dataset["val"], self.batch_size, shuffle=True)

        return train_loader, val_loader

    def cutMix(self):
        pass

    def show(self, batchId, images, titles):
        #convert images to numpy and transpose them to H, W, C
        images = images.numpy().transpose(0, 2, 3, 1)
        images = np.clip(images, 0, 1)

        rows = 4
        columns = 8
        #make a grid of size (8, 4), increase it with 1.5 to better fit the titles
        plt.figure(figsize=(columns*1.5, rows*1.5))
        grid = plt.GridSpec(rows, columns, wspace=.25, hspace=.25)
        for i in range(self.batch_size):
            exec(f"plt.subplot(grid{[i]})")
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis("off")

        plt.savefig(f"training_batch_{batchId}.png")

        return

    def visualize(self):

        self.train_loader, self.val_loader = self.data_loader()
        self.train_classes = self.dataset["train"].classes

        titles = []

        """
        #iterate through all batches
        for idx, (images, labels) in enumerate(self.train_loader):
            for label in labels:
                titles.append(self.train_classes[label])
            if(len(images) == self.batch_size):
                self.show(idx, images, titles)
        """

        #iterate through one batch
        images, labels = next(iter(self.train_loader))
        for label in labels:
            titles.append(self.train_classes[label])
        self.show(100, images, titles)
        return

