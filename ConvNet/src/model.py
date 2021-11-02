"""
Building a ConvNet
"""

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        """
        these are the CNN layers
        """

        super().__init__()

        layers = []

        #layer 1
        #3x32x32
        cnn = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        #16x28x28
        relu = nn.ReLU(inplace=True)
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #16x14x14

        self.layer1 = nn.Sequential(cnn, relu, pool)
        layers.append(self.layer1)

        #layer2
        cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,  padding=0)
        #32x10x10
        relu2= nn.ReLU()
        pool2= nn.MaxPool2d(kernel_size=2, stride=2)
        #32x5x5
        self.layer2 = nn.Sequential(cnn2, relu2, pool2)
        layers.append(self.layer2)

        #F.C for classification
        input_dim = 5*5*32
        self.dropout = nn.Dropout(p=0.3)
        layers.append(self.dropout)

        self.fc1 = nn.Linear(in_features=input_dim, out_features=10)
        layers.append(self.fc1)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.shape[0]
        out1 = self.layer1(x)
        out2 = self.layer2(out1)

        out_flattened = out2.view(batch_size, -1)
        out_flattened = self.dropout(out_flattened)
        y = self.fc1(out_flattened)

        return y


