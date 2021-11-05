import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, fixed=False):
        super().__init__()
        self.fixed = fixed

    def model(self):
        vgg = models.vgg16(pretrained=True)

        if (self.fixed):
            for param in vgg.parameters():
                param.requires_grad = False

        input_dim = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Linear(input_dim, 2)

        return vgg


class ResNet(nn.Module):
    def __init__(self, fixed=False):
        super().__init__()
        self.fixed = fixed

    def model(self, svm_wanted=False):
        resNet = models.resnet18(pretrained=True)

        if (self.fixed):
            for param in resNet.parameters():
                param.requires_grad = False

        if(svm_wanted):
            identity = Identity()
            resNet.fc = identity
        else:
            input_dim = resNet.fc.in_features
            resNet.fc = nn.Linear(input_dim, 2)

        return resNet


class DenseNet(nn.Module):
    def __init__(self, fixed=False):
        super().__init__()
        self.fixed = fixed

    def model(self):
        denseNet = models.densenet121(pretrained=True)

        if (self.fixed):
            for param in denseNet.parameters():
                param.requires_grad = False

        input_dim = denseNet.classifier.in_features
        denseNet.classifier = nn.Linear(input_dim, 2)

        return denseNet


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x