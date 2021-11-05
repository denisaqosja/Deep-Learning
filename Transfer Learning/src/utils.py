import torch.optim as optim
import torch

import os

from model import VGG, ResNet, DenseNet

def set_model(required_model):
    if (required_model == "VGG"):
        m = VGG()
    elif (required_model == "ResNet"):
        m = ResNet()
    elif (required_model == "DenseNet"):
        m = DenseNet()

    return m.model()

def set_optimizer(optimizerName, parameters, learning_rate):
    if (optimizerName == "Adam"):
        optimizer = optim.Adam(parameters, learning_rate)
    elif (optimizerName == "SGD"):
        optimizer = optim.SGD(parameters, learning_rate)
    elif (optimizerName == "SGD_momentum"):
        optimizer = optim.SGD(parameters, learning_rate, momentum=0.95)
    elif (optimizerName == "RMSprop"):
        optimizer = optim.RMSprop(parameters, learning_rate)

    return optimizer

def set_regularizer(regularizerName, parameters):

    l_lambda = 1e-4

    if(regularizerName == "L2"):
        l2_norm = 0
        for param in parameters:
            l2_norm += param.pow(2).sum()
        loss_penalty = l_lambda * l2_norm

    elif(regularizerName == "L1"):
        l1_norm = 0
        for param in parameters:
            l1_norm += param.abs().sum()

        loss_penalty = l_lambda * l1_norm

    return loss_penalty


def save_stats(train_loss, test_loss, test_acc):
    stats = {
        "Train_loss": train_loss,
        "Test_loss": test_loss,
        "Accuracy": test_acc
    }

def export_model(model):
    with open("network_architecture.txt", "w") as file:
        file.write(str(model.children))

def savepath(epoch):
    if (not os.path.exists("../models/")):
        os.makedirs("../models/")
    path = os.path.join("../models/", f"checkpoint_epoch_{epoch}")

    return path

def save_model(epoch, model, optimizer, test_loss):
    path = savepath(epoch)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': test_loss
    }

    torch.save(checkpoint, path)

def load_model(model, optimizer):
    for epoch in range(15):
        path = savepath(epoch)
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        val_loss = checkpoint["valid_loss"]

        return model, optimizer, val_loss