import torch
import torch.optim as optim

import os

from regularization import l1_regularizer, l2_regularizer, elastic_regularizer
from model import CNN


def number_of_model_parameters(parameters):
    sum = 0
    for param in parameters:
        sum += param.numel()
    return sum

def export_model():
    model = CNN()
    with open("network_architecture.txt", "w") as file:
        for layer in model.layers:
            file.write(str(layer))
            file.write("\n")

def save_model(path, epoch, model, optimizer, val_loss):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'valid_loss': val_loss
        }

    torch.save(checkpoint, path)

def load_model(path, model, optimizer):
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    valid_loss = checkpoint["valid_loss"]

    return model, optimizer, epoch, valid_loss


def set_optimizer(optimizerName, parameters, learning_rate):
    if(optimizerName == "SGD"):
        optimizer = optim.SGD(parameters, learning_rate)
    elif(optimizerName == "SGD_momentum"):
        optimizer = optim.SGD(parameters, learning_rate, momentum=0.95)
    elif(optimizerName == "RMSprop"):
        optimizer = optim.RMSprop(parameters, learning_rate)
    elif(optimizerName == "Adam"):
        optimizer = optim.Adam(parameters, learning_rate)

    return optimizer

def set_regularizer(regularizerName, parameters):
    if(regularizerName == "L1"):
        norm = l1_regularizer(parameters)
    elif(regularizerName == "L2"):
        norm = l2_regularizer(parameters)
    elif(regularizerName == "Elastic"):
        norm = elastic_regularizer(parameters)
    elif regularizerName == None:
        norm = 0.0

    return norm
