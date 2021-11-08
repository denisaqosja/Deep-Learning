import torch.optim as optim
import torch

import os, json
import matplotlib.pyplot as plt
import numpy as np

from model import LSTM, LSTM_scratch, GRU

def set_model(required_model):
    if (required_model == "LSTM"):
        model = LSTM(input_dim=28, hidden_dim=64, output_dim=10, num_layers=2)
    elif (required_model == "LSTM_scratch"):
        model = LSTM_scratch(input_dim=28, hidden_dim=64, output_dim=10, num_layers=1)
    elif (required_model == "GRU"):
        model = GRU()

    return model

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

def load_model(model, optimizer, epochs):
    for epoch in range(epochs):
        path = savepath(epoch)
        checkpoint = torch.load(path)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        val_loss = checkpoint["valid_loss"]

        return model, optimizer, val_loss

def save_stats(stats):
    savepath = os.path.join(os.getcwd(), "training_logs.json")
    logs = {
        "loss": {
            "train": stats["train_loss"],
            "test": stats["valid_loss"],
        },
        "accuracy": {
            "valid": stats["accuracy"],
        }
    }
    with open(savepath, "w") as f:
        json.dump(logs, f)
    return

def plot_curves(stats, epochs):
    plt.style.use("seaborn")
    fig, ax = plt.subplots(1, 2)

    epochs = np.arange(epochs)
    ax[0].plot(epochs, stats["train_loss"], c="green", label="Train loss")
    ax[0].plot(epochs, stats["valid_loss"], c="blue", label="Test loss")
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Loss curves")

    ax[1].plot(epochs, stats["accuracy"], c="red", label="Accuracy")
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Validation accuracy")

    plt.show()