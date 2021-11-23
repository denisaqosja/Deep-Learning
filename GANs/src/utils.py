import torch.optim as optim
import torch

import os, json
import matplotlib.pyplot as plt
import numpy as np

"""
from model import Discriminator, Generator

def set_model(required_model):
    if (required_model == "DenoisingAE"):
        model = DAE()
    elif (required_model == "VariationalAE"):
        model = VAE()

    return model
"""

def vectors_to_images(vector, shape):
    img = vector.view(shape)
    return img

def images_to_vectors(images):
    b_size = images.shape[0]
    vec = images.view(b_size, -1)
    return vec

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

def export_model(model1, model2):
    with open("network_architecture.txt", "w") as file:
        file.write(str(model1.children))
        file.write(str(model2.children))

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
    fig, ax = plt.subplots(2)

    epochs = np.arange(epochs)
    ax[0].plot(epochs, stats["discriminator_loss"], c="green", label="Discriminator loss")
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")

    ax[1].plot(epochs, stats["generator_loss"], c="blue", label="Generator loss")
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")

    fig.suptitle("Models Loss Curves")

    fig.savefig("loss_curves.png", dpi=300)
    plt.show()
