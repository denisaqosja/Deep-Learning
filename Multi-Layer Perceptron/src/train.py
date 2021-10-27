import torch.nn as nn
import torch.optim as optimizer
import torch

from tqdm import tqdm
import numpy as np

from model import MLP
from data import load_train_dataset
from data import load_test_dataset

EPOCHS = 20

class Trainer:
    def __init__(self):

        self.lr = 3e-3
        self.setup_model()
        self.train_model()


    def setup_model(self):
        self.model = MLP()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer.Adam(self.model.parameters(), lr = self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        return

    def train_model(self):
        self.train_loss = []
        self.test_loss = []

        for epoch in range(EPOCHS):
            print(f"----------EPOCH {epoch}-----------")
            self.train_loss.append(self.train_epoch())
            print(f"Train Loss: {self.train_loss[-1]}")

            self.test_loss.append(self.test_epoch())
            print(f"Test Loss: {self.test_loss[-1]}")


    def train_epoch(self):

        train_losses = []

        self.model.train()

        for batchId, (images, labels) in enumerate(tqdm(load_train_dataset())):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)
            output = self.model(images)
            loss = self.loss(output, labels)
            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step()

        train_loss = np.mean(train_losses)

        return train_loss

    @torch.no_grad()
    def test_epoch(self):
        test_losses = []
        correct_preds = 0

        self.model.eval()
        test_dataLoader = load_test_dataset()

        for id, (images, labels) in enumerate(tqdm(test_dataLoader)):
            images, labels = images.to(self.device), labels.to(self.device)

            output = self.model(images)
            loss = self.loss(output, labels)

            test_losses.append(loss.item())

            predictions = torch.argmax(output, axis = 1)

            correct_preds+= predictions.eq(labels).sum().item()

        test_loss = np.mean(test_losses)
        total_length = len(test_dataLoader.dataset)
        self.test_acc = correct_preds*100/total_length

        print(f"Test accuracy: {self.test_acc}%")

        return test_loss

train = Trainer()