"""
Training a NN
"""
from data import Dataset
from model import CNN
from utils import set_optimizer, set_regularizer
from utils import export_model, save_model, load_model

import torch.nn as nn
import torch

import numpy as np
from tqdm import tqdm
import time
import os

EPOCHS = 100

class Trainer:
    def __init__(self, optimizerName = "Adam", lr = 3e-4, regularizerName = "L2"):
        self.lr = lr
        self.optimizerName = optimizerName
        self.regularizerName = regularizerName

        #A dictionary with statistics: #TODO
        self.test_acc = 0
        self.test_loss = []
        self.train_loss = []

    def load_data(self):
        dataset = Dataset()
        self.test_data = dataset.test_data()
        self.train_data = dataset.train_data()

        return

    def setup_model(self):
        self.load_data()

        self.model = CNN()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.optimizer = set_optimizer(self.optimizerName, self.model.parameters(), self.lr)
        self.regularizer = set_regularizer(self.regularizerName, self.model.parameters())
        self.lossFunction = nn.CrossEntropyLoss()

        return

    def train_model(self):

        for epoch in range(EPOCHS):

            print(f"------------Epoch {epoch}-----------")

            start_training_time = time.time()
            self.train_epoch()
            print(f"TRAIN: Loss = {self.train_loss[-1]}, Training time (s) = {time.time() - start_training_time}")

            self.eval_epoch()
            print(f"TEST: Loss = {self.test_loss[-1]}, Test Accuracy = {self.test_acc}%")

            #save checkpoint
            if not os.path.exists("models"):
                os.makedirs("models")

            path = os.path.join("models", "checkpoints.pth")

            save_model(path, epoch, self.model, self.optimizer, self.test_loss[-1])


    def train_epoch(self):

        train_losses = []

        self.model.train()

        for batchId, (image, label) in enumerate(tqdm(self.train_data)):
            self.optimizer.zero_grad()
            image, label = image.to(self.device), label.to(self.device)
            output = self.model(image)
            loss = self.lossFunction(output, label) + self.regularizer
            loss.backward(retain_graph=True)
            train_losses.append(loss.item())

            self.optimizer.step()

        self.train_loss.append(np.mean(train_losses))

        return

    @torch.no_grad()
    def eval_epoch(self):

        test_losses = []
        correct_preds = 0

        self.model.eval()

        for id, (image, label) in enumerate(tqdm(self.test_data)):
            image, label = image.to(self.device), label.to(self.device)

            output = self.model(image)
            loss = self.lossFunction(output, label)
            test_losses.append(loss.item())

            prediction = torch.argmax(output, axis = 1)
            correct_preds += prediction.eq(label).sum().item()

        self.test_loss.append(np.mean(test_losses))
        total_length = len(self.test_data.dataset)
        self.test_acc = correct_preds * 100 / total_length

        return

if __name__ == "__main__":
    trainer = Trainer()
    trainer.setup_model()
    export_model()
    trainer.train_model()