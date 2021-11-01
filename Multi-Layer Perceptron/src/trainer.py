import torch.nn as nn
import torch

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser
import os
import pickle

from model import MLP
from data import load_train_dataset
from data import load_test_dataset
from utils import load_optimizer

EPOCHS = 20

class Trainer:
    def __init__(self, optimizer="Adam", lr=3e-3, **kwargs):
        self.lr = lr
        self.optimizerName = optimizer


    def load_data(self, split):
        if split == "Train":
            dataLoader = load_train_dataset()[0]
        elif split == "Valid":
            dataLoader = load_train_dataset()[1]
        elif split=="Test":
            dataLoader = load_test_dataset()

        return dataLoader


    def setup_model(self):
        self.model = MLP()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = load_optimizer(optimizerName=self.optimizerName, parameters=self.model.parameters(), lr=self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        return

    def train_model(self):
        self.train_loss = []
        self.valid_loss = []

        self.labelList = []
        self.predictionList = []

        for epoch in range(EPOCHS):
            print(f"----------EPOCH {epoch}-----------")
            self.train_loss.append(self.train_epoch())
            print(f"Train Loss: {self.train_loss[-1]}")

            self.valid_loss.append(self.valid_epoch())
            print(f"Validation Loss: {self.valid_loss[-1]}")

        plt.plot(range(EPOCHS), self.train_loss)
        plt.show()


    def train_epoch(self):

        train_losses = []

        self.model.train()

        for batchId, (images, labels) in enumerate(tqdm(self.load_data("Train"))):
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
    def valid_epoch(self):
        valid_losses = []
        correct_preds = 0

        self.model.eval()

        labels_lst = torch.zeros(0, dtype=torch.long)
        preds_lst = torch.zeros(0, dtype=torch.long)

        for id, (images, labels) in enumerate(tqdm(self.load_data("Valid"))):
            images, labels = images.to(self.device), labels.to(self.device)

            output = self.model(images)
            loss = self.loss(output, labels)

            valid_losses.append(loss.item())
            predictions = torch.argmax(output, axis = 1)
            correct_preds+= predictions.eq(labels).sum().item()

            labels_lst = torch.cat([labels_lst, labels], dim=0)
            preds_lst = torch.cat([preds_lst, predictions], dim=0)

        valid_loss = np.mean(valid_losses)
        total_length = len(labels_lst)
        self.valid_acc = correct_preds*100/total_length

        print(f"Validation accuracy: {self.valid_acc}%")

        return valid_loss

    @torch.no_grad()
    def test_model(self):
        test_losses=[]
        correct_preds = 0

        labels_lst = torch.zeros(0, dtype=torch.long)
        preds_lst = torch.zeros(0, dtype=torch.long)

        for id, (images, labels) in enumerate(tqdm(self.load_data("Test"))):
            images, labels = images.to(self.device), labels.to(self.device)

            output = self.model(images)
            loss = self.loss(output, labels)

            test_losses.append(loss.item())
            predictions = torch.argmax(output, axis = 1)
            correct_preds+= predictions.eq(labels).sum().item()

            labels_lst = torch.cat([labels_lst, labels], dim=0)
            preds_lst = torch.cat([preds_lst, predictions.view(-1)])

        test_loss = np.mean(test_losses)
        total_length = len(labels_lst)
        self.test_acc = correct_preds*100/total_length
        self.confusion_matrix = confusion_matrix(labels_lst, preds_lst)

        print(f"Test accuracy: {self.test_acc}%")
        print(f"Confusion Matrix: \n{self.confusion_matrix}")

        plt.imshow(self.confusion_matrix, interpolation="nearest")
        plt.title("Confusion matrix")
        plt.colorbar()
        plt.show()

        return test_loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--path",
                        default=None,
                        help="load parameters from file")
    args = parser.parse_args()

    if(args.path is None):
        trainer = Trainer()
    else:
        if(os.path.exists(args.path)):

            with open("study.pkl", "rb") as paramsFile:
                study = pickle.load(paramsFile)

            best_params = study.best_trial.params
            trainer = Trainer(**best_params)
            print(f"The parameters selected from optuna, {best_params = }")

        else:
            raise Exception(f"File does not exist, {args.path = }")


    trainer.setup_model()
    trainer.train_model()
    trainer.test_model()