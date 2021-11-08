import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time

from data import Dataset
from utils import set_model, set_optimizer, set_regularizer
from utils import save_model, export_model, save_stats, plot_curves

EPOCHS = 15

class Trainer():
    def __init__(self, optimizerName="Adam", regularizerName="L2"):
        self.learning_rate = 3e-4
        self.optimizerName = optimizerName
        self.regularizerName = regularizerName

        self.stats = {
            "train_loss": [],
            "valid_loss": [],
            "accuracy": []
        }

    def load_data(self):
        datasets = Dataset()
        self.train_loader = datasets.train_loader()
        self.test_loader = datasets.test_loader()

        return

    def setup_model(self):
        self.load_data()

        self.model = set_model("LSTM_scratch")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        export_model(self.model)

        self.optimizer = set_optimizer(self.optimizerName, self.model.parameters(), self.learning_rate)
        self.regularizer = set_regularizer(self.regularizerName, self.model.parameters())
        self.lossFunction = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        return

    def train_model(self):
        for epoch in range(EPOCHS):
            start_training_time = time.time()
            print(f"--------------EPOCH {epoch}--------------")
            train_loss = self.train_epoch()
            print(f"Training loss: {train_loss}, training time: {time.time() - start_training_time}")
            test_loss, test_acc = self.test_epoch()
            print(f"Testing loss: {test_loss}, accuracy: {test_acc}")

            self.scheduler.step()

            self.stats["train_loss"].append(train_loss)
            self.stats["valid_loss"].append(test_loss)
            self.stats["accuracy"].append(test_acc)

            save_model(epoch, self.model, self.optimizer, test_loss)

        save_stats(stats=self.stats)
        plot_curves(stats=self.stats, epochs=EPOCHS)

        return

    def train_epoch(self):
        train_losses = []

        self.model.train()

        for batchId, (images, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)

            loss = self.lossFunction(outputs, labels)
            loss.backward(retain_graph=True)
            train_losses.append(loss.item())

            self.optimizer.step()

        train_loss = np.mean(train_losses)

        return train_loss

    @torch.no_grad()
    def test_epoch(self):
        test_losses = []
        correct_pred = 0

        for batchId, (images, labels) in enumerate(self.test_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            output = self.model(images)
            loss = self.lossFunction(output, labels)
            test_losses.append(loss.item())

            predictions = torch.argmax(output, axis = 1)
            correct_pred += predictions.eq(labels).sum().item()

        total_length = len(self.test_loader.dataset)
        accuracy = correct_pred * 100 / total_length
        test_loss = np.mean(test_losses)

        return test_loss, accuracy


if __name__=="__main__":
    trainer = Trainer()
    trainer.setup_model()
    trainer.train_model()

