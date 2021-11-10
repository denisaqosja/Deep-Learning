import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

import numpy as np
import time, os
from tqdm import tqdm

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
            "test_loss": []
        }

    def load_data(self):
        datasets = Dataset()
        self.train_loader = datasets.train_loader()
        self.test_loader = datasets.test_loader()

        return

    def setup_model(self):
        self.load_data()

        self.model = set_model("DenoisingAE")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        export_model(self.model)

        self.optimizer = set_optimizer(self.optimizerName, self.model.parameters(), self.learning_rate)
        self.regularizer = set_regularizer(self.regularizerName, self.model.parameters())
        self.lossFunction = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        return

    def train_model(self):
        self.train_loss = []
        self.test_loss = []

        self.output_images_train = []
        self.output_images_test = []

        for epoch in range(EPOCHS):
            start_training_time = time.time()
            print(f"--------------EPOCH {epoch}--------------")

            images_train, reconstructions_train = self.train_epoch()
            self.output_images_train.append((epoch, images_train, reconstructions_train),)
            print(f"Training loss: {self.train_loss[-1]}, training time: {time.time() - start_training_time}")

            images_test, reconstructions_test = self.test_epoch(epoch)
            self.output_images_test.append((epoch, images_test, reconstructions_test),)
            print(f"Testing loss: {self.test_loss[-1]}")

            self.scheduler.step()

            self.stats["train_loss"].append(self.train_loss[-1])
            self.stats["test_loss"].append(self.test_loss[-1])
            save_model(epoch, self.model, self.optimizer, self.test_loss[-1])

        plot_curves(stats=self.stats, epochs=EPOCHS)

        return

    def train_epoch(self):
        train_losses = []
        outputs_images = []
        outputs_recons = []
        self.model.train()

        for batchId, (images, _) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            images = images.to(self.device)

            recons = self.model(images)
            loss = self.lossFunction(recons, images)
            loss.backward()
            train_losses.append(loss.item())

            self.optimizer.step()
            outputs_images.append(images)
            outputs_recons.append(recons)

        self.train_loss.append(np.mean(train_losses))

        return outputs_images, outputs_recons

    @torch.no_grad()
    def test_epoch(self, epoch):
        test_losses = []

        outputs_images = []
        outputs_recons = []
        for batchId, (images, _) in enumerate(tqdm(self.test_loader)):
            images = images.to(self.device)

            recons = self.model(images)
            loss = self.lossFunction(recons, images)
            test_losses.append(loss.item())

            outputs_images.append(images)
            outputs_recons.append(recons)

            # save images of the first batch:
            if (batchId == 0):
                if not os.path.exists("images/"):
                    os.mkdir("images/")

                save_image(images[:32].cpu(), f"images/image_orig_epoch_{epoch}.png")
                save_image(recons[:32].cpu(), f"images/recons_epoch_{epoch}.png")

        self.test_loss.append(np.mean(test_losses))
        return outputs_images, outputs_recons


if __name__ == "__main__":
    trainer = Trainer()
    trainer.setup_model()
    trainer.train_model()

