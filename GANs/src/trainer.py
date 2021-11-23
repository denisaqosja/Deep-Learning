import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

import numpy as np
import time, os
from tqdm import tqdm

from data import STL10Dataset
from model import Discriminator, Generator
from utils import set_optimizer
from utils import save_model, export_model, save_stats, plot_curves

EPOCHS = 100


class Trainer:
    def __init__(self, optimizerName="Adam"):
        self.learning_rate = 3e-4
        self.optimizerName = optimizerName
        self.latent_size = 128

        self.stats = {
            "discriminator_loss": [],
            "generator_loss": []
        }

    def load_data(self):
        datasets = STL10Dataset()
        self.train_loader = datasets.train_loader()
        self.test_loader = datasets.test_loader()

        return

    def setup_model(self):
        self.load_data()

        self.discriminator = Discriminator()
        self.generator = Generator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)
        export_model(self.discriminator, self.generator)

        self.optimizerDisc = set_optimizer(self.optimizerName, self.discriminator.parameters(), self.learning_rate)
        self.optimizerGen = set_optimizer(self.optimizerName, self.generator.parameters(), self.learning_rate)

        self.lossFunction = nn.BCELoss()
        self.schedulerDisc = optim.lr_scheduler.MultiStepLR(self.optimizerDisc, milestones=[25, 50, 75], gamma=0.1)
        self.schedulerGen = optim.lr_scheduler.MultiStepLR(self.optimizerGen, milestones=[25, 50, 75], gamma=0.1)

        return


    def train_models(self):
        self.lossDiscriminator = []
        self.lossGenerator = []

        for epoch in range(EPOCHS):
            self.train_epoch()
            print(f"--------------Epoch {epoch}----------------")
            self.stats["discriminator_loss"].append(self.lossDiscriminator[-1])
            self.stats["generator_loss"].append(self.lossGenerator[-1])
            print(self.lossDiscriminator[-1])
            print(self.lossGenerator[-1])

            #generating images using Generator model
            generated_images = self.testing()

            self.schedulerDisc.step()
            self.schedulerGen.step()

            #plot images after each epoch
            if not os.path.exists("../generated_images/"):
                os.mkdir("../generated_images/")
            save_image(generated_images[:16].cpu(), f"../generated_images/images_epoch_{epoch}.png")


        plot_curves(stats=self.stats, epoch=EPOCHS)

        return


    def train_epoch(self):
        lossDiscriminator = []
        lossGenerator = []
        for batchId, (images, _) in enumerate(tqdm(self.train_loader)):
            """ train discriminator """
            self.discriminator.train()
            self.optimizerDisc.zero_grad()

            """ train on real data """
            prediction_real = self.discriminator(images)
            loss_real_data = self.lossFunction(prediction_real, torch.ones_like(prediction_real))

            """ train on fake data """
            latent_space = torch.randn(images.shape[0], self.latent_size)
            fake_data = self.generator(latent_space)
            prediction_fake = self.discriminator(fake_data.detach())
            loss_fake_data = self.lossFunction(prediction_fake, torch.zeros_like(prediction_fake))

            loss_d = loss_real_data + loss_fake_data
            lossDiscriminator.append(loss_d.item())
            loss_d.backward()
            self.optimizerDisc.step()

            """ train generator """
            self.generator.train()
            self.optimizerGen.zero_grad()

            prediction_discriminator = self.discriminator(fake_data)
            loss_fake_data_gen = self.lossFunction(prediction_discriminator, torch.ones_like(prediction_discriminator))
            lossGenerator.append(loss_fake_data_gen.item())
            loss_fake_data_gen.backward()
            self.optimizerGen.step()

        self.lossDiscriminator.append(np.mean(lossDiscriminator))
        self.lossGenerator.append(np.mean(lossGenerator))

        return

    @torch.no_grad()
    def testing(self):
        """ generate data from Generator """
        self.generator.eval()
        batch_size = self.test_loader.batch_size
        latent_space = torch.randn(batch_size, self.latent_size)
        generated_images = self.generator(latent_space)

        return generated_images


if __name__ == "__main__":
    trainer = Trainer()
    trainer.setup_model()
    trainer.train_models()
