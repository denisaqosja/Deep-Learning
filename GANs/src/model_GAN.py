"""
GAN for training STL10 dataset
"""

import torch.nn as nn
import numpy as np

class GAN(nn.Module):
    def __init__(self, img_size=[3, 96, 96], hidden_size=[256, 512, 2048], latent_dim=100, output_dim=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim

class Discriminator(GAN):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(in_features=np.prod(self.img_size), out_features=self.hidden_size[2]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.hidden_size[2], out_features=self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.hidden_size[0], out_features=self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, input_img):
        #input is an image of size Bx3x96x96
        input_vec = input_img.view(input_img.shape[0], -1)
        output_disc = self.discriminator(input_vec)

        return output_disc

class Generator(GAN):
    def __init__(self):
        super().__init__()

        self.generator = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.hidden_size[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2]),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.hidden_size[2], out_features=np.prod(self.img_size)),
            nn.Tanh()
        )

    def forward(self, input_vec):
        #input is a vector of size Bx128
        output_vec = self.generator(input_vec)
        output_img = output_vec.view(input_vec.shape[0], self.img_size[0], self.img_size[1], self.img_size[2])

        return output_img