"""
DCGAN for training STL10 dataset
"""

import torch.nn as nn
import torch.nn.functional as F

class DCGAN(nn.Module):
    def __init__(self, img_size=[3, 96, 96], hidden_size=[64, 128, 256, 512], latent_dim=100, output_dim=2, wgan=False):
        super().__init__()
        self.size = hidden_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.img_size = img_size
        self.wgan = wgan


class Discriminator_dcgan(DCGAN):
    def __init__(self):
        super().__init__()

        self.discriminator_conv = nn.Sequential(
            #3x96x96
            nn.Conv2d(in_channels=3, out_channels=self.size[0], kernel_size=3, stride=2, padding=1),
            #64x48x48
            nn.BatchNorm2d(self.size[0]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.size[0], out_channels=self.size[1], kernel_size=3, stride=2, padding=1),
            #128x24x24
            nn.BatchNorm2d(self.size[1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.size[1], out_channels=self.size[2], kernel_size=3, stride=2, padding=1),
            #256x12x12
            nn.BatchNorm2d(self.size[2]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=self.size[2], out_channels=self.size[3], kernel_size=3, stride=2),
            #512x5x5
        )

        self.discriminator_fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.size[3], out_features=self.output_dim, bias=False)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        out_conv = self.discriminator_conv(input)

        #global average pooling: out = out.mean([2,3]) or:
        output = F.adaptive_avg_pool2d(out_conv, (1,1))
        output_flatten = self.discriminator_fc(output.view(-1, self.size[3]))

        output = output_flatten if self.wgan else self.sigmoid(output_flatten)

        return output


class Generator_dcgan(DCGAN):
    def __init__(self):
        super().__init__()

        self.generator_fc = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.size[3] * 5 * 5, bias=False),
            nn.BatchNorm1d(self.size[3] * 5 * 5),
            nn.ReLU()
        )

        self.generator_conv = nn.Sequential(
            #512x5x5
            nn.ConvTranspose2d(in_channels=self.size[3], out_channels=self.size[2], kernel_size=3, stride=2),
            #256x11x11
            nn.BatchNorm2d(self.size[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.size[2], out_channels=self.size[1], kernel_size=3, stride=2),
            #128x23x23
            nn.BatchNorm2d(self.size[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.size[1], out_channels=self.size[0], kernel_size=3, stride=2),
            #64x47x47
            nn.BatchNorm2d(self.size[0]),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.size[0], out_channels=3, kernel_size=3, stride=2, output_padding=1),
            #3x96x96
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

        self.tanh = nn.Tanh()

    def forward(self, input_vector):
        out_vector = self.generator_fc(input_vector)

        image = out_vector.view(-1, self.size[3], 5, 5)

        output = self.generator_conv(image)

        return self.tanh(output)