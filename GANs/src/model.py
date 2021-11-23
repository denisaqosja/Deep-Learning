"""
DCGAN for training STL10 dataset
"""


import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, size = [64, 128, 256, 512], latent_size=128):
        super().__init__()

        self.discriminator_conv = nn.Sequential(
            #3x96x96
            nn.Conv2d(in_channels=3, out_channels=size[0], kernel_size=3, stride=2, padding=1),
            #64x48x48
            nn.BatchNorm2d(size[0]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=size[0], out_channels=size[1], kernel_size=3, stride=2, padding=1),
            #128x24x24
            nn.BatchNorm2d(size[1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=size[1], out_channels=size[2], kernel_size=3, stride=2, padding=1),
            #256x12x12
            nn.BatchNorm2d(size[2]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=size[2], out_channels=size[3], kernel_size=3, stride=2),
            #512x5x5
        )

        self.discriminator_fc = nn.Sequential(
            nn.Linear(in_features=size[3]*5*5, out_features=1024, bias=False),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=latent_size, bias=False)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, input):
        out_conv = self.discriminator_conv(input)

        b_size = input.shape[0]
        out_flattened = out_conv.view(b_size, -1)
        output_flattened = self.discriminator_fc(out_flattened)
        output = self.sigmoid(output_flattened)

        return output


class Generator(nn.Module):
    def __init__(self, size=[512, 256, 128, 64], latent_size=128):
        super().__init__()
        self.size = size

        self.generator_fc = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=1024, bias=False),
            nn.Linear(in_features=1024, out_features=size[0] * 5 * 5, bias=False)
        )

        self.generator_conv = nn.Sequential(
            #512x5x5
            nn.ConvTranspose2d(in_channels=size[0], out_channels=size[1], kernel_size=3, stride=2),
            #256x11x11
            nn.BatchNorm2d(size[1]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=size[1], out_channels=size[2], kernel_size=3, stride=2),
            #128x23x23
            nn.BatchNorm2d(size[2]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=size[2], out_channels=size[3], kernel_size=3, stride=2),
            #64x47x47
            nn.BatchNorm2d(size[3]),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=size[3], out_channels=3, kernel_size=3, stride=2, output_padding=1),
            #3x96x96
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_vector):
        out_vector = self.generator_fc(input_vector)

        batch_size = input_vector.shape[0]
        image = out_vector.view(batch_size, self.size[0], 5, 5)

        output = self.generator_conv(image)

        return self.sigmoid(output)