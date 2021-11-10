import torch.nn as nn
import torch


class DAE(nn.Module):
    def __init__(self, size=[128, 64, 32], latent_dim=16):
        super().__init__()
        self.size = size

        self.encoder = nn.Sequential(
            #1x28x28
            nn.Conv2d(in_channels=1, out_channels=size[0], kernel_size=5, stride=1),
            nn.ReLU(),
            #128x24x24
            nn.Conv2d(in_channels=size[0], out_channels=size[1], kernel_size=5, stride=1),
            #64x20x20
            nn.ReLU(),
            nn.Conv2d(in_channels=size[1], out_channels=size[2], kernel_size=5, stride=1),
            #32x20x20
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 32x8x8

        self.fc = nn.Sequential(
            #encoder
            nn.Linear(in_features=size[2]*8*8, out_features=latent_dim),
            #dec
            nn.Linear(in_features=latent_dim, out_features=size[2]*8*8)
        )

        self.out_layer = nn.MaxUnpool2d(kernel_size=2, stride=2),

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=size[2], out_channels=size[1], kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=size[1], out_channels=size[0], kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=size[0], out_channels=1, kernel_size=5),
            nn.Sigmoid()

        )


    def forward(self, input):
        noisy_input = self.add_noise(input)
        out_enc = self.encoder(noisy_input)
        out_pool, out_indices = self.pool(out_enc)

        batch_size = out_enc.shape[0]
        out_flattened = out_pool.view(batch_size, -1)
        out_ = self.fc(out_flattened)
        out_unflattened = out_.view(batch_size, self.size[2], 8, 8)

        unpooled_out = nn.MaxUnpool2d(kernel_size=2, stride=2)(out_unflattened, out_indices)
        recons = self.decoder(unpooled_out)

        return recons

    def add_noise(self, input):
        mean_vec = 0 * torch.ones(input.shape)
        std = 0.5
        noise = torch.normal(mean_vec, std)

        noisy_input = input + noise
        return noisy_input.clamp(0,1)

class VAE(nn.Module):
    def __init__(self, size=[128, 64, 32], latent_dim=16):
        super().__init__()
        self.size = size

        self.encoder = nn.Sequential(
            # 1x28x28
            nn.Conv2d(in_channels=1, out_channels=size[0], kernel_size=5, stride=1),
            nn.ReLU(),
            # 128x24x24
            nn.Conv2d(in_channels=size[0], out_channels=size[1], kernel_size=5, stride=1),
            # 64x20x20
            nn.ReLU(),
            nn.Conv2d(in_channels=size[1], out_channels=size[2], kernel_size=5, stride=1),
            # 32x20x20
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # 32x8x8

        self.fc = nn.Sequential(
            # encoder
            nn.Linear(in_features=size[2]*8*8, out_features=latent_dim),
            # dec
            nn.Linear(in_features=latent_dim, out_features=size[2]*8*8)
        )

        self.fc1_enc = nn.Linear(in_features=size[2]*8*8, out_features=128)
        self.fc2_mean = nn.Linear(in_features=128, out_features=latent_dim)
        self.fc2_var = nn.Linear(in_features=128, out_features=latent_dim)

        self.fc_dec1 = nn.Linear(in_features=latent_dim, out_features=128)
        self.fc_dec2 = nn.Linear(in_features=128, out_features=size[2]*8*8)

        self.unpool_layer = nn.MaxUnpool2d(kernel_size=2, stride=2),

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=size[2], out_channels=size[1], kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=size[1], out_channels=size[0], kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=size[0], out_channels=1, kernel_size=5),
            nn.Sigmoid()
        )

    def reparametrize(self, mean, var):
        eps = torch.randn(var.shape)
        z = eps * var + mean
        return z

    def forward(self, input):
        out_enc = self.encoder(input)
        out_pool, out_indices = self.pool(out_enc)

        # flattening
        batch_size = out_pool.shape[0]
        out_flattened = out_pool.view(batch_size, -1)

        out_fc1 = self.fc1_enc(out_flattened)
        mean = self.fc2_mean(out_fc1)
        var = self.fc2_var(out_fc1)
        z = self.reparametrize(mean, var)

        out_fc_dec1 = self.fc_dec1(z)
        out_fc_dec2 = self.fc_dec2(out_fc_dec1)

        # unflattening
        out_unflattened = out_fc_dec2.view(batch_size, self.size[2], 8, 8)

        out_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)(out_unflattened, out_indices)
        recons = self.decoder(out_unpool)

        return recons
