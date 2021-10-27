"""
Multi-layer perceptron
"""
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=3072,  hidden_dim=[256, 64], output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(in_features = self.input_dim, out_features = self.hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim[0], out_features=self.hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_dim[1], out_features=self.output_dim)
        )

    def forward(self, x):
        input_flattened = x.view(-1, self.input_dim)
        output = self.model(input_flattened)

        return output