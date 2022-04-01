import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(Encoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.encoder(x)


class LinearClassifier(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(LinearClassifier, self).__init__()
        self.classify = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.classify(x)


class Momi(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates, combination, depths, noisy):
        super(Momi, self).__init__()

    def forward(self, expression, mutation, cna):
        pass