import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_dim, drop_rate):
        super(Classifier, self).__init__()
        self.model = torch.nn.Sequential(nn.Linear(input_dim, 1), nn.Dropout(drop_rate))

    def forward(self, encoded_e, encoded_m, encoded_c):
        integrated_test_omics = torch.cat((encoded_e, encoded_m, encoded_c), 1)
        output = self.model(integrated_test_omics)
        return output


class SupervisedEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(SupervisedEncoder, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        output = self.model(x)
        return output

    def encode(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(AutoEncoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

        self.decoder = nn.Linear(output_dim, input_dim)
        

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def encode(self, x):
        return self.encoder(x)
