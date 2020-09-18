import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(Encoder, self).__init__()
        self.En = torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        output = self.En(x)
        return output


class Classifier(nn.Module):
    def __init__(self, input_size, rate):
        super(Classifier, self).__init__()
        self.FC = torch.nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Dropout(rate))

    def forward(self, x):
        return self.FC(x)
