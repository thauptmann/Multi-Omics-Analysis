import torch
from torch import nn
import torch.nn.functional as F


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
    def __init__(self, input_size, dropout_rate):
        super(Classifier, self).__init__()
        self.FC = torch.nn.Sequential(
            nn.Linear(input_size, 1),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.FC(x)


class Moli(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates):
        super(Moli, self).__init__()
        self.expression_encoder = Encoder(input_sizes[0], output_sizes[0], dropout_rates[0])
        self.mutation_encoder = Encoder(input_sizes[1], output_sizes[1], dropout_rates[1])
        self.cna_encoder = Encoder(input_sizes[2], output_sizes[2], dropout_rates[2])
        z_in = sum(output_sizes)
        self.classifier = Classifier(z_in, dropout_rates[3])

    def forward(self, expression, mutation, cna):
        expression_out = self.expression_encoder(expression)
        mutation_out = self.mutation_encoder(mutation)
        cna_out = self.cna_encoder(cna)
        zt = torch.cat((expression_out, mutation_out, cna_out), 1)
        zt = F.normalize(zt, p=2, dim=0)
        return self.classifier(zt), zt
