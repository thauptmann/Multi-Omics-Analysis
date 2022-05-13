import torch
from torch import nn


class MOLIEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(MOLIEncoder, self).__init__()
        self.encode = torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.encode(x)


class MOLIClassifier(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(MOLIClassifier, self).__init__()
        self.classifier = torch.nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class Moli(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates):
        super(Moli, self).__init__()
        self.expression_encoder = MOLIEncoder(input_sizes[0], output_sizes[0], dropout_rates[0])
        self.mutation_encoder = MOLIEncoder(input_sizes[1], output_sizes[1], dropout_rates[1])
        self.cna_encoder = MOLIEncoder(input_sizes[2], output_sizes[2], dropout_rates[2])
        self.classifier = MOLIClassifier(output_sizes[0] + output_sizes[1] + output_sizes[2], dropout_rates[4])

    def forward(self, expression, mutation, cna):
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return [self.classifier(left_middle_right), left_middle_right]


class MoliWithReconstruction(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates):
        super(MoliWithReconstruction, self).__init__()
        self.expression_encoder = MOLIEncoder(input_sizes[0], output_sizes[0], dropout_rates[0])
        self.mutation_encoder = MOLIEncoder(input_sizes[1], output_sizes[1], dropout_rates[1])
        self.cna_encoder = MOLIEncoder(input_sizes[2], output_sizes[2], dropout_rates[2])

        self.classifier = MOLIClassifier(output_sizes[0] + output_sizes[1] + output_sizes[2], dropout_rates[4])

        self.expression_decoder = nn.Linear(output_sizes[0], input_sizes[0])
        self.mutation_decoder = nn.Linear(output_sizes[1], input_sizes[1])
        self.cna_decoder = nn.Linear(output_sizes[2], input_sizes[2])

    def forward(self, expression, mutation, cna):
        left_out = self.expression_encoder(expression)
        middle_out = self.mutation_encoder(mutation)
        right_out = self.cna_encoder(cna)
        expression_reconstruction = self.expression_decoder(left_out)
        mutation_reconstruction = self.mutation_decoder(middle_out)
        cna_reconstruction = self.cna_decoder(right_out)
        left_middle_right = torch.cat((left_out, middle_out, right_out), 1)
        return [self.classifier(left_middle_right), left_middle_right, expression_reconstruction,
                mutation_reconstruction, cna_reconstruction]
