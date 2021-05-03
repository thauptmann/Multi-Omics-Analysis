import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveEncoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate, depth):
        super(AdaptiveEncoder, self).__init__()
        self.module_list = nn.ModuleList()
        self.module_list.extend(torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate)))
        for i in range(depth-1):
            dense = torch.nn.Sequential(
                nn.Linear(output_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate))
            self.module_list.extend(dense)

    def forward(self, x):
        output = x
        for layer in self.module_list:
            output = layer(output)
        return output


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate, depth):
        super(Classifier, self).__init__()
        self.module_list = nn.ModuleList()
        self.module_list.extend(torch.nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate)))
        for i in range(depth-1):
            dense = torch.nn.Sequential(
                nn.Linear(output_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate))
            self.module_list.extend(dense)
        self.module_list.extend(torch.nn.Sequential(nn.Linear(output_size, 1)))

    def forward(self, x):
        output = x
        for layer in self.module_list:
            output = layer(output)
        return output


class AdaptiveMoli(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates, combination, depths):
        super(AdaptiveMoli, self).__init__()
        self.combination = combination
        if combination != 4:
            self.expression_encoder = AdaptiveEncoder(input_sizes[0], output_sizes[0], dropout_rates[0], depths[0])
            self.mutation_encoder = AdaptiveEncoder(input_sizes[1], output_sizes[1], dropout_rates[1], depths[1])
            self.cna_encoder = AdaptiveEncoder(input_sizes[2], output_sizes[2], dropout_rates[2], depths[2])

        if combination == 0:
            self.left_encoder = AdaptiveEncoder(output_sizes[0] + output_sizes[1], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = Classifier(output_sizes[3] + output_sizes[2], output_sizes[4],
                                         dropout_rates[4], depths[4])

        elif combination == 1:
            self.left_encoder = AdaptiveEncoder(output_sizes[2] + output_sizes[1], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = Classifier(output_sizes[3] + output_sizes[0], output_sizes[4],
                                         dropout_rates[4], depths[4])
        elif combination == 2:
            self.left_encoder = AdaptiveEncoder(output_sizes[2] + output_sizes[0], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = Classifier(output_sizes[3] + output_sizes[1], output_sizes[4], dropout_rates[4],
                                         depths[4])
        elif combination == 3:
            self.left_encoder = nn.Identity()
            self.classifier = Classifier(output_sizes[0] + output_sizes[1] + output_sizes[2], output_sizes[4],
                                         dropout_rates[4], depths[4])
        elif combination == 4:
            self.expression_encoder = nn.Identity()
            self.mutation_encoder = nn.Identity()
            self.cna_encoder = nn.Identity()
            self.left_encoder = nn.Identity()
            self.classifier = Classifier(input_sizes[0] + input_sizes[1] + input_sizes[2], output_sizes[4],
                                         dropout_rates[4], depths[4])

    def forward(self, expression, mutation, cna):
        if self.combination == 0:
            left_out = self.expression_encoder(expression)
            middle_out = self.mutation_encoder(mutation)
            right_out = self.cna_encoder(cna)
        elif self.combination == 1:
            left_out = self.cna_encoder(cna)
            middle_out = self.mutation_encoder(mutation)
            right_out = self.expression_encoder(expression)
        elif self.combination == 2:
            left_out = self.cna_encoder(cna)
            middle_out = self.expression_encoder(expression)
            right_out = self.mutation_encoder(mutation)
        elif self.combination == 3:
            left_out = self.expression_encoder(expression)
            middle_out = self.mutation_encoder(mutation)
            right_out = self.cna_encoder(cna)
        else:
            left_out = self.expression_encoder(expression)
            middle_out = self.mutation_encoder(mutation)
            right_out = self.cna_encoder(cna)
        left_middle = torch.cat((left_out, middle_out), 1)
        left_middle_out = self.left_encoder(left_middle)
        left_middle_right = torch.cat((left_middle_out, right_out), 1)
        return self.classifier(left_middle_right), left_middle_right
