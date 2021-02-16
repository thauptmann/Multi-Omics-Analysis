import torch
import torch.nn.functional as F
from torch import nn


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
        for i in range(depth):
            dense = torch.nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate))
            self.module_list.extend(dense)
        self.module_list = torch.nn.Sequential(nn.Linear(input_size, 1))

    def forward(self, x):
        output = x
        for layer in self.module_list:
            output = layer(output)
        return output


class Moli(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates, combination, depths):
        super(Moli, self).__init__()
        self.combination = combination
        expression_encoder = AdaptiveEncoder(input_sizes[0], output_sizes[0], dropout_rates[0], depths[0])
        mutation_encoder = AdaptiveEncoder(input_sizes[1], output_sizes[1], dropout_rates[1], depths[1])
        cna_encoder = AdaptiveEncoder(input_sizes[2], output_sizes[2], dropout_rates[2], depths[2])

        if combination == 0:
            self.left = expression_encoder
            self.middle = mutation_encoder
            self.right = cna_encoder
            self.left_encoder = AdaptiveEncoder(output_sizes[0] + output_sizes[1], output_sizes[3], dropout_rates[4],
                                                depths[3])
            self.classifier = Classifier(output_sizes[3] + output_sizes[2], output_sizes[4],
                                         dropout_rates[3], depths[4])

        elif combination == 1:
            self.left = cna_encoder
            self.middle = mutation_encoder
            self.right = expression_encoder
            self.left_encoder = AdaptiveEncoder(output_sizes[2] + output_sizes[1], output_sizes[3], dropout_rates[4],
                                                depths[3])
            self.classifier = Classifier(output_sizes[3] + output_sizes[0], output_sizes[4],
                                         dropout_rates[3], depths[4])
        elif combination == 2:
            self.left = cna_encoder
            self.middle = expression_encoder
            self.right = mutation_encoder
            self.left_encoder = AdaptiveEncoder(output_sizes[2] + output_sizes[0], output_sizes[3], dropout_rates[4],
                                                depths[3])
            self.classifier = Classifier(output_sizes[3] + output_sizes[1], output_sizes[4], dropout_rates[3],
                                         depths[4])
        elif combination == 3:
            self.left = expression_encoder
            self.middle = mutation_encoder
            self.right = cna_encoder
            self.left_encoder = nn.Identity(output_sizes[0] + output_sizes[1] + output_sizes[2])
            self.classifier = Classifier(output_sizes[0] + output_sizes[1] + output_sizes[2], output_sizes[4],
                                         dropout_rates[3], depths[4])

    def forward(self, expression, mutation, cna):
        if self.combination == 0:
            left_input = expression
            middle_input = mutation
            right_input = cna
        elif self.combination == 1:
            left_input = cna
            middle_input = mutation
            right_input = expression
        elif self.combination == 2:
            left_input = cna
            middle_input = expression
            right_input = mutation
        else:
            left_input = expression
            middle_input = mutation
            right_input = cna
        left_out = self.left(left_input)
        middle_out = self.middle(middle_input)
        right_out = self.right(right_input)
        left_middle = torch.cat((left_out, middle_out), 1)
        left_middle_out = self.left_encoder(left_middle)
        left_middle_right = torch.cat((left_middle_out, right_out), 1)
        return self.classifier(left_middle_right), left_middle_right
