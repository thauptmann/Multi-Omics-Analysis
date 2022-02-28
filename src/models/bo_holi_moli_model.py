import torch
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
                nn.ReLU(),
                nn.BatchNorm1d(output_size),
                nn.Dropout(dropout_rate))
            self.module_list.extend(dense)

    def forward(self, x):
        output = x
        for layer in self.module_list:
            output = layer(output)
        return output


class MOLIClassifier(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(MOLIClassifier, self).__init__()
        self.module_list = nn.ModuleList(torch.nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, 1),
        )
        )

    def forward(self, x):
        output = x
        for layer in self.module_list:
            output = layer(output)
        return output


class AdaptiveMoli(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates, combination, depths, noisy):
        super(AdaptiveMoli, self).__init__()
        self.combination = combination
        self.noisy = noisy
        if self.noisy:
            self.dropout = nn.Dropout(0.2)
        if combination != 4:
            self.expression_encoder = AdaptiveEncoder(input_sizes[0], output_sizes[0], dropout_rates[0], depths[0])
            self.mutation_encoder = AdaptiveEncoder(input_sizes[1], output_sizes[1], dropout_rates[1], depths[1])
            self.cna_encoder = AdaptiveEncoder(input_sizes[2], output_sizes[2], dropout_rates[2], depths[2])

        if combination == 0:
            self.left_encoder = AdaptiveEncoder(output_sizes[0] + output_sizes[1], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = MOLIClassifier(output_sizes[3] + output_sizes[2], dropout_rates[4])

        elif combination == 1:
            self.left_encoder = AdaptiveEncoder(output_sizes[2] + output_sizes[1], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = MOLIClassifier(output_sizes[3] + output_sizes[0], dropout_rates[4])
        elif combination == 2:
            self.left_encoder = AdaptiveEncoder(output_sizes[2] + output_sizes[0], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = MOLIClassifier(output_sizes[3] + output_sizes[1], dropout_rates[4])
        elif combination == 3:
            self.left_encoder = nn.Identity()
            self.classifier = MOLIClassifier(output_sizes[0] + output_sizes[1] + output_sizes[2], dropout_rates[4])
        elif combination == 4:
            self.expression_encoder = nn.Identity()
            self.mutation_encoder = nn.Identity()
            self.cna_encoder = nn.Identity()
            self.left_encoder = nn.Identity()
            self.classifier = MOLIClassifier(input_sizes[0] + input_sizes[1] + input_sizes[2], dropout_rates[4])

    def forward(self, expression, mutation, cna):
        if self.noisy:
            expression = self.dropout(expression)
            mutation = self.dropout(mutation)
            cna = self.dropout(cna)
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
        return [self.classifier(left_middle_right), left_middle_right]


class AdaptiveMoliWithReconstruction(nn.Module):
    def __init__(self, input_sizes, output_sizes, dropout_rates, combination, depths, noisy):
        super(AdaptiveMoliWithReconstruction, self).__init__()
        self.combination = combination
        self.noisy = noisy
        if self.noisy:
            self.dropout = nn.Dropout(0.2)
        if combination != 4:
            self.expression_encoder = AdaptiveEncoder(input_sizes[0], output_sizes[0], dropout_rates[0], depths[0])
            self.mutation_encoder = AdaptiveEncoder(input_sizes[1], output_sizes[1], dropout_rates[1], depths[1])
            self.cna_encoder = AdaptiveEncoder(input_sizes[2], output_sizes[2], dropout_rates[2], depths[2])

        if combination == 0:
            self.left_encoder = AdaptiveEncoder(output_sizes[0] + output_sizes[1], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = MOLIClassifier(output_sizes[3] + output_sizes[2], dropout_rates[4])

        elif combination == 1:
            self.left_encoder = AdaptiveEncoder(output_sizes[2] + output_sizes[1], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = MOLIClassifier(output_sizes[3] + output_sizes[0], dropout_rates[4])
        elif combination == 2:
            self.left_encoder = AdaptiveEncoder(output_sizes[2] + output_sizes[0], output_sizes[3], dropout_rates[3],
                                                depths[3])
            self.classifier = MOLIClassifier(output_sizes[3] + output_sizes[1], dropout_rates[4])
        elif combination == 3:
            self.left_encoder = nn.Identity()
            self.classifier = MOLIClassifier(output_sizes[0] + output_sizes[1] + output_sizes[2], dropout_rates[4])
        elif combination == 4:
            self.expression_encoder = nn.Identity()
            self.mutation_encoder = nn.Identity()
            self.cna_encoder = nn.Identity()
            self.left_encoder = nn.Identity()
            self.classifier = MOLIClassifier(input_sizes[0] + input_sizes[1] + input_sizes[2], dropout_rates[4])

        self.expression_decoder = nn.Linear(input_sizes[0], output_sizes[0])
        self.mutation_decoder = nn.Linear(input_sizes[1], output_sizes[1])
        self.cna_decoder = nn.Linear(input_sizes[2], output_sizes[2])

    def forward(self, expression, mutation, cna):
        if self.noisy:
            expression = self.dropout(expression)
            mutation = self.dropout(mutation)
            cna = self.dropout(cna)
        if self.combination == 0:
            left_out = self.expression_encoder(expression)
            middle_out = self.mutation_encoder(mutation)
            right_out = self.cna_encoder(cna)
            expression_reconstruction = self.expression_decoder(left_out)
            mutation_reconstruction = self.mutation_encoder(middle_out)
            cna_reconstruction = self.cna_reconstruction(right_out)
        elif self.combination == 1:
            left_out = self.cna_encoder(cna)
            middle_out = self.mutation_encoder(mutation)
            right_out = self.expression_encoder(expression)
            expression_reconstruction = self.expression_decoder(right_out)
            mutation_reconstruction = self.mutation_encoder(middle_out)
            cna_reconstruction = self.cna_reconstruction(left_out)
        elif self.combination == 2:
            left_out = self.cna_encoder(cna)
            middle_out = self.expression_encoder(expression)
            right_out = self.mutation_encoder(mutation)
            expression_reconstruction = self.expression_decoder(middle_out)
            mutation_reconstruction = self.mutation_encoder(right_out)
            cna_reconstruction = self.cna_reconstruction(left_out)
        elif self.combination == 3:
            left_out = self.expression_encoder(expression)
            middle_out = self.mutation_encoder(mutation)
            right_out = self.cna_encoder(cna)
            expression_reconstruction = self.expression_decoder(left_out)
            mutation_reconstruction = self.mutation_encoder(middle_out)
            cna_reconstruction = self.cna_reconstruction(right_out)
        else:
            left_out = self.expression_encoder(expression)
            middle_out = self.mutation_encoder(mutation)
            right_out = self.cna_encoder(cna)
            expression_reconstruction = expression
            mutation_reconstruction = mutation
            cna_reconstruction = cna

        left_middle = torch.cat((left_out, middle_out), 1)
        left_middle_out = self.left_encoder(left_middle)
        left_middle_right = torch.cat((left_middle_out, right_out), 1)
        return [self.classifier(left_middle_right), left_middle_right, expression_reconstruction,
                mutation_reconstruction, cna_reconstruction]
