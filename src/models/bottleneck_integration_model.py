import torch
from torch import nn
from numpy import mean


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


class Mobi(nn.Module):
    def __init__(self, input_sizes, encoding_sizes, bottleneck_size, dropout):
        super(Mobi, self).__init__()

        self.expression_encoder = Encoder(input_sizes[0], encoding_sizes[0], dropout[0])
        self.mutation_encoder = Encoder(input_sizes[1], encoding_sizes[1], dropout[1])
        self.cna_encoder = Encoder(input_sizes[2], encoding_sizes[2], dropout[2])

        self.e_classifier = torch.nn.Sequential(nn.Linear(encoding_sizes[0] + bottleneck_size, 1))
        self.m_classifier = torch.nn.Sequential(nn.Linear(encoding_sizes[1] + bottleneck_size, 1))
        self.c_classifier = torch.nn.Sequential(nn.Linear(encoding_sizes[2] + bottleneck_size, 1))
        self.bottleneck_classifier = torch.nn.Sequential(nn.Linear(bottleneck_size, 1))

        self.bottleneck_layer = nn.Linear(sum(encoding_sizes), bottleneck_size)

    def forward(self, expression, mutation, cna):
        encoded_e = self.expression_encoder(expression)
        encoded_m = self.mutation_encoder(mutation)
        encoded_c = self.cna_encoder(cna)
        bottleneck_features = self.bottleneck_layer(torch.concat((encoded_e, encoded_m, encoded_c), dim=1))

        classified_e = self.e_classifier(torch.concat((encoded_e, bottleneck_features), dim=1))
        classified_m = self.m_classifier(torch.concat((encoded_m, bottleneck_features), dim=1))
        classified_c = self.c_classifier(torch.concat((encoded_c, bottleneck_features), dim=1))

        stacked_logits = torch.mean(torch.concat((classified_e, classified_m, classified_c), dim=1), dim=1)
        return [stacked_logits, torch.concat((encoded_e, encoded_m, encoded_c), dim=1)]


class MobiWithReconstruction(nn.Module):
    def __init__(self, input_sizes, encoding_sizes, bottleneck_size, dropout):
        super(MobiWithReconstruction, self).__init__()

        self.expression_encoder = Encoder(input_sizes[0], encoding_sizes[0], dropout[0])
        self.mutation_encoder = Encoder(input_sizes[1], encoding_sizes[1], dropout[1])
        self.cna_encoder = Encoder(input_sizes[2], encoding_sizes[2], dropout[2])

        self.e_classifier = torch.nn.Sequential(nn.Linear(encoding_sizes[0] + bottleneck_size, 1))
        self.m_classifier = torch.nn.Sequential(nn.Linear(encoding_sizes[1] + bottleneck_size, 1))
        self.c_classifier = torch.nn.Sequential(nn.Linear(encoding_sizes[2] + bottleneck_size, 1))
        self.bottleneck_classifier = torch.nn.Sequential(nn.Linear(bottleneck_size, 1))

        self.bottleneck_layer = nn.Linear(sum(encoding_sizes), bottleneck_size)

        self.expression_decoder = nn.Linear(encoding_sizes[0], input_sizes[0])
        self.mutation_decoder = nn.Linear(encoding_sizes[1], input_sizes[1])
        self.cna_decoder = nn.Linear(encoding_sizes[2], input_sizes[2])

    def forward(self, expression, mutation, cna):
        encoded_e = self.expression_encoder(expression)
        encoded_m = self.mutation_encoder(mutation)
        encoded_c = self.cna_encoder(cna)
        bottleneck_features = self.bottleneck_layer(torch.concat((encoded_e, encoded_m, encoded_c), dim=1))

        classified_e = self.e_classifier(torch.concat((encoded_e, bottleneck_features), dim=1))
        classified_m = self.m_classifier(torch.concat((encoded_m, bottleneck_features), dim=1))
        classified_c = self.c_classifier(torch.concat((encoded_c, bottleneck_features), dim=1))

        expression_reconstruction = self.expression_decoder(encoded_e)
        mutation_reconstruction = self.mutation_decoder(encoded_m)
        cna_reconstruction = self.cna_decoder(encoded_c)

        stacked_logits = torch.mean(torch.concat((classified_e, classified_m, classified_c), dim=1), dim=1)
        return [stacked_logits, torch.concat((encoded_e, encoded_m, encoded_c), dim=1),
                expression_reconstruction, mutation_reconstruction, cna_reconstruction]
