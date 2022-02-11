import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_dim, drop_rate):
        super(Classifier, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, encoded_e, encoded_m, encoded_c):
        integrated_test_omics = torch.cat((encoded_e, encoded_m, encoded_c), 1)
        output = self.model(integrated_test_omics)
        return output


class OnlineTestTriplet(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(OnlineTestTriplet, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        return triplets


class SupervisedEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(SupervisedEncoder, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        output = self.model(x)
        return output


class SupervisedVariationalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(SupervisedVariationalEncoder, self).__init__()
        self.mu_layer = nn.Linear(input_dim, output_dim)
        self.log_var_layer = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        latent = self.reparametrize(mu, log_var)
        return self.dropout(self.relu(self.batch_norm(latent)))

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma * z