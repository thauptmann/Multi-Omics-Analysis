import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_dim, drop_rate):
        super(Classifier, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Dropout(drop_rate)
        )

    def forward(self, encoded_e, encoded_m, encoded_c):
        integrated_test_omics = torch.cat((encoded_e, encoded_m, encoded_c), 1)
        output = self.model(integrated_test_omics)
        return output


class NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, drop_rate):
        super(NonLinearClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(input_dim, input_dim * 0.5),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(input_dim * 0.5, 1)
        )

    def forward(self, encoded_e, encoded_m, encoded_c):
        integrated_test_omics = torch.cat((encoded_e, encoded_m, encoded_c), 1)
        return self.model(integrated_test_omics)


class AdaptedClassifier(nn.Module):
    def __init__(self, input_dim, drop_rate):
        super(AdaptedClassifier, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(input_dim, 1)
        )

    def forward(self, encoded_e, encoded_m, encoded_c):
        integrated_test_omics = torch.cat((encoded_e, encoded_m, encoded_c), 1)
        return self.model(integrated_test_omics)


class OnlineTestTriplet(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(OnlineTestTriplet, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        return triplets


class SupervisedEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate, noisy=False):
        super(SupervisedEncoder, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.noisy = noisy
        if self.noisy:
            self.noise_layer = nn.Dropout(0.02)

    def forward(self, x):
        if self.noisy:
            x = self.noise_layer(x)
        output = self.model(x)
        return output

    def encode(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate, noisy=False):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.noisy = noisy
        if self.noisy:
            self.noise_layer = nn.Dropout(0.02)

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim))

    def forward(self, x):
        if self.noisy:
            x = self.noise_layer(x)
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction

    def encode(self, x):
        return self.encoder(x)


class SupervisedVariationalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate, noisy=False):
        super(SupervisedVariationalEncoder, self).__init__()
        self.mu_layer = nn.Linear(input_dim, output_dim)
        self.log_var_layer = nn.Linear(input_dim, output_dim)
        self.regularisation = nn.Sequential(nn.BatchNorm1d(output_dim),
                                            nn.ReLU(),
                                            nn.Dropout(drop_rate))
        self.noisy = noisy
        if self.noisy:
            self.noise_layer = nn.Dropout(0.02)

    def forward(self, x):
        if self.noisy:
            x = self.noise_layer(x)
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        param = self.reparametrize(mu, log_var)
        return self.regularisation(param), mu, log_var

    def encode(self, x):
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        param = self.reparametrize(mu, log_var)
        return self.regularisation(param)

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma * z


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate, noisy=False):
        super(VariationalAutoEncoder, self).__init__()
        self.mu = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)

        self.regularisation = nn.Sequential(
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim)
        )

    def forward(self, x):
        mu = self.regularisation(self.mu(x))
        log_var = self.regularisation(self.mu(x))
        latent = self.reparametrize(mu, log_var)
        reconstruction = self.decoder(latent)
        return latent, reconstruction, mu, log_var

    def encode(self, x):
        mu = self.regularisation(self.mu(x))
        log_var = self.regularisation(self.mu(x))
        return self.reparametrize(mu, log_var)

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1)))
        z = z.type_as(mu)  # Setting z to be .cuda when using GPU training
        return mu + sigma * z
