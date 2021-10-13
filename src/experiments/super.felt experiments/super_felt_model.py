import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(Classifier, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output


class ClassifierEAndMFirst(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(ClassifierEAndMFirst, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output


class ClassifierEAndCFirst(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(ClassifierEAndCFirst, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output


class ClassifierMAndCFirst(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate):
        super(ClassifierMAndCFirst, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        return output


class OnlineTestTriplet(nn.Module):
    def __init__(self, marg, triplet_selector):
        super(OnlineTestTriplet, self).__init__()
        self.marg = marg
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
