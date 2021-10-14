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
    def __init__(self, input_dims, layer_dim, drop_rate):
        super(ClassifierEAndMFirst, self).__init__()
        self.dense = torch.nn.Sequential(
            nn.Linear(input_dims[0] + input_dims[1], layer_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.linear = torch.nn.Sequential(
            nn.Linear(sum(input_dims), 1),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, encoded_e, encoded_m, encoded_c):
        integrated_e_m = torch.cat((encoded_e, encoded_m), 1)
        encoded_e_m = self.dense(integrated_e_m)
        integrated_e_m_c = torch.cat((encoded_e_m, encoded_c), 1)
        return self.linear(integrated_e_m_c)


class ClassifierEAndCFirst(nn.Module):
    def __init__(self, input_dims, layer_dim, drop_rate):
        super(ClassifierEAndCFirst, self).__init__()
        self.dense = torch.nn.Sequential(
            nn.Linear(input_dims[0] + input_dims[2], layer_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.linear = torch.nn.Sequential(
            nn.Linear(sum(input_dims), 1),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, encoded_e, encoded_m, encoded_c):
        integrated_e_c = torch.cat((encoded_e, encoded_c), 1)
        encoded_e_c = self.dense(integrated_e_c)
        integrated_e_m_c = torch.cat((encoded_e_c, encoded_m), 1)
        return self.linear(integrated_e_m_c)


class ClassifierMAndCFirst(nn.Module):
    def __init__(self, input_dims, layer_dim, drop_rate):
        super(ClassifierMAndCFirst, self).__init__()
        self.dense = torch.nn.Sequential(
            nn.Linear(input_dims[1] + input_dims[2], layer_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.linear = torch.nn.Sequential(
            nn.Linear(sum(input_dims), 1),
            nn.Dropout(drop_rate),
            nn.Sigmoid()
        )

    def forward(self, encoded_e, encoded_m, encoded_c):
        integrated_m_c = torch.cat((encoded_m, encoded_c), 1)
        encoded_m_c = self.dense(integrated_m_c)
        integrated_e_m_c = torch.cat((encoded_m_c, encoded_m), 1)
        return self.linear(integrated_e_m_c)


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
