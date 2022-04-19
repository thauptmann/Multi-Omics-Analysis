import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.encoder(x)


class StackingFeaturesModel(nn.Module):
    def __init__(self, input_sizes, encoding_sizes, dropout):
        super(StackingFeaturesModel, self).__init__()

        self.e_encoder = Encoder(input_sizes[0], encoding_sizes[0], dropout[0])
        self.m_encoder = Encoder(input_sizes[1], encoding_sizes[1], dropout[1])
        self.c_encoder = Encoder(input_sizes[2], encoding_sizes[2], dropout[2])

        e_encoding_size = encoding_sizes[0]
        m_encoding_size = encoding_sizes[1]
        c_encoding_size = encoding_sizes[2]
        em_encoding_size = e_encoding_size + m_encoding_size
        mc_encoding_size = m_encoding_size + c_encoding_size
        ec_encoding_size = e_encoding_size + c_encoding_size
        emc_encoding_size = e_encoding_size + m_encoding_size + c_encoding_size
        stacking_sizes = 0

        stacking_sizes += e_encoding_size
        stacking_sizes += m_encoding_size
        stacking_sizes += c_encoding_size
        self.encode_em = nn.Sequential(
            nn.Linear(em_encoding_size, int(em_encoding_size / 2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(em_encoding_size / 2)),
            nn.Dropout(dropout[3]))
        stacking_sizes += int(em_encoding_size / 2)
        self.encode_mc = nn.Sequential(
            nn.Linear(mc_encoding_size, int(mc_encoding_size / 2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(mc_encoding_size / 2)),
            nn.Dropout(dropout[3]))
        stacking_sizes += int(mc_encoding_size / 2)
        self.encode_ec = nn.Sequential(
            nn.Linear(ec_encoding_size, int(ec_encoding_size / 2)), nn.ReLU(),
            nn.BatchNorm1d(int(ec_encoding_size / 2)),
            nn.Dropout(dropout[3]))
        stacking_sizes += int(ec_encoding_size / 2)
        self.encode_emc = nn.Sequential(
            nn.Linear(emc_encoding_size, int(emc_encoding_size / 2)), nn.ReLU(),
            nn.BatchNorm1d(int(emc_encoding_size / 2)),
            nn.Dropout(dropout[3]))
        stacking_sizes += int(emc_encoding_size / 2)
        self.classify_all = nn.Linear(stacking_sizes, 1)

    def forward(self, expression, mutation, cna):
        encoded_e = self.e_encoder(expression)
        encoded_m = self.m_encoder(mutation)
        encoded_c = self.c_encoder(cna)
        encoded_em = self.encode_em(torch.concat((encoded_e, encoded_m), dim=1))
        encoded_mc = self.encode_mc(torch.concat((encoded_m, encoded_c), dim=1))
        encoded_ec = self.encode_ec(torch.concat((encoded_e, encoded_c), dim=1))
        encoded_emc = self.encode_emc(torch.concat((encoded_e, encoded_m, encoded_c), dim=1))

        return [self.classify_all(torch.concat((encoded_e, encoded_m, encoded_c, encoded_em, encoded_mc,
                                                encoded_ec, encoded_emc), dim=1)),
                torch.concat((encoded_e, encoded_m, encoded_c), dim=1)]


class StackingSigmoidModel(nn.Module):
    def __init__(self, input_sizes, encoding_sizes, dropout):
        super(StackingSigmoidModel, self).__init__()
        use_sigmoid = True
        self.e_encoder = Encoder(input_sizes[0], encoding_sizes[0], dropout[0])
        self.m_encoder = Encoder(input_sizes[1], encoding_sizes[1], dropout[1])
        self.c_encoder = Encoder(input_sizes[2], encoding_sizes[2], dropout[2])

        e_encoding_size = encoding_sizes[0]
        m_encoding_size = encoding_sizes[1]
        c_encoding_size = encoding_sizes[2]
        em_encoding_size = e_encoding_size + m_encoding_size
        mc_encoding_size = m_encoding_size + c_encoding_size
        ec_encoding_size = e_encoding_size + c_encoding_size
        emc_encoding_size = e_encoding_size + m_encoding_size + c_encoding_size
        stacking_sizes = 7
        if use_sigmoid:
            activation_function = nn.Sigmoid()
        else:
            activation_function = nn.Identity()

        self.e_classify = nn.Sequential(nn.Linear(encoding_sizes[0], 1), activation_function)
        self.m_classify = nn.Sequential(nn.Linear(encoding_sizes[1], 1), activation_function)
        self.c_classify = nn.Sequential(nn.Linear(encoding_sizes[2], 1), activation_function)
        self.em_classify = nn.Sequential(nn.Linear(em_encoding_size, 1), activation_function)
        self.mc_classify = nn.Sequential(nn.Linear(mc_encoding_size, 1), activation_function)
        self.ec_classify = nn.Sequential(nn.Linear(ec_encoding_size, 1), activation_function)
        self.emc_classify = nn.Sequential(nn.Linear(emc_encoding_size, 1), activation_function)
        self.classify_all = nn.Linear(stacking_sizes, 1)

    def forward(self, expression, mutation, cna):
        encoded_e = self.e_encoder(expression)
        encoded_m = self.m_encoder(mutation)
        encoded_c = self.c_encoder(cna)

        classified_e = self.e_classify(encoded_e)
        classified_m = self.m_classify(encoded_m)
        classified_c = self.c_classify(encoded_c)
        classified_em = self.em_classify(torch.concat((encoded_e, encoded_m), dim=1))
        classified_mc = self.mc_classify(torch.concat((encoded_m, encoded_c), dim=1))
        classified_ec = self.ec_classify(torch.concat((encoded_e, encoded_c), dim=1))
        classified_emc = self.emc_classify(torch.concat((encoded_e, encoded_m, encoded_c), dim=1))

        return [self.classify_all(torch.concat((classified_e, classified_m, classified_c, classified_em, classified_mc,
                                                classified_ec, classified_emc), dim=1)),
                torch.concat((encoded_e, encoded_m, encoded_c), dim=1)]
