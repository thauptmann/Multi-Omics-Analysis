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


class StackingSigmoidModel(nn.Module):
    def __init__(self, input_sizes, encoding_sizes, dropout, less_stacking=False):
        super(StackingSigmoidModel, self).__init__()
        use_sigmoid = True
        self.less_stacking = less_stacking
        self.expression_encoder = Encoder(input_sizes[0], encoding_sizes[0], dropout[0])
        self.mutation_encoder = Encoder(input_sizes[1], encoding_sizes[1], dropout[1])
        self.cna_encoder = Encoder(input_sizes[2], encoding_sizes[2], dropout[2])

        e_encoding_size = encoding_sizes[0]
        m_encoding_size = encoding_sizes[1]
        c_encoding_size = encoding_sizes[2]
        em_encoding_size = e_encoding_size + m_encoding_size
        mc_encoding_size = m_encoding_size + c_encoding_size
        ec_encoding_size = e_encoding_size + c_encoding_size
        emc_encoding_size = e_encoding_size + m_encoding_size + c_encoding_size
        if not less_stacking:
            stacking_sizes = 7
        else:
            stacking_sizes = 4
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
        encoded_e = self.expression_encoder(expression)
        encoded_m = self.mutation_encoder(mutation)
        encoded_c = self.cna_encoder(cna)

        classified_e = self.e_classify(encoded_e)
        classified_m = self.m_classify(encoded_m)
        classified_c = self.c_classify(encoded_c)
        if not self.less_stacking:
            classified_em = self.em_classify(torch.concat((encoded_e, encoded_m), dim=1))
            classified_mc = self.mc_classify(torch.concat((encoded_m, encoded_c), dim=1))
            classified_ec = self.ec_classify(torch.concat((encoded_e, encoded_c), dim=1))
            classified_emc = self.emc_classify(torch.concat((encoded_e, encoded_m, encoded_c), dim=1))

            return [self.classify_all(torch.concat((classified_e, classified_m, classified_c, classified_em,
                                                    classified_mc,
                                                    classified_ec, classified_emc), dim=1)),
                    torch.concat((encoded_e, encoded_m, encoded_c), dim=1)]
        else:
            classified_emc = self.emc_classify(torch.concat((encoded_e, encoded_m, encoded_c), dim=1))

            return [
                self.classify_all(torch.concat((classified_e, classified_m, classified_c, classified_emc), dim=1)),
                torch.concat((encoded_e, encoded_m, encoded_c), dim=1)]


class StackingSigmoidModelWithReconstruction(nn.Module):
    def __init__(self, input_sizes, encoding_sizes, dropout, less_stacking=False):
        super(StackingSigmoidModelWithReconstruction, self).__init__()
        self.expression_encoder = Encoder(input_sizes[0], encoding_sizes[0], dropout[0])
        self.mutation_encoder = Encoder(input_sizes[1], encoding_sizes[1], dropout[1])
        self.cna_encoder = Encoder(input_sizes[2], encoding_sizes[2], dropout[2])
        self.less_stacking = less_stacking
        e_encoding_size = encoding_sizes[0]
        m_encoding_size = encoding_sizes[1]
        c_encoding_size = encoding_sizes[2]
        em_encoding_size = e_encoding_size + m_encoding_size
        mc_encoding_size = m_encoding_size + c_encoding_size
        ec_encoding_size = e_encoding_size + c_encoding_size
        emc_encoding_size = e_encoding_size + m_encoding_size + c_encoding_size
        if not less_stacking:
            stacking_sizes = 7
        else:
            stacking_sizes = 4
        activation_function = nn.Sigmoid()

        self.e_classify = nn.Sequential(nn.Linear(encoding_sizes[0], 1), activation_function)
        self.m_classify = nn.Sequential(nn.Linear(encoding_sizes[1], 1), activation_function)
        self.c_classify = nn.Sequential(nn.Linear(encoding_sizes[2], 1), activation_function)
        self.em_classify = nn.Sequential(nn.Linear(em_encoding_size, 1), activation_function)
        self.mc_classify = nn.Sequential(nn.Linear(mc_encoding_size, 1), activation_function)
        self.ec_classify = nn.Sequential(nn.Linear(ec_encoding_size, 1), activation_function)
        self.emc_classify = nn.Sequential(nn.Linear(emc_encoding_size, 1), activation_function)
        self.classify_all = nn.Linear(stacking_sizes, 1)

        self.expression_decoder = nn.Linear(encoding_sizes[0], input_sizes[0])
        self.mutation_decoder = nn.Linear(encoding_sizes[1], input_sizes[1])
        self.cna_decoder = nn.Linear(encoding_sizes[2], input_sizes[2])

    def forward(self, expression, mutation, cna):
        encoded_e = self.expression_encoder(expression)
        encoded_m = self.mutation_encoder(mutation)
        encoded_c = self.cna_encoder(cna)

        classified_e = self.e_classify(encoded_e)
        classified_m = self.m_classify(encoded_m)
        classified_c = self.c_classify(encoded_c)

        expression_reconstruction = self.expression_decoder(encoded_e)
        mutation_reconstruction = self.mutation_decoder(encoded_m)
        cna_reconstruction = self.cna_decoder(encoded_c)

        if not self.less_stacking:
            classified_em = self.em_classify(torch.concat((encoded_e, encoded_m), dim=1))
            classified_mc = self.mc_classify(torch.concat((encoded_m, encoded_c), dim=1))
            classified_ec = self.ec_classify(torch.concat((encoded_e, encoded_c), dim=1))
            classified_emc = self.emc_classify(torch.concat((encoded_e, encoded_m, encoded_c), dim=1))

            return [self.classify_all(torch.concat((classified_e, classified_m, classified_c, classified_em,
                                                    classified_mc,
                                                    classified_ec, classified_emc), dim=1)),
                    torch.concat((encoded_e, encoded_m, encoded_c), dim=1), expression_reconstruction,
                    mutation_reconstruction, cna_reconstruction]
        else:
            classified_emc = self.emc_classify(torch.concat((encoded_e, encoded_m, encoded_c), dim=1))

            return [
                self.classify_all(torch.concat((classified_e, classified_m, classified_c, classified_emc), dim=1)),
                torch.concat((encoded_e, encoded_m, encoded_c), dim=1), expression_reconstruction,
                mutation_reconstruction, cna_reconstruction]
