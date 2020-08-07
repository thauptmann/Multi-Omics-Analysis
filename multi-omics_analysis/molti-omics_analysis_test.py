import torch
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import pandas as pd
import hyperparameter
from tqdm import trange
import encoder

from sklearn.feature_selection import VarianceThreshold
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from siamese_triplet.metrics import AverageNonzeroTripletsMetric
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def main(parameter):
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    data_path = Path('data/MOLI')
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'

    expression_train = pd.read_csv(expressions_homogenized_path / parameter['expression_train'],
                                   sep="\t", index_col=0, decimal=',')
    expression_train = pd.DataFrame.transpose(expression_train)

    response_train = pd.read_csv(response_path / parameter['response_train'],
                                 sep="\t", index_col=0, decimal=',')

    mutation_train = pd.read_csv(sna_binary_path / parameter['mutation_train'],
                                 sep="\t", index_col=0, decimal='.')
    mutation_train = pd.DataFrame.transpose(mutation_train)

    cna_train = pd.read_csv(cna_binary_path / parameter['cna_train'],
                            sep="\t", index_col=0, decimal='.')
    cna_train.drop_duplicates(keep='last')
    cna_train = pd.DataFrame.transpose(cna_train)

    expression_test = pd.read_csv(expressions_homogenized_path / parameter['expression_test'],
                                  sep="\t", index_col=0, decimal=',')
    expression_test = pd.DataFrame.transpose(expression_test)

    mutation_test = pd.read_csv(sna_binary_path / parameter['mutation_test'],
                                sep="\t", index_col=0, decimal='.')
    mutation_test = pd.DataFrame.transpose(mutation_test)

    cna_test = pd.read_csv(cna_binary_path / parameter['cna_test'],
                           sep="\t", index_col=0, decimal='.')
    cna_test.drop_duplicates(keep='last')
    cna_test = pd.DataFrame.transpose(cna_test)
    cna_test = cna_test.loc[:, ~cna_test.columns.duplicated()]

    response_test = pd.read_csv(response_path / parameter['response_test'],
                                sep="\t", index_col=0, decimal=',')
    response_train.rename(mapper=str, axis='index', inplace=True)

    selector = VarianceThreshold(0.05)
    selector.fit(expression_train)
    expression_train = expression_train[expression_train.columns[selector.get_support(indices=True)]]

    cna_test = cna_test.fillna(0)
    cna_test[cna_test != 0.0] = 1
    cna_train = cna_train.fillna(0)
    cna_train[cna_train != 0.0] = 1

    mutation_test = mutation_test.fillna(0)
    mutation_test[mutation_test != 0.0] = 1
    mutation_train = mutation_train.fillna(0)
    mutation_train[mutation_train != 0.0] = 1

    ls = expression_train.columns.intersection(mutation_train.columns)
    ls = ls.intersection(cna_train.columns)
    ls = ls.intersection(expression_test.columns)
    ls = ls.intersection(mutation_test.columns)
    ls = ls.intersection(cna_test.columns)
    ls2 = expression_train.index.intersection(mutation_train.index)
    ls2 = ls2.intersection(cna_train.index)
    ls3 = expression_test.index.intersection(mutation_test.index)
    ls3 = ls3.intersection(cna_test.index)
    ls = pd.unique(ls)

    expression_test = expression_test.loc[ls3, ls]
    mutation_test = mutation_test.loc[ls3, ls]
    cna_test = cna_test.loc[ls3, ls]
    expression_train = expression_train.loc[ls2, ls]
    mutation_train = mutation_train.loc[ls2, ls]
    cna_train = cna_train.loc[ls2, ls]

    response_test.loc[response_test.response == 'R'] = 0
    response_test.loc[response_test.response == 'S'] = 1
    response_train.loc[response_train.response == 'R'] = 0
    response_train.loc[response_train.response == 'S'] = 1
    response_test = response_test.loc[ls3, :]
    response_train = response_train.loc[ls2, :]

    X_trainE = expression_train.values
    X_testE = expression_test.values
    X_trainM = mutation_train.values
    X_testM = mutation_test.values
    X_trainC = cna_train.values
    X_testC = cna_test.values
    y_train = response_train.response.values.astype(int)
    y_test = response_test.response.values.astype(int)

    scalerGDSC = StandardScaler()
    X_trainE = scalerGDSC.fit_transform(X_trainE)
    X_testE = scalerGDSC.transform(X_testE)

    X_trainM = np.nan_to_num(X_trainM)
    X_trainC = np.nan_to_num(X_trainC)
    X_testM = np.nan_to_num(X_testM)
    X_testC = np.nan_to_num(X_testC)

    # Train
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)

    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                  torch.FloatTensor(X_trainC),
                                                  torch.FloatTensor(y_train))

    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=parameter['mini_batch'],
                                              shuffle=False,
                                              num_workers=8, sampler=sampler, pin_memory=True)

    n_sampE, IE_dim = X_trainE.shape
    n_sampM, IM_dim = X_trainM.shape
    n_sampC, IC_dim = X_trainC.shape

    Z_in = parameter['h_dim1'] + parameter['h_dim2'] + parameter['h_dim3']

    random_negative_triplet_selector = RandomNegativeTripletSelector(parameter['margin'])
    all_triplet_selector = AllTripletSelector()

    AutoencoderE = encoder.Encoder(IE_dim, parameter['h_dim1'], parameter['dropout_rateE']).to(device)
    AutoencoderM = encoder.Encoder(IM_dim, parameter['h_dim2'], parameter['dropout_rateM']).to(device)
    AutoencoderC = encoder.Encoder(IC_dim, parameter['h_dim3'], parameter['dropout_rateC']).to(device)

    optimE = torch.optim.Adagrad(AutoencoderE.parameters(), lr=parameter['lrE'])
    optimM = torch.optim.Adagrad(AutoencoderM.parameters(), lr=parameter['lrM'])
    optimC = torch.optim.Adagrad(AutoencoderC.parameters(), lr=parameter['lrC'])

    trip_criterion = torch.nn.TripletMarginLoss(margin=parameter['margin'], p=2)

    Clas = encoder.Classifier(Z_in, parameter['dropout_rateClf']).to(device)
    optim_clas = torch.optim.Adagrad(Clas.parameters(), lr=parameter['lrCL'],
                                     weight_decay=parameter['weight_decay'])
    bce_loss = torch.nn.BCELoss()
    for _ in trange(parameter['epochs']):
        train(trainLoader, AutoencoderE, AutoencoderM, AutoencoderC, Clas, optimE, optimM, optimC, optim_clas,
          all_triplet_selector, trip_criterion, bce_loss, device)


    # test
    auc_test = validate(AutoencoderE, AutoencoderM, AutoencoderC, Clas, X_testE, X_testM, X_testC, y_test, device)

    print(f'{parameter["drug"]}: AUC = {auc_test}')


def train(trainLoader, AutoencoderE, AutoencoderM, AutoencoderC, Clas, optimE, optimM, optimC, optim_clas,
          all_triplet_selector, trip_criterion, bce_with_logits_loss, device):
    # Creates a GradScaler once at the beginning of training.
    for (dataE, dataM, dataC, target) in trainLoader:
        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            dataE = dataE.to(device)
            dataM = dataM.to(device)
            dataC = dataC.to(device)
            target = target.to(device)

            for optim in (optimE, optimM, optimC, optim_clas):
                optim.zero_grad()

            AutoencoderE.train()
            AutoencoderM.train()
            AutoencoderC.train()
            Clas.train()

            ZEX = AutoencoderE(dataE)
            ZMX = AutoencoderM(dataM)
            ZCX = AutoencoderC(dataC)

            ZT = torch.cat((ZEX, ZMX, ZCX), 1)
            ZT = F.normalize(ZT, p=2, dim=0)
            y_pred = Clas(ZT)
            Triplets = all_triplet_selector.get_triplets(ZT, target)
            target = target.view(-1, 1)
            loss = parameter['gamma'] * trip_criterion(ZT[Triplets[:, 0], :], ZT[Triplets[:, 1], :],
                                                       ZT[Triplets[:, 2], :]) + bce_with_logits_loss(y_pred, target)

            # print(roc_auc_score(target.cpu().detach(), y_pred.cpu().detach()))

            loss.backward()
            for optim in (optim_clas, optimE, optimM, optimC):
                optim.step()


def validate(AutoencoderE, AutoencoderM, AutoencoderC, Clas, X_testE, X_testM, X_testC, y_test, device):
    TX_testE = torch.FloatTensor(X_testE).to(device)
    TX_testM = torch.FloatTensor(X_testM).to(device)
    TX_testC = torch.FloatTensor(X_testC).to(device)
    ty_testE = torch.FloatTensor(y_test).to(device)
    with torch.no_grad():
        AutoencoderE.eval()
        AutoencoderM.eval()
        AutoencoderC.eval()
        Clas.eval()
        ZET = AutoencoderE(TX_testE)
        ZMT = AutoencoderM(TX_testM)
        ZCT = AutoencoderC(TX_testC)

        ZTT = torch.cat((ZET, ZMT, ZCT), 1)
        ZTT = F.normalize(ZTT, p=2, dim=0)
        PredT = Clas(ZTT)
        y_truet = ty_testE.view(-1, 1)

        auc_test = roc_auc_score(y_truet.cpu().detach(), PredT.cpu().detach())
        return auc_test


# possible = gemcitabine_tcga, cisplatin, docetaxel, erlotinib, cetuximab, gemcitabine_pdx, paclitaxel
if __name__ == "__main__":
    # execute only if run as a script
    for drug_hyperparameters in hyperparameter.drugs_hyperparameters:
        parameter = drug_hyperparameters
        main(parameter)
