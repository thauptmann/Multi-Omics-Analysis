from pathlib import Path
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from tqdm import trange
import torch
from torch import nn

from sklearn.feature_selection import VarianceThreshold
from siamese_triplet.utils import AllTripletSelector, RandomNegativeTripletSelector
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def main(optimal_parameters):
    # reproducibility
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    mini_batch = optimal_parameters['mini_batch']
    h_dim1 = optimal_parameters['h_dim1']
    h_dim2 = optimal_parameters['h_dim2']
    h_dim3 = optimal_parameters['h_dim3']
    lr_e = optimal_parameters['lr_e']
    lr_m = optimal_parameters['lr_m']
    lr_c = optimal_parameters['lr_c']
    lr_cl = optimal_parameters['lr_classifier']
    dropout_rate_e = optimal_parameters['dropout_rate_e']
    dropout_rate_m = optimal_parameters['dropout_rate_m']
    dropout_rate_c = optimal_parameters['dropout_rate_c']
    weight_decay = optimal_parameters['weight_decay']
    dropout_rate_clf = optimal_parameters['dropout_rate_classifier']
    gamma = optimal_parameters['gamma']
    epochs = optimal_parameters['epochs']
    margin = optimal_parameters['margin']

    data_path = Path('data/MOLI')
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'

    expression_train = pd.read_csv(expressions_homogenized_path / optimal_parameters['expression_train'],
                                   sep="\t", index_col=0, decimal=',')
    expression_train = pd.DataFrame.transpose(expression_train)

    response_train = pd.read_csv(response_path / optimal_parameters['response_train'],
                                 sep="\t", index_col=0, decimal=',')

    mutation_train = pd.read_csv(sna_binary_path / optimal_parameters['mutation_train'],
                                 sep="\t", index_col=0, decimal='.')
    mutation_train = pd.DataFrame.transpose(mutation_train)

    cna_train = pd.read_csv(cna_binary_path / optimal_parameters['cna_train'],
                            sep="\t", index_col=0, decimal='.')
    cna_train.drop_duplicates(keep='last')
    cna_train = pd.DataFrame.transpose(cna_train)

    expression_test = pd.read_csv(expressions_homogenized_path / optimal_parameters['expression_test'],
                                  sep="\t", index_col=0, decimal=',')
    expression_test = pd.DataFrame.transpose(expression_test)

    mutation_test = pd.read_csv(sna_binary_path / optimal_parameters['mutation_test'],
                                sep="\t", index_col=0, decimal='.')
    mutation_test = pd.DataFrame.transpose(mutation_test)

    cna_test = pd.read_csv(cna_binary_path / optimal_parameters['cna_test'],
                           sep="\t", index_col=0, decimal='.')
    cna_test.drop_duplicates(keep='last')
    cna_test = pd.DataFrame.transpose(cna_test)
    cna_test = cna_test.loc[:, ~cna_test.columns.duplicated()]

    response_test = pd.read_csv(response_path / optimal_parameters['response_test'],
                                sep="\t", index_col=0, decimal=',')
    response_train.rename(mapper=str, axis='index', inplace=True)
    response_test.rename(mapper=str, axis='index', inplace=True)

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
    response_test = response_test.loc[ls3, :]

    expression_train = expression_train.loc[ls2, ls]
    mutation_train = mutation_train.loc[ls2, ls]
    cna_train = cna_train.loc[ls2, ls]
    response_train = response_train.loc[ls2, :]

    response_test.loc[response_test.response == 'R'] = 0
    response_test.loc[response_test.response == 'S'] = 1
    response_train.loc[response_train.response == 'R'] = 0
    response_train.loc[response_train.response == 'S'] = 1

    x_train_e = expression_train.values
    x_test_e = expression_test.values
    x_train_m = mutation_train.values
    x_test_m = mutation_test.values
    x_train_c = cna_train.values
    x_test_c = cna_test.values
    y_train = response_train.response.values.astype(int)
    y_test = response_test.response.values.astype(int)

    scaler_gdsc = StandardScaler()
    x_train_e = scaler_gdsc.fit_transform(x_train_e)
    x_test_e = scaler_gdsc.transform(x_test_e)

    x_train_m = np.nan_to_num(x_train_m)
    x_train_c = np.nan_to_num(x_train_c)
    x_test_m = np.nan_to_num(x_test_m)
    x_test_c = np.nan_to_num(x_test_c)

    # Train
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                                   torch.FloatTensor(x_train_c),
                                                   torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=mini_batch,
                                               shuffle=False,
                                               num_workers=8, sampler=sampler, pin_memory=True, drop_last=True)

    validate_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e), torch.FloatTensor(x_test_m),
                                                      torch.FloatTensor(x_test_c), torch.FloatTensor(y_test))
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=mini_batch,
                                                  shuffle=False, num_workers=8, pin_memory=True)

    n_samp_e, ie_dim = x_train_e.shape
    n_samp_m, im_dim = x_train_m.shape
    n_samp_c, ic_dim = x_train_c.shape

    Z_in = h_dim1 + h_dim2 + h_dim3

    triplet_selector = RandomNegativeTripletSelector(margin)
    triplet_selector2 = AllTripletSelector()

    class AEE(nn.Module):
        def __init__(self):
            super(AEE, self).__init__()
            self.EnE = torch.nn.Sequential(
                nn.Linear(ie_dim, h_dim1),
                nn.BatchNorm1d(h_dim1),
                nn.ReLU(),
                nn.Dropout(dropout_rate_e))

        def forward(self, x):
            output = self.EnE(x)
            return output

    class AEM(nn.Module):
        def __init__(self):
            super(AEM, self).__init__()
            self.EnM = torch.nn.Sequential(
                nn.Linear(im_dim, h_dim2),
                nn.BatchNorm1d(h_dim2),
                nn.ReLU(),
                nn.Dropout(dropout_rate_m))

        def forward(self, x):
            output = self.EnM(x)
            return output

    class AEC(nn.Module):
        def __init__(self):
            super(AEC, self).__init__()
            self.EnC = torch.nn.Sequential(
                nn.Linear(im_dim, h_dim3),
                nn.BatchNorm1d(h_dim3),
                nn.ReLU(),
                nn.Dropout(dropout_rate_c))

        def forward(self, x):
            output = self.EnC(x)
            return output

    class OnlineTriplet(nn.Module):
        def __init__(self, margin, triplet_selector):
            super(OnlineTriplet, self).__init__()
            self.marg = margin
            self.triplet_selector = triplet_selector

        def forward(self, embeddings, target):
            triplets = self.triplet_selector.get_triplets(embeddings, target)
            return triplets

    class OnlineTestTriplet(nn.Module):
        def __init__(self, margin, triplet_selector):
            super(OnlineTestTriplet, self).__init__()
            self.marg = margin
            self.triplet_selector = triplet_selector

        def forward(self, embeddings, target):
            triplets = self.triplet_selector.get_triplets(embeddings, target)
            return triplets

    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.FC = torch.nn.Sequential(
                nn.Linear(Z_in, 1),
                nn.Dropout(dropout_rate_clf),
                nn.Sigmoid())

        def forward(self, x):
            return self.FC(x)

    torch.cuda.manual_seed_all(42)

    autoencoder_e = AEE().to(device)
    autoencoder_m = AEM().to(device)
    autoencoder_c = AEC().to(device)

    optim_e = torch.optim.Adagrad(autoencoder_e.parameters(), lr=lr_e)
    optim_m = torch.optim.Adagrad(autoencoder_m.parameters(), lr=lr_m)
    optim_c = torch.optim.Adagrad(autoencoder_c.parameters(), lr=lr_c)

    trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    TripSel = OnlineTriplet(margin, triplet_selector)
    TripSel2 = OnlineTestTriplet(margin, triplet_selector2).to(device)

    Clas = Classifier().to(device)
    SolverClass = torch.optim.Adagrad(Clas.parameters(), lr=lr_cl, weight_decay=weight_decay)
    C_loss = torch.nn.BCELoss()

    num_minibatches = int(n_samp_e / optimal_parameters['mini_batch'])
    cost_train = 0
    auc_train = []
    for _ in trange(epochs):
        train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, Clas, optim_e, optim_m, optim_c, SolverClass,
              TripSel2, trip_criterion, C_loss, device, cost_train, num_minibatches,
              auc_train, gamma)
    print(f'{optimal_parameters["drug"]}: AUROC Train = {auc_train[-1]}')

    # test
    auc_test = validate(validate_loader, autoencoder_e, autoencoder_m, autoencoder_c, Clas, device)

    print(f'{optimal_parameters["drug"]}: AUROC Test = {auc_test}')


def train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e, optim_m, optim_c, optim_clas,
          TripSel2, trip_criterion, C_loss, device, cost_train, num_minibatches,
          auc_train, gamma):
    for (data_e, data_m, data_c, target) in train_loader:
        flag = 0

        autoencoder_e.train()
        autoencoder_m.train()
        autoencoder_c.train()
        clas.train()

        data_e = data_e.to(device)
        data_m = data_m.to(device)
        data_c = data_c.to(device)
        target = target.to(device)

        if torch.mean(target) != 0. and torch.mean(target) != 1.:

            zex = autoencoder_e(data_e)
            zmx = autoencoder_m(data_m)
            zcx = autoencoder_c(data_c)

            zt = torch.cat((zex, zmx, zcx), 1)
            zt = F.normalize(zt, p=2, dim=0)
            y_pred = clas(zt)
            triplets = TripSel2(zt, target)
            target = target.view(-1, 1)
            loss = gamma * trip_criterion(zt[triplets[:, 0], :], zt[triplets[:, 1], :],
                                                       zt[triplets[:, 2], :]) + C_loss(y_pred, target)

            auc = roc_auc_score(target.detach().cpu(), y_pred.detach().cpu())

            optim_e.zero_grad()
            optim_m.zero_grad()
            optim_c.zero_grad()
            optim_clas.zero_grad()

            loss.backward()

            optim_clas.step()
            optim_e.step()
            optim_m.step()
            optim_c.step()

            flag = 1

    if flag == 1:
        cost_train += (loss.item() / num_minibatches)
        auc_train.append(auc)
    return auc_train, cost_train


def validate(data_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, device):
    y_true_test = []
    prediction_test = []
    for (data_e, data_m, data_c, target) in data_loader:
        tx_test_e = torch.FloatTensor(data_e).to(device)
        tx_test_m = torch.FloatTensor(data_m).to(device)
        tx_test_c = torch.FloatTensor(data_c).to(device)
        y_true_test.extend(target.view(-1, 1))
        with torch.no_grad():
            autoencoder_e.eval()
            autoencoder_m.eval()
            autoencoder_c.eval()
            clas.eval()

            zet = autoencoder_e(tx_test_e)
            zmt = autoencoder_m(tx_test_m)
            zct = autoencoder_c(tx_test_c)

            ztt = torch.cat((zet, zmt, zct), 1)
            ztt = F.normalize(ztt, p=2, dim=0)
            prediction_test.extend(clas(ztt).cpu().detach())

    auc_test = roc_auc_score(y_true_test, prediction_test)
    return auc_test


if __name__ == "__main__":
    with open("hyperparameter.json") as json_data_file:
        hyperparameter = json.load(json_data_file)
    # execute only if run as a script
    for drug_hyperparameters in hyperparameter['drugs_hyperparameters']:
        main(drug_hyperparameters)
