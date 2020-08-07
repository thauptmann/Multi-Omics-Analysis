import torch
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import pandas as pd
import hyperparameter
from tqdm import trange
import encoder
from sklearn.feature_selection import VarianceThreshold
# Strategies for selecting triplets within a minibatch
from siamese_triplet.utils import AllTripletSelector, RandomNegativeTripletSelector
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import random

mini_batch_list = [13, 36, 64]
dim_list = [1023, 512, 256, 128, 64, 32, 16]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
epoch_list = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]
drop_rate_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.01, 0.001, 0.1, 0.0001]
gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


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

    save_results_to = Path(f'./results/{parameter["drug"]}')
    save_results_to.mkdir(parents=True, exist_ok=True)

    expression_train = pd.read_csv(expressions_homogenized_path / parameter['expression_train'],
                                   sep="\t", index_col=0, decimal=',')
    expression_train = pd.DataFrame.transpose(expression_train)

    response_train = pd.read_csv(response_path / parameter['response_train'],
                                 sep="\t", index_col=0, decimal=',')
    response_train.loc[response_train.response == 'R'] = 0
    response_train.loc[response_train.response == 'S'] = 1

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

    response_test = pd.read_csv(response_path / parameter['response_train'],
                                sep="\t", index_col=0, decimal=',')
    response_test.loc[response_test.response == 'R'] = 0
    response_test.loc[response_test.response == 'S'] = 1

    selector = VarianceThreshold(0.05)
    selector.fit(expression_train)
    expression_train = expression_train[expression_train.columns[selector.get_support(indices=True)]]

    cna_test = cna_test.fillna(0)
    cna_test[cna_test != 0.0] = 1
    mutation_test = mutation_test.fillna(0)
    mutation_test[mutation_test != 0.0] = 1
    mutation_train = mutation_train.fillna(0)
    mutation_train[mutation_train != 0.0] = 1
    cna_train = cna_train.fillna(0)
    cna_train[cna_train != 0.0] = 1

    ls = expression_train.columns.intersection(mutation_train.columns)
    ls = ls.intersection(cna_train.columns)
    ls = ls.intersection(expression_test.columns)
    ls = ls.intersection(mutation_test.columns)
    ls = ls.intersection(cna_test.columns)
    ls = pd.unique(ls)
    ls2 = expression_train.index.intersection(mutation_train.index)
    ls2 = ls2.intersection(cna_train.index)
    ls3 = expression_test.index.intersection(mutation_test.index)
    ls3 = ls3.intersection(cna_test.index)

    expression_train = expression_train.loc[ls2, ls]
    mutation_train = mutation_train.loc[ls2, ls]
    cna_train = cna_train.loc[ls2, ls]
    expression_test = expression_test.loc[ls3, ls]
    mutation_test = mutation_test.loc[ls3, ls]
    cna_test = cna_test.loc[ls3, ls]

    y_train = response_train.response.to_numpy(dtype=np.int)

    skf = StratifiedKFold(n_splits=5)

    max_iter = 50
    for _ in trange(max_iter):

        mini_batch = random.choice(mini_batch_list)
        h_dim1 = random.choice(dim_list)
        h_dim2 = random.choice(dim_list)
        h_dim3 = random.choice(dim_list)
        lr_e = random.choice(learning_rate_list)
        lr_m = random.choice(learning_rate_list)
        lr_c = random.choice(learning_rate_list)
        lr_cl = random.choice(learning_rate_list)
        dropout_rate_e = random.choice(drop_rate_list)
        dropout_rate_m = random.choice(drop_rate_list)
        dropout_rate_c = random.choice(drop_rate_list)
        weight_decay = random.choice(weight_decay_list)
        dropout_rate_clf = random.choice(drop_rate_list)
        gamma = random.choice(gamma_list)
        epochs = random.choice(epoch_list)
        margin = random.choice(margin_list)

        costs_validate = []
        aucs_validate = []
        best_auc = 0
        for train_index, test_index in skf.split(expression_train.to_numpy(), y_train):
            x_train_e = expression_train.values[train_index, :]
            x_test_e = expression_train.values[test_index, :]
            x_train_m = mutation_train.values[train_index, :]
            x_test_m = mutation_train.values[test_index, :]
            x_train_c = cna_train.values[train_index, :]
            x_test_c = mutation_train.values[test_index, :]
            y_train = y_train[train_index]
            y_test = y_train[test_index]

            scaler_gdsc = StandardScaler()
            x_train_e = scaler_gdsc.fit_transform(x_train_e)
            x_test_e = scaler_gdsc.transform(x_test_e)

            x_train_m = np.nan_to_num(x_train_m)
            x_train_c = np.nan_to_num(x_train_c)
            x_test_m = np.nan_to_num(x_test_m)
            x_test_c = np.nan_to_num(x_test_c)

            # initialisation
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
                                                       num_workers=8, sampler=sampler, pin_memory=True)

            n_samp_e, ie_dim = x_train_e.shape
            n_samp_m, im_dim = x_train_m.shape
            n_samp_c, ic_dim = x_train_c.shape
            z_in = h_dim1 + h_dim2 + h_dim3

            random_negative_triplet_selector = RandomNegativeTripletSelector(parameter['margin'])
            all_triplet_selector = AllTripletSelector()

            autoencoder_e = encoder.Encoder(ie_dim, h_dim1, dropout_rate_e).to(device)
            autoencoder_m = encoder.Encoder(im_dim, h_dim2, dropout_rate_m).to(device)
            autoencoder_c = encoder.Encoder(ic_dim, h_dim3, dropout_rate_c).to(device)

            optim_e = torch.optim.Adagrad(autoencoder_e.parameters(), lr=lr_e)
            optim_m = torch.optim.Adagrad(autoencoder_m.parameters(), lr=lr_m)
            optim_c = torch.optim.Adagrad(autoencoder_c.parameters(), lr=lr_c)

            trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

            clas = encoder.Classifier(z_in, dropout_rate_clf).to(device)
            optim_clas = torch.optim.Adagrad(clas.parameters(), lr=lr_cl,
                                             weight_decay=weight_decay)

            bce_loss = torch.nn.BCELoss()

            # train
            for epoch in range(epochs):
                cost_train = 0
                auc_train = []
                num_minibatches = int(n_samp_e / mini_batch)
                auc_train, cost_train = train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e,
                                              optim_m, optim_c, optim_clas, all_triplet_selector, trip_criterion,
                                              bce_loss, device, cost_train, num_minibatches, auc_train, gamma)

                # validate
                auc_validate, loss_validate = test(autoencoder_e, autoencoder_m, autoencoder_c, clas, x_test_e, x_test_m,
                                            x_test_c,
                                            y_test, all_triplet_selector, trip_criterion, bce_loss, device, gamma)
            costs_validate.append(loss_validate)
            aucs_validate.append(auc_validate)

        if np.mean(aucs_validate) > best_auc:
            best_mini_batch = mini_batch
            best_h_dim1 = h_dim1
            best_h_dim2 = h_dim2
            best_h_dim3 = h_dim3
            best_lr_e = lr_e
            best_lr_m = lr_m
            best_lr_c = lr_c
            best_lr_cl = lr_cl
            best_dropout_rate_e = dropout_rate_e
            best_dropout_rate_m = dropout_rate_m
            best_dropout_rate_c = dropout_rate_c
            best_weight_decay = weight_decay
            best_dropout_rate_clf = dropout_rate_clf
            best_gamma = gamma
            best_epochs = epochs
            best_margin = margin

    # test

    x_train_e = expression_train.values
    x_test_e = expression_test.values
    x_train_m = mutation_train.values
    x_test_m = mutation_test.values
    x_train_c = cna_train.values
    x_test_c = cna_test.values
    y_train = response_train.response.to_numpy(dtype=np.int)
    y_test = response_test.response.to_numpy(dtype=np.int)

    scaler_gdsc = StandardScaler()
    x_train_e = scaler_gdsc.fit_transform(x_train_e)
    x_test_e = scaler_gdsc.transform(x_test_e)

    x_train_m = np.nan_to_num(x_train_m)
    x_train_c = np.nan_to_num(x_train_c)
    x_test_m = np.nan_to_num(x_test_m)
    x_test_c = np.nan_to_num(x_test_c)
    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                                      torch.FloatTensor(x_train_c),
                                                      torch.FloatTensor(y_train))

    n_samp_e, ie_dim = x_train_e.shape
    n_samp_m, im_dim = x_train_m.shape
    n_samp_c, ic_dim = x_test_m.shape

    Z_in = best_h_dim1 + best_h_dim2 + best_h_dim3

    random_negative_triplet_selector = RandomNegativeTripletSelector(best_margin)
    all_triplet_selector = AllTripletSelector()

    autoencoder_e = encoder.Encoder(ie_dim, best_h_dim1, best_dropout_rate_e).to(device)
    autoencoder_m = encoder.Encoder(im_dim, best_h_dim2, best_dropout_rate_m).to(device)
    autoencoder_c = encoder.Encoder(ic_dim, best_h_dim3, best_dropout_rate_c).to(device)

    optim_e = torch.optim.Adagrad(autoencoder_e.parameters(), lr=best_lr_e)
    optim_m = torch.optim.Adagrad(autoencoder_m.parameters(), lr=best_lr_m)
    optim_c = torch.optim.Adagrad(autoencoder_c.parameters(), lr=best_lr_c)

    trip_criterion = torch.nn.TripletMarginLoss(margin=best_margin, p=2)

    clas = encoder.Classifier(Z_in, best_dropout_rate_clf).to(device)
    optim_clas = torch.optim.Adagrad(clas.parameters(), lr=best_lr_cl,
                                     weight_decay=best_weight_decay)
    bce_loss = torch.nn.BCELoss()


    train_loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=best_mini_batch,
                                                  shuffle=False,
                                                  num_workers=8, sampler=sampler, pin_memory=True)
    for _ in trange(best_epochs):
        train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e,
                                              optim_m, optim_c, optim_clas, all_triplet_selector, trip_criterion,
                                              bce_loss, device, cost_train, num_minibatches, auc_train, best_gamma)

    auc_test = test(autoencoder_e, autoencoder_m, autoencoder_c, clas, x_test_e, x_test_m, x_test_c, y_test,
                        all_triplet_selector, trip_criterion, bce_loss, device, best_gamma)
    print(f'{parameter["drug"]}: AUC = {auc_test}')


def train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e, optim_m, optim_c, optim_clas,
          all_triplet_selector, trip_criterion, bce_loss, device, cost_train, num_minibatches,
          auc_train, gamma):
    for (data_e, data_m, data_c, target) in train_loader:
        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            data_e = data_e.to(device)
            data_m = data_m.to(device)
            data_c = data_c.to(device)
            target = target.to(device)

            for optimizer in (optim_e, optim_m, optim_c, optim_clas):
                optimizer.zero_grad()

            autoencoder_e.train()
            autoencoder_m.train()
            autoencoder_c.train()
            clas.train()

            zex = autoencoder_e(data_e)
            zmx = autoencoder_m(data_m)
            zcx = autoencoder_c(data_c)

            zt = torch.cat((zex, zmx, zcx), 1)
            zt = F.normalize(zt, p=2, dim=0)
            y_prediction = clas(zt)
            triplets = all_triplet_selector.get_triplets(zt, target)
            target = target.view(-1, 1)
            loss = gamma * trip_criterion(zt[triplets[:, 0], :], zt[triplets[:, 1], :],
                                          zt[triplets[:, 2], :]) + bce_loss(y_prediction, target)

            auc = roc_auc_score(target.detach().cpu(), y_prediction.detach().cpu())

            loss.backward()

            for optimizer in (optim_clas, optim_e, optim_m, optim_c):
                optimizer.step()

            cost_train += (loss.item() / num_minibatches)
            auc_train.append(auc)
    return auc_train, cost_train


def test(autoencoder_e, autoencoder_m, autoencoder_c, clas, x_test_e, x_test_m, x_test_c, y_test,
         all_triplet_selector, trip_criterion, bce_loss, device, gamma):
    tx_test_e = torch.FloatTensor(x_test_e).to(device)
    tx_test_m = torch.FloatTensor(x_test_m).to(device)
    tx_test_c = torch.FloatTensor(x_test_c).to(device)
    ty_test_e = torch.FloatTensor(y_test).to(device)
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
        prediction_test = clas(ztt)
        y_true_test = ty_test_e.view(-1, 1)
        triplets_test = all_triplet_selector.get_triplets(ztt, ty_test_e)
        loss_test = gamma * trip_criterion(ztt[triplets_test[:, 0], :], ztt[triplets_test[:, 1], :],
                                           ztt[triplets_test[:, 2], :]) + bce_loss(prediction_test, y_true_test)

        auc_test = roc_auc_score(y_true_test.cpu().detach(), prediction_test.cpu().detach())
        return auc_test, loss_test.item()


# possible = gemcitabine_tcga, cisplatin, docetaxel, erlotinib, cetuximab, gemcitabine_pdx, paclitaxel
if __name__ == "__main__":
    # execute only if run as a script
    for drug_hyperparameters in hyperparameter.drugs_hyperparameters:
        main(drug_hyperparameters)
