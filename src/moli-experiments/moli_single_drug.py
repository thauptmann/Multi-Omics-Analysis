import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange

from utils import network_training_util
from utils.network_training_util import read_and_transpose_csv
from models.moli_model import Moli
# Strategies for selecting triplets within a minibatch
from siamese_triplet.utils import AllTripletSelector


def cv_and_train(parameter, drug, run_test, max_iter):
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    cross_validation = parameter['cross_validation']
    mini_batch_list = cross_validation['mini_batch_list']
    dim_list = cross_validation['dim_list']
    margin_list = cross_validation['margin_list']
    learning_rate_list = cross_validation['learning_rate_list']
    epoch_list = cross_validation['epoch_list']
    drop_rate_list = cross_validation['drop_rate_list']
    weight_decay_list = cross_validation['weight_decay_list']
    gamma_list = cross_validation['gamma_list']

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_path = Path('../../data')
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'

    save_results_to = Path(f'../results', drug)
    save_results_to.mkdir(parents=True, exist_ok=True)

    expression_train = read_and_transpose_csv(expressions_homogenized_path / parameter['expression_train'])
    response_train = pd.read_csv(response_path / parameter['response_train'],sep="\t", index_col=0, decimal=',')
    mutation_train = read_and_transpose_csv(sna_binary_path / parameter['mutation_train'])
    cna_train = read_and_transpose_csv(cna_binary_path / parameter['cna_train'], drop=True)

    expression_test = read_and_transpose_csv(expressions_homogenized_path / parameter['expression_test'])
    mutation_test = read_and_transpose_csv(sna_binary_path / parameter['mutation_test'])
    cna_test = read_and_transpose_csv(cna_binary_path / parameter['cna_test'], drop=True)
    response_test = pd.read_csv(response_path / parameter['response_test'], sep="\t", index_col=0, decimal=',')

    response_train.loc[response_train.response == 'R'] = 0
    response_train.loc[response_train.response == 'S'] = 1
    response_test.loc[response_test.response == 'R'] = 0
    response_test.loc[response_test.response == 'S'] = 1
    response_test.rename(mapper=str, axis='index', inplace=True)
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
    ls = pd.unique(ls)
    ls2 = expression_train.index.intersection(mutation_train.index)
    ls2 = ls2.intersection(cna_train.index)
    ls3 = expression_test.index.intersection(mutation_test.index)
    ls3 = ls3.intersection(cna_test.index)

    expression_test = expression_test.loc[ls3, ls]
    mutation_test = mutation_test.loc[ls3, ls]
    cna_test = cna_test.loc[ls3, ls]
    response_test = response_test.loc[ls3, :]
    expression_train = expression_train.loc[ls2, ls]
    mutation_train = mutation_train.loc[ls2, ls]
    cna_train = cna_train.loc[ls2, ls]
    response_train = response_train.loc[ls2, :]

    y = response_train.response.to_numpy(dtype=np.int)

    stratified_k_fold = StratifiedKFold(n_splits=5)

    best_auc = 0
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
        dropout_rate_clf = random.choice(drop_rate_list)
        weight_decay = random.choice(weight_decay_list)
        gamma = random.choice(gamma_list)
        epochs = random.choice(epoch_list)
        margin = random.choice(margin_list)

        aucs_validate = []
        for train_index, test_index in stratified_k_fold.split(expression_train.to_numpy(), y):
            x_train_e = expression_train.values[train_index]
            x_test_e = expression_train.values[test_index]
            x_train_m = mutation_train.values[train_index]
            x_test_m = mutation_train.values[test_index]
            x_train_c = cna_train.values[train_index]
            x_test_c = cna_train.values[test_index]
            y_train = y[train_index]
            y_test = y[test_index]

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

            validate_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e), torch.FloatTensor(x_test_m),
                                                              torch.FloatTensor(x_test_c), torch.FloatTensor(y_test))
            validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=mini_batch,
                                                          shuffle=False, num_workers=8, pin_memory=True)

            n_sample_e, ie_dim = x_train_e.shape
            _, im_dim = x_train_m.shape
            _, ic_dim = x_train_c.shape

            all_triplet_selector = AllTripletSelector()

            moli_model = Moli([ie_dim, im_dim, ic_dim],
                              [h_dim1, h_dim2, h_dim3],
                              [dropout_rate_e, dropout_rate_m, dropout_rate_c, dropout_rate_clf]).to(device)

            moli_optimiser = torch.optim.Adagrad([
                {'params': moli_model.expression_encoder.parameters(), 'lr': lr_e},
                {'params': moli_model.mutation_encoder.parameters(), 'lr': lr_m},
                {'params': moli_model.cna_encoder.parameters(), 'lr': lr_c},
                {'params': moli_model.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
            ])

            trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
            bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

            # validate
            auc_validate = 0
            for _ in trange(epochs):
                network_training_util.train(train_loader, moli_model, moli_optimiser,
                                            all_triplet_selector, trip_criterion, bce_with_logits_loss, device, gamma)

                # validate
                auc_validate = network_training_util.validate(validate_loader, moli_model, device)
            aucs_validate.append(auc_validate)

        auc_cv = np.mean(aucs_validate)
        if auc_cv > best_auc:
            best_auc = auc_cv
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
    print(f'{drug}: best AUROC validation: {best_auc}')

    # test
    if run_test:
        x_train_e = expression_train.values
        x_train_m = mutation_train.values
        x_train_c = cna_train.values
        y_train = response_train.response.to_numpy(dtype=np.int)

        x_test_e = expression_test.values
        x_test_m = mutation_test.values
        x_test_c = cna_test.values
        y_test = response_test.response.to_numpy(dtype=np.int)

        scaler = StandardScaler()
        x_train_e = scaler.fit_transform(x_train_e)
        x_test_e = scaler.transform(x_test_e)

        x_train_m = np.nan_to_num(x_train_m)
        x_train_c = np.nan_to_num(x_train_c)
        x_test_m = np.nan_to_num(x_test_m)
        x_test_c = np.nan_to_num(x_test_c)
        _, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_c.shape
        z_in = best_h_dim1 + best_h_dim2 + best_h_dim3

        # random_negative_triplet_selector = RandomNegativeTripletSelector(best_margin)
        all_triplet_selector = AllTripletSelector()
        moli_model = Moli([ie_dim, im_dim, ic_dim],
                          [best_h_dim1, best_h_dim2, best_h_dim3],
                          [best_dropout_rate_e, best_dropout_rate_m, best_dropout_rate_c, best_dropout_rate_clf])\
            .to(device)

        moli_optimiser = torch.optim.Adagrad([
            {'params': moli_model.expression_encoder.parameters(), 'lr': best_lr_e},
            {'params': moli_model.mutation_encoder.parameters(), 'lr': best_lr_m},
            {'params': moli_model.cna_encoder.parameters(), 'lr': best_lr_c},
            {'params': moli_model.classifier.parameters(), 'lr': best_lr_cl, 'weight_decay': best_weight_decay},
        ])

        trip_criterion = torch.nn.TripletMarginLoss(margin=best_margin, p=2)
        bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                                       torch.FloatTensor(x_train_c),
                                                       torch.FloatTensor(y_train))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=best_mini_batch,
                                                   shuffle=False,
                                                   num_workers=8, sampler=sampler, pin_memory=True, drop_last=True)

        validate_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e), torch.FloatTensor(x_test_m),
                                                          torch.FloatTensor(x_test_c), torch.FloatTensor(y_test))
        validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=best_mini_batch,
                                                      shuffle=False, num_workers=8, pin_memory=True)

        num_minibatches = int(n_sample_e / best_mini_batch)
        cost_train = 0
        auc_train = []
        for _ in trange(best_epochs):
            auc_train, cost_train = network_training_util.train(train_loader, moli_model,
                                                                moli_optimiser,
                                                                all_triplet_selector,
                                                                trip_criterion,
                                                                bce_with_logits_loss, device,
                                                                best_gamma)
        auc_test = network_training_util.validate(validate_loader, moli_model, device)
        print(f'{drug}: AUROC Train = {auc_train[-1]}')
        print(f'{drug}: AUROC Test = {auc_test}')


# possible = gemcitabine_tcga, cisplatin, docetaxel, erlotinib, cetuximab, gemcitabine_pdx, paclitaxel
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=False, type=bool)
    parser.add_argument('--random_search_iteration', default=1, type=int)
    parser.add_argument('--drug', default='paclitaxel', choices=['gemcitabine_tcga', 'cisplatin', 'docetaxel',
                                                                 'erlotinib', 'cetuximab', 'gemcitabine_pdx',
                                                                 'paclitaxel'])
    args = parser.parse_args()

    with open("../utils/hyperparameter.json") as json_data_file:
        hyperparameter = json.load(json_data_file)
    drug_hyperparameters = hyperparameter[args.drug]
    cv_and_train(drug_hyperparameters, args.drug, args.test, args.random_search_iteration)
