import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange

from utils import network_training_util
from models.moli_model import Moli
from siamese_triplet.utils import AllTripletSelector


def main(optimal_parameters):
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

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

    data_path = Path('../../../data/')
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

    response_test[response_test.response == 'R'] = 0
    response_test[response_test.response == 'S'] = 1
    response_train[response_train.response == 'R'] = 0
    response_train[response_train.response == 'S'] = 1

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

    train_loader = network_training_util.create_dataloader(x_train_e, x_train_m, x_train_c, y_train, mini_batch,
                                                           pin_memory, sampler, True)
    test_loader = network_training_util.create_dataloader(x_test_e, x_test_m, x_test_c, y_test, mini_batch, pin_memory)

    n_sample_e, ie_dim = x_train_e.shape
    _, im_dim = x_train_m.shape
    _, ic_dim = x_train_c.shape

    triplet_selector = AllTripletSelector()

    moli_model = Moli([ie_dim, im_dim, ic_dim], [h_dim1, h_dim2, h_dim3],
                      [dropout_rate_e, dropout_rate_m, dropout_rate_c,
                       dropout_rate_clf]).to(device)

    moli_optimiser = torch.optim.Adagrad([
        {'params': moli_model.expression_encoder.parameters(), 'lr': lr_e},
        {'params': moli_model.mutation_encoder.parameters(), 'lr': lr_m},
        {'params': moli_model.cna_encoder.parameters(), 'lr': lr_c},
        {'params': moli_model.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
    ])

    trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    cross_entropy = torch.nn.BCEWithLogitsLoss()

    auc = 0
    for _ in trange(epochs):
        auc, cost = network_training_util.train(train_loader, moli_model, moli_optimiser, triplet_selector,
                                                trip_criterion, cross_entropy, device, gamma)
    print(f'{optimal_parameters["drug"]}: AUROC Train = {auc}')

    # test
    auc_test = network_training_util.validate(test_loader, moli_model, device)
    print(f'{optimal_parameters["drug"]}: AUROC Test = {auc_test}')


if __name__ == "__main__":
    with open("../../utils/hyperparameter.json") as json_data_file:
        hyperparameter = json.load(json_data_file)
    for drug in hyperparameter:
        drug_hyperparameters = hyperparameter[drug]
    main(drug_hyperparameters)
