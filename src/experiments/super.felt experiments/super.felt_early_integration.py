import argparse
import sys
from pathlib import Path

import sklearn.preprocessing as sk
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import optim
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.network_training_util import calculate_mean_and_std_auc, get_triplet_selector, feature_selection
from utils import multi_omics_data
from models.super_felt_model import SupervisedEncoder, OnlineTestTriplet, AdaptedClassifier, \
    SupervisedVariationalEncoder, AutoEncoder, VariationalAutoEncoder

from utils.choose_gpu import get_free_gpu

drugs = {
    'Gemcitabine_tcga': 'TCGA',
    'Gemcitabine_pdx': 'PDX',
    'Cisplatin': 'TCGA',
    'Docetaxel': 'TCGA',
    'Erlotinib': 'PDX',
    'Cetuximab': 'PDX',
    'Paclitaxel': 'PDX'
}

# common hyperparameters
mb_size = 55
OE_dim = 512
marg = 1
lrE = 0.01
lrCL = 0.01
sigmoid = torch.nn.Sigmoid()
mse = torch.nn.MSELoss()

hyperparameters_set_list = [
 {'E_dr': 0.1, 'C_dr': 0.1, 'Cwd': 0.0, 'Ewd': 0.0},
 {'E_dr': 0.3, 'C_dr': 0.3, 'Cwd': 0.01, 'Ewd': 0.01},
 {'E_dr': 0.3, 'C_dr': 0.3, 'Cwd': 0.01, 'Ewd': 0.05},
 {'E_dr': 0.5, 'C_dr': 0.5, 'Cwd': 0.01, 'Ewd': 0.01},
 {'E_dr': 0.5, 'C_dr': 0.7, 'Cwd': 0.15, 'Ewd': 0.1},
 {'E_dr': 0.3, 'C_dr': 0.5, 'Cwd': 0.01, 'Ewd': 0.01},
 {'E_dr': 0.4, 'C_dr': 0.4, 'Cwd': 0.01, 'Ewd': 0.01},
 {'E_dr': 0.5, 'C_dr': 0.5, 'Cwd': 0.1, 'Ewd': 0.1} ]


Supervised_Encoder_epoch = 10
Classifier_epoch = 5

random_seed = 42


def kl_loss_function(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


def super_felt(experiment_name, drug_name, extern_dataset_name, gpu_number, noisy, architecture):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    triplet_selector = get_triplet_selector(marg, False)
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=marg, p=2)
    BCE_loss_fun = torch.nn.BCEWithLogitsLoss()

    data_path = Path('..', '..', '..', 'data')
    result_path = Path('..', '..', '..', 'results', 'experiments', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = open(result_path / 'results.txt', 'w')
    gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
        = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name, return_data_frames=True)
    GDSCE, GDSCM, GDSCC = feature_selection(gdsc_e, gdsc_m, gdsc_c)
    expression_intersection_genes_index = GDSCE.columns.intersection(extern_e.columns)
    mutation_intersection_genes_index = GDSCM.columns.intersection(extern_m.columns)
    cna_intersection_genes_index = GDSCC.columns.intersection(extern_c.columns)
    GDSCR = gdsc_r
    gdsc = torch.cat((torch.FloatTensor(GDSCE.values), torch.FloatTensor(GDSCM.values),
                      torch.FloatTensor(GDSCC.values)), 1)

    ExternalE = extern_e.loc[:, expression_intersection_genes_index]
    ExternalM = extern_m.loc[:, mutation_intersection_genes_index]
    ExternalC = extern_c.loc[:, cna_intersection_genes_index]
    ExternalY = extern_r

    external = torch.cat((torch.FloatTensor(ExternalE.values), torch.FloatTensor(ExternalM.values),
                          torch.FloatTensor(ExternalC.values)), 1)

    test_auc_list = []
    extern_auc_list = []
    test_auprc_list = []
    extern_auprc_list = []
    if architecture in ('vae', 'supervised-vae'):
        encoder = VariationalAutoEncoder
    elif architecture in ('ae', 'supervised-ae'):
        encoder = AutoEncoder
    elif architecture == 'supervised-ve':
        encoder = SupervisedVariationalEncoder
    else:
        encoder = SupervisedEncoder
    cv_splits = 5
    skf_outer = StratifiedKFold(n_splits=cv_splits, random_state=random_seed, shuffle=True)
    for train_index_outer, test_index in tqdm(skf_outer.split(gdsc, GDSCR), total=skf_outer.get_n_splits(),
                                              desc=" Outer k-fold"):
        X_train_val = gdsc[train_index_outer]
        X_test = gdsc[test_index]
        Y_train_val = GDSCR[train_index_outer]
        Y_test = GDSCR[test_index]
        skf = StratifiedKFold(n_splits=cv_splits)

        best_auroc = -1
        best_hyperparameter = None
        for hyperparameters_set in hyperparameters_set_list:
            E_dr = hyperparameters_set['E_dr']
            C_dr = hyperparameters_set['C_dr']
            Cwd = hyperparameters_set['Cwd']
            Ewd = hyperparameters_set['Ewd']
            all_validation_aurocs = []
            for train_index, validate_index in tqdm(skf.split(X_train_val, Y_train_val), total=skf.get_n_splits(),
                                                    desc="k-fold"):
                X_train = X_train_val[train_index]
                X_val = X_train_val[validate_index]
                Y_train = Y_train_val[train_index]
                Y_val = Y_train_val[validate_index]
                class_sample_count = np.array([len(np.where(Y_train == t)[0]) for t in np.unique(Y_train)])
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[t] for t in Y_train])

                samples_weight = torch.from_numpy(samples_weight)
                sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                                replacement=True)
                scalerGDSC = sk.StandardScaler()
                scalerGDSC.fit(X_train)
                X_train = scalerGDSC.transform(X_train)
                X_val = torch.FloatTensor(scalerGDSC.transform(X_val)).to(device)
                trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train),
                                                              torch.FloatTensor(Y_train.astype(int)))

                trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                                          num_workers=1, sampler=sampler)

                n_sampE, IE_dim = X_train.shape
                Supervised_Encoder = encoder(IE_dim, OE_dim, E_dr)
                Supervised_Encoder.to(device)

                optimizer = optim.Adagrad(Supervised_Encoder.parameters(), lr=lrE, weight_decay=Ewd)
                TripSel = OnlineTestTriplet(marg, triplet_selector)

                train_Clas = AdaptedClassifier(OE_dim, C_dr)
                train_Clas.to(device)
                Cl_optimizer = optim.Adagrad(train_Clas.parameters(), lr=lrCL, weight_decay=Cwd)

                # train each Supervised_Encoder with triplet loss
                for e_epoch in range(Supervised_Encoder_epoch):
                    Supervised_Encoder.train()
                    for i, (dataE, target) in enumerate(trainLoader):
                        optimizer.zero_grad()
                        if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                            original_E = dataE.clone()
                            dataE = dataE.to(device)
                            if noisy:
                                dataE += torch.normal(0.0, 0.05, dataE.shape)
                            if architecture == 'ae':
                                encoded_E, reconstruction = Supervised_Encoder(dataE)
                                E_loss = mse(reconstruction, original_E)
                            elif architecture == 'vae':
                                encoded_E, reconstruction, mu, log_var = Supervised_Encoder(dataE)
                                print(mse(reconstruction, original_E))
                                print(kl_loss_function(mu, log_var))
                                E_loss = mse(reconstruction, original_E) + kl_loss_function(mu, log_var)
                            elif architecture == 'supervised-ae':
                                encoded_E, reconstruction = Supervised_Encoder(dataE)
                                E_Triplets_list = TripSel(encoded_E, target)
                                E_triplets_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                                                encoded_E[E_Triplets_list[:, 1], :],
                                                                encoded_E[E_Triplets_list[:, 2], :])
                                E_reconstruction_loss = mse(reconstruction, original_E)
                                E_loss = E_triplets_loss + E_reconstruction_loss
                            elif architecture == 'supervised-vae':
                                encoded_E, reconstruction, mu, log_var = Supervised_Encoder(dataE)
                                E_Triplets_list = TripSel(encoded_E, target)
                                E_triplets_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                                                encoded_E[E_Triplets_list[:, 1], :],
                                                                encoded_E[E_Triplets_list[:, 2], :])
                                E_reconstruction_loss = mse(reconstruction, original_E)
                                E_loss = E_triplets_loss + E_reconstruction_loss + kl_loss_function(mu, log_var)
                            elif architecture == 'supervised-ve':
                                encoded_E, mu, log_var = Supervised_Encoder(dataE)
                                E_Triplets_list = TripSel(encoded_E, target)
                                E_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                                       encoded_E[E_Triplets_list[:, 1], :],
                                                       encoded_E[E_Triplets_list[:, 2], :]) +\
                                        0.1 * kl_loss_function(mu, log_var)
                            else:
                                encoded_E = Supervised_Encoder(dataE)
                                E_Triplets_list = TripSel(encoded_E, target)
                                E_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                                       encoded_E[E_Triplets_list[:, 1], :],
                                                       encoded_E[E_Triplets_list[:, 2], :])
                            E_loss.backward()
                            optimizer.step()
                Supervised_Encoder.eval()

                # train classifier
                for cl_epoch in range(Classifier_epoch):
                    train_Clas.train()
                    for i, (dataE, target) in enumerate(trainLoader):
                        Cl_optimizer.zero_grad()
                        if torch.mean(target) != 0. and torch.mean(target) != 1.:
                            dataE = dataE.to(device)
                            target = target.to(device)
                            encoded_E = Supervised_Encoder.encode(dataE)
                            Pred = train_Clas(encoded_E, torch.FloatTensor().to(device),
                                              torch.FloatTensor().to(device))
                            cl_loss = BCE_loss_fun(Pred, target.view(-1, 1))
                            cl_loss.backward()
                            Cl_optimizer.step()

                    with torch.no_grad():
                        train_Clas.eval()
                        """
                            inner validation
                        """
                        encoded_val_E = Supervised_Encoder.encode(X_val)

                        #print(encoded_val_C)
                        test_Pred = train_Clas(encoded_val_E, torch.FloatTensor().to(device),
                                               torch.FloatTensor().to(device))
                        test_Pred = sigmoid(test_Pred)
                        #print(test_Pred)
                        val_AUC = roc_auc_score(Y_val, test_Pred.cpu().detach().numpy())
                print(f'validation auroc: {val_AUC}')

                all_validation_aurocs.append(val_AUC)

            val_AUC = np.mean(all_validation_aurocs)
            if val_AUC > best_auroc:
                best_auroc = val_AUC
                best_hyperparameter = hyperparameters_set

        # retrain best
        E_dr = best_hyperparameter['E_dr']
        C_dr = best_hyperparameter['C_dr']
        Cwd = best_hyperparameter['Cwd']
        Ewd = best_hyperparameter['Ewd']
        class_sample_count = np.array([len(np.where(Y_train_val == t)[0]) for t in np.unique(Y_train_val)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in Y_train_val])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)
        final_scalerGDSC = sk.StandardScaler()
        final_scalerGDSC.fit(X_train_val)
        X_train_val = final_scalerGDSC.transform(X_train_val)
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_val),
                                                      torch.FloatTensor(Y_train_val.astype(int)))

        trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                                  num_workers=1, sampler=sampler)

        n_sampE, IE_dim = X_train_val.shape

        final_E_Supervised_Encoder = encoder(IE_dim, OE_dim, E_dr)

        final_E_Supervised_Encoder.to(device)

        optimizer = optim.Adagrad(final_E_Supervised_Encoder.parameters(), lr=lrE, weight_decay=Ewd)
        TripSel = OnlineTestTriplet(marg, triplet_selector)

        final_Clas = AdaptedClassifier(OE_dim, C_dr)
        final_Clas.to(device)
        Cl_optimizer = optim.Adagrad(final_Clas.parameters(), lr=lrCL, weight_decay=Cwd)

        # train each Supervised_Encoder with triplet loss
        for e_epoch in range(Supervised_Encoder_epoch):
            final_E_Supervised_Encoder.train()
            for i, (dataE, target) in enumerate(trainLoader):
                if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                    dataE = dataE.to(device)
                    originalE = dataE.clone()
                    if noisy:
                        dataE += torch.normal(0.0, 0.05, dataE.shape)
                    if architecture == 'ae':
                        encoded_E, reconstruction = final_E_Supervised_Encoder(dataE)
                        E_loss = mse(reconstruction, originalE)
                    elif architecture == 'supervised-ae':
                        encoded_E, reconstruction = final_E_Supervised_Encoder(dataE)
                        E_Triplets_list = TripSel(encoded_E, target)
                        E_triplets_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                                        encoded_E[E_Triplets_list[:, 1], :],
                                                        encoded_E[E_Triplets_list[:, 2], :])
                        E_reconstruction_loss = mse(reconstruction, originalE)
                        E_loss = E_triplets_loss + E_reconstruction_loss
                    elif architecture == 'vae':
                        encoded_E, reconstruction, mu, log_var = final_E_Supervised_Encoder(dataE)
                        E_loss = mse(reconstruction, originalE) + kl_loss_function(mu, log_var)
                    elif architecture == 'supervised-vae':
                        encoded_E, reconstruction, mu, log_var = final_E_Supervised_Encoder(dataE)
                        E_Triplets_list = TripSel(encoded_E, target)
                        E_triplets_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                                        encoded_E[E_Triplets_list[:, 1], :],
                                                        encoded_E[E_Triplets_list[:, 2], :])
                        E_reconstruction_loss = mse(reconstruction, originalE)
                        E_loss = E_triplets_loss + E_reconstruction_loss + kl_loss_function(mu, log_var)
                    elif architecture == 'supervised-ve':
                        encoded_E, mu, log_var = final_E_Supervised_Encoder(dataE)
                        E_Triplets_list = TripSel(encoded_E, target)
                        E_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                               encoded_E[E_Triplets_list[:, 1], :],
                                               encoded_E[E_Triplets_list[:, 2], :]) + 0.1 * kl_loss_function(mu, log_var)
                    else:
                        encoded_E = final_E_Supervised_Encoder(dataE)
                        E_Triplets_list = TripSel(encoded_E, target)
                        E_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                               encoded_E[E_Triplets_list[:, 1], :],
                                               encoded_E[E_Triplets_list[:, 2], :])

                    optimizer.zero_grad()
                    E_loss.backward()
                    optimizer.step()

        final_E_Supervised_Encoder.eval()

        # train classifier
        for cl_epoch in range(Classifier_epoch):
            final_Clas.train()
            for i, (dataE,  target) in enumerate(trainLoader):
                Cl_optimizer.zero_grad()
                if torch.mean(target) != 0. and torch.mean(target) != 1.:
                    dataE = dataE.to(device)
                    target = target.to(device)
                    encoded_E = final_E_Supervised_Encoder.encode(dataE)

                    Pred = final_Clas(encoded_E, torch.FloatTensor().to(device),
                                      torch.FloatTensor().to(device))
                    cl_loss = BCE_loss_fun(Pred, target.view(-1, 1))

                    cl_loss.backward()
                    Cl_optimizer.step()

        final_Clas.eval()

        # Test
        X_test = torch.FloatTensor(final_scalerGDSC.transform(X_test))
        encoded_test_E = final_E_Supervised_Encoder.encode(torch.FloatTensor(X_test).to(device))
        test_Pred = final_Clas(encoded_test_E, torch.FloatTensor(), torch.FloatTensor())
        test_y_pred = sigmoid(test_Pred).cpu().detach().numpy()

        test_AUC = roc_auc_score(Y_test, test_y_pred)
        test_AUCPR = average_precision_score(Y_test, test_y_pred)

        # Extern
        external = torch.FloatTensor(final_scalerGDSC.transform(external))
        encoded_external_E = final_E_Supervised_Encoder.encode(torch.FloatTensor(external).to(device))
        external_Pred = final_Clas(encoded_external_E, torch.FloatTensor(), torch.FloatTensor())
        external_Pred = sigmoid(external_Pred)
        external_y_pred = external_Pred.cpu().detach().numpy()
        external_AUC = roc_auc_score(ExternalY, external_y_pred)
        external_AUCPR = average_precision_score(ExternalY, external_y_pred)

        test_auc_list.append(test_AUC)
        extern_auc_list.append(external_AUC)
        test_auprc_list.append(test_AUCPR)
        extern_auprc_list.append(external_AUCPR)

    print("Done!")

    result_dict = {
        'test auroc': test_auc_list,
        'test auprc': test_auprc_list,
        'extern auroc': extern_auc_list,
        'extern auprc': extern_auprc_list
    }
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)
    result_file.write(f'\n test auroc list: {test_auc_list} \n')
    result_file.write(f'\n test auprc list: {test_auprc_list} \n')
    result_file.write(f'\n extern auroc list: {extern_auc_list} \n')
    result_file.write(f'\n extern auprc list: {extern_auprc_list} \n')
    result_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--gpu_number', type=int)
    parser.add_argument('--noisy', default=False, action='store_true')
    parser.add_argument('--architecture', default=None, choices=['supervised-vae', 'vae', 'ae', 'supervised-ae',
                                                                 'supervised-e', 'supervised-ve'])
    parser.add_argument('--drug', default='all', choices=['Gemcitabine_tcga', 'Gemcitabine_pdx', 'Cisplatin',
                                                          'Docetaxel', 'Erlotinib', 'Cetuximab', 'Paclitaxel'])
    args = parser.parse_args()

    for drug, extern_dataset in drugs.items():
        super_felt(args.experiment_name, drug, extern_dataset, args.gpu_number, args.noisy, args.architecture)
