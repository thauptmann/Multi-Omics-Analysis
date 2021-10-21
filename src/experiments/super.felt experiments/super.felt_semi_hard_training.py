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
from siamese_triplet.utils import HardestNegativeTripletSelector, SemihardNegativeTripletSelector
from utils.network_training_util import calculate_mean_and_std_auc, feature_selection
from utils import multi_omics_data
from super_felt_model import SupervisedEncoder, Classifier

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
mb_size_list = [128, 256, 512, 1024]
OE_dim = 256
OM_dim = 32
OC_dim = 64
margin_list = [0.2, 1]
lrE = 0.01
lrM = 0.01
lrC = 0.01
lrCL = 0.01

hyperparameters_set_list = []
hyperparameters_set1 = {'E_dr': 0.1, 'C_dr': 0.1, 'Cwd': 0.0, 'Ewd': 0.0}
hyperparameters_set2 = {'E_dr': 0.3, 'C_dr': 0.3, 'Cwd': 0.01, 'Ewd': 0.01}
hyperparameters_set3 = {'E_dr': 0.3, 'C_dr': 0.3, 'Cwd': 0.01, 'Ewd': 0.05}
hyperparameters_set4 = {'E_dr': 0.5, 'C_dr': 0.5, 'Cwd': 0.01, 'Ewd': 0.01}
hyperparameters_set5 = {'E_dr': 0.5, 'C_dr': 0.7, 'Cwd': 0.15, 'Ewd': 0.1}
hyperparameters_set6 = {'E_dr': 0.3, 'C_dr': 0.5, 'Cwd': 0.01, 'Ewd': 0.01}
hyperparameters_set7 = {'E_dr': 0.4, 'C_dr': 0.4, 'Cwd': 0.01, 'Ewd': 0.01}
hyperparameters_set8 = {'E_dr': 0.5, 'C_dr': 0.5, 'Cwd': 0.1, 'Ewd': 0.1}

for hyperparameter_set in (hyperparameters_set1, hyperparameters_set2, hyperparameters_set3,
                           hyperparameters_set4, hyperparameters_set5, hyperparameters_set6, hyperparameters_set7,
                           hyperparameters_set8):
    for margin in margin_list:
        hyperparameter_set['margin'] = margin
        for batch_size in mb_size_list:
            hyperparameter_set['batch_size'] = batch_size
            hyperparameters_set_list.append(hyperparameter_set.copy())


E_Supervised_Encoder_epoch = 10
C_Supervised_Encoder_epoch = 5
M_Supervised_Encoder_epoch = 3
Classifier_epoch = 5

random_seed = 42


def super_felt(experiment_name, drug_name, extern_dataset_name, gpu_number):
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

    data_path = Path('..', '..', '..', 'data')
    result_path = Path('..', '..', '..', 'results', 'super.felt', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = open(result_path / 'results.txt', 'w')
    gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
        = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name, return_data_frames=True)
    GDSCE, GDSCM, GDSCC = feature_selection(gdsc_e, gdsc_m, gdsc_c)
    expression_intersection_genes_index = GDSCE.columns.intersection(extern_e.columns)
    mutation_intersection_genes_index = GDSCM.columns.intersection(extern_m.columns)
    cna_intersection_genes_index = GDSCC.columns.intersection(extern_c.columns)
    GDSCR = gdsc_r

    ExternalE = extern_e.loc[:, expression_intersection_genes_index]
    ExternalM = extern_m.loc[:, mutation_intersection_genes_index]
    ExternalC = extern_c.loc[:, cna_intersection_genes_index]
    ExternalY = extern_r

    test_auc_list = []
    extern_auc_list = []
    test_auprc_list = []
    extern_auprc_list = []
    cv_splits = 5
    BCE_loss_fun = torch.nn.BCELoss()
    skf_outer = StratifiedKFold(n_splits=cv_splits, random_state=random_seed, shuffle=True)
    for train_index_outer, test_index in tqdm(skf_outer.split(GDSCE, GDSCR), total=skf_outer.get_n_splits(),
                                              desc=" Outer k-fold"):
        X_train_valE = GDSCE.to_numpy()[train_index_outer]
        X_testE = GDSCE.to_numpy()[test_index]
        X_train_valM = GDSCM.to_numpy()[train_index_outer]
        X_testM = GDSCM.to_numpy()[test_index]
        X_train_valC = GDSCC.to_numpy()[train_index_outer]
        X_testC = GDSCC.to_numpy()[test_index]
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
            marg = hyperparameters_set['margin']
            mb_size = hyperparameters_set['batch_size']
            all_validation_aurocs = []

            hardest_triplet_selector = HardestNegativeTripletSelector(marg)
            semi_hard_triplet_selector = SemihardNegativeTripletSelector(marg)
            trip_loss_fun = torch.nn.TripletMarginLoss(margin=marg, p=2)

            for train_index, validate_index in tqdm(skf.split(X_train_valE, Y_train_val), total=skf.get_n_splits(),
                                                    desc="k-fold"):
                X_trainE = X_train_valE[train_index]
                X_valE = X_train_valE[validate_index]
                X_trainM = X_train_valM[train_index]
                X_valM = X_train_valM[validate_index]
                X_trainC = X_train_valC[train_index]
                X_valC = X_train_valC[validate_index]
                Y_train = Y_train_val[train_index]
                Y_val = Y_train_val[validate_index]
                class_sample_count = np.array([len(np.where(Y_train == t)[0]) for t in np.unique(Y_train)])
                weight = 1. / class_sample_count
                samples_weight = np.array([weight[t] for t in Y_train])

                samples_weight = torch.from_numpy(samples_weight)
                sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                                replacement=True)
                scalerGDSC = sk.StandardScaler()
                scalerGDSC.fit(X_trainE)
                X_trainE = scalerGDSC.transform(X_trainE)
                X_valE = torch.FloatTensor(scalerGDSC.transform(X_valE)).to(device)
                trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                              torch.FloatTensor(X_trainC),
                                                              torch.FloatTensor(Y_train.astype(int)))

                trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                                          num_workers=1, sampler=sampler)

                n_sampE, IE_dim = X_trainE.shape
                n_sampM, IM_dim = X_trainM.shape
                n_sampC, IC_dim = X_trainC.shape

                cost_tr = []
                auc_tr = []

                E_Supervised_Encoder = SupervisedEncoder(IE_dim, OE_dim, E_dr)
                M_Supervised_Encoder = SupervisedEncoder(IM_dim, OM_dim, E_dr)
                C_Supervised_Encoder = SupervisedEncoder(IC_dim, OC_dim, E_dr)

                E_Supervised_Encoder.to(device)
                M_Supervised_Encoder.to(device)
                C_Supervised_Encoder.to(device)

                E_optimizer = optim.Adagrad(E_Supervised_Encoder.parameters(), lr=lrE, weight_decay=Ewd)
                M_optimizer = optim.Adagrad(M_Supervised_Encoder.parameters(), lr=lrM, weight_decay=Ewd)
                C_optimizer = optim.Adagrad(C_Supervised_Encoder.parameters(), lr=lrC, weight_decay=Ewd)

                final_Clas = Classifier(OE_dim + OM_dim + OC_dim, 1, C_dr)
                final_Clas.to(device)
                Cl_optimizer = optim.Adagrad(final_Clas.parameters(), lr=lrCL, weight_decay=Cwd)

                # train each Supervised_Encoder with triplet loss
                pre_loss = 100
                break_num = 0
                for e_epoch in range(E_Supervised_Encoder_epoch):
                    E_Supervised_Encoder.train()
                    flag = 0
                    for i, (dataE, _, _, target) in enumerate(trainLoader):
                        if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                            dataE = dataE.to(device)
                            encoded_E = E_Supervised_Encoder(dataE)
                            if e_epoch < (E_Supervised_Encoder_epoch-2):
                                E_Triplets_list = semi_hard_triplet_selector(encoded_E, target)
                            else:
                                E_Triplets_list = hardest_triplet_selector(encoded_E, target)
                            E_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                                   encoded_E[E_Triplets_list[:, 1], :],
                                                   encoded_E[E_Triplets_list[:, 2], :])

                            E_optimizer.zero_grad()
                            E_loss.backward()
                            E_optimizer.step()
                            flag = 1

                    if flag == 1:
                        del E_loss
                    with torch.no_grad():
                        E_Supervised_Encoder.eval()
                        """
                            inner validation
                        """
                        encoded_val_E = E_Supervised_Encoder(X_valE)
                        E_Triplets_list = hardest_triplet_selector(encoded_val_E, torch.FloatTensor(Y_val))
                        val_E_loss = trip_loss_fun(encoded_val_E[E_Triplets_list[:, 0], :],
                                                   encoded_val_E[E_Triplets_list[:, 1], :],
                                                   encoded_val_E[E_Triplets_list[:, 2], :])

                        if pre_loss <= val_E_loss:
                            break_num += 1

                        if break_num > 1:
                            pass  # break
                        else:
                            pre_loss = val_E_loss

                E_Supervised_Encoder.eval()

                pre_loss = 100
                break_num = 0
                for m_epoch in range(M_Supervised_Encoder_epoch):
                    M_Supervised_Encoder.train().to(device)
                    flag = 0
                    for i, (_, dataM, _, target) in enumerate(trainLoader):
                        if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                            dataM = dataM.to(device)
                            encoded_M = M_Supervised_Encoder(dataM)
                            if m_epoch < (M_Supervised_Encoder_epoch - 2):
                                M_Triplets_list = semi_hard_triplet_selector(encoded_M, target)
                            else:
                                M_Triplets_list = hardest_triplet_selector(encoded_M, target)
                            M_loss = trip_loss_fun(encoded_M[M_Triplets_list[:, 0], :],
                                                   encoded_M[M_Triplets_list[:, 1], :],
                                                   encoded_M[M_Triplets_list[:, 2], :])

                            M_optimizer.zero_grad()
                            M_loss.backward()
                            M_optimizer.step()
                            flag = 1

                    if flag == 1:
                        del M_loss
                    with torch.no_grad():
                        M_Supervised_Encoder.eval()
                        """
                            validation
                        """
                        encoded_val_M = M_Supervised_Encoder(torch.FloatTensor(X_valM).to(device))
                        M_Triplets_list = hardest_triplet_selector(encoded_val_M, torch.FloatTensor(Y_val))
                        val_M_loss = trip_loss_fun(encoded_val_M[M_Triplets_list[:, 0], :],
                                                   encoded_val_M[M_Triplets_list[:, 1], :],
                                                   encoded_val_M[M_Triplets_list[:, 2], :])

                        if pre_loss <= val_M_loss:
                            break_num += 1

                        if break_num > 1:
                            pass  # break
                        else:
                            pre_loss = val_M_loss

                M_Supervised_Encoder.eval()

                pre_loss = 100
                break_num = 0
                for c_epoch in range(C_Supervised_Encoder_epoch):
                    C_Supervised_Encoder.train()
                    flag = 0
                    for i, (_, _, dataC, target) in enumerate(trainLoader):
                        if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                            dataC = dataC.to(device)
                            encoded_C = C_Supervised_Encoder(dataC)

                            if c_epoch < (C_Supervised_Encoder_epoch - 2):
                                C_Triplets_list = semi_hard_triplet_selector(encoded_C, target)
                            else:
                                C_Triplets_list = hardest_triplet_selector(encoded_C, target)
                            C_loss = trip_loss_fun(encoded_C[C_Triplets_list[:, 0], :],
                                                   encoded_C[C_Triplets_list[:, 1], :],
                                                   encoded_C[C_Triplets_list[:, 2], :])

                            C_optimizer.zero_grad()
                            C_loss.backward()
                            C_optimizer.step()

                            flag = 1

                    if flag == 1:
                        del C_loss
                    with torch.no_grad():
                        C_Supervised_Encoder.eval()
                        """
                            inner validation
                        """
                        encoded_val_C = C_Supervised_Encoder(torch.FloatTensor(X_valC).to(device))
                        C_Triplets_list = hardest_triplet_selector(encoded_val_C, torch.FloatTensor(Y_val))
                        val_C_loss = trip_loss_fun(encoded_val_C[C_Triplets_list[:, 0], :],
                                                   encoded_val_C[C_Triplets_list[:, 1], :],
                                                   encoded_val_C[C_Triplets_list[:, 2], :])
                        if pre_loss <= val_C_loss:
                            break_num += 1

                        if break_num > 1:
                            pass  # break
                        else:
                            pre_loss = val_C_loss

                C_Supervised_Encoder.eval()

                # train classifier
                for cl_epoch in range(Classifier_epoch):
                    epoch_cost = 0
                    epoch_auc_list = []
                    num_minibatches = int(n_sampE / mb_size)
                    flag = 0
                    final_Clas.train()
                    for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):

                        if torch.mean(target) != 0. and torch.mean(target) != 1.:
                            dataE = dataE.to(device)
                            dataM = dataM.to(device)
                            dataC = dataC.to(device)
                            target = target.to(device)
                            encoded_E = E_Supervised_Encoder(dataE)
                            encoded_M = M_Supervised_Encoder(dataM)
                            encoded_C = C_Supervised_Encoder(dataC)

                            Pred = final_Clas(encoded_E, encoded_M, encoded_C)

                            y_true = target.view(-1, 1).cpu()

                            cl_loss = BCE_loss_fun(Pred, target.view(-1, 1))
                            y_pred = Pred.cpu()
                            AUC = roc_auc_score(y_true.detach().numpy(), y_pred.detach().numpy())

                            Cl_optimizer.zero_grad()
                            cl_loss.backward()
                            Cl_optimizer.step()

                            epoch_cost = epoch_cost + (cl_loss / num_minibatches)
                            epoch_auc_list.append(AUC)
                            flag = 1

                    if flag == 1:
                        cost_tr.append(torch.mean(epoch_cost))
                        auc_tr.append(np.mean(epoch_auc_list))
                        del cl_loss

                    with torch.no_grad():
                        final_Clas.eval()
                        """
                            inner validation
                        """
                        encoded_val_E = E_Supervised_Encoder(X_valE)
                        encoded_val_M = M_Supervised_Encoder(torch.FloatTensor(X_valM).to(device))
                        encoded_val_C = C_Supervised_Encoder(torch.FloatTensor(X_valC).to(device))

                        test_Pred = final_Clas(encoded_val_E, encoded_val_M, encoded_val_C)

                        test_y_true = Y_val
                        test_y_pred = test_Pred.cpu()

                        val_AUC = roc_auc_score(test_y_true, test_y_pred.detach().numpy())

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
        marg = best_hyperparameter['margin']
        mb_size = best_hyperparameter['batch_size']
        class_sample_count = np.array([len(np.where(Y_train_val == t)[0]) for t in np.unique(Y_train_val)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in Y_train_val])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)
        final_scalerGDSC = sk.StandardScaler()
        final_scalerGDSC.fit(X_train_valE)
        X_train_valE = final_scalerGDSC.transform(X_train_valE)
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_valE), torch.FloatTensor(X_train_valM),
                                                      torch.FloatTensor(X_train_valC),
                                                      torch.FloatTensor(Y_train_val.astype(int)))

        trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                                  num_workers=1, sampler=sampler)

        n_sampE, IE_dim = X_train_valE.shape
        n_sampM, IM_dim = X_train_valM.shape
        n_sampC, IC_dim = X_train_valC.shape

        final_E_Supervised_Encoder = SupervisedEncoder(IE_dim, OE_dim, E_dr)
        final_M_Supervised_Encoder = SupervisedEncoder(IM_dim, OM_dim, E_dr)
        final_C_Supervised_Encoder = SupervisedEncoder(IC_dim, OC_dim, E_dr)

        final_E_Supervised_Encoder.to(device)
        final_M_Supervised_Encoder.to(device)
        final_C_Supervised_Encoder.to(device)

        E_optimizer = optim.Adagrad(final_E_Supervised_Encoder.parameters(), lr=lrE, weight_decay=Ewd)
        M_optimizer = optim.Adagrad(final_M_Supervised_Encoder.parameters(), lr=lrM, weight_decay=Ewd)
        C_optimizer = optim.Adagrad(final_C_Supervised_Encoder.parameters(), lr=lrC, weight_decay=Ewd)

        hardest_triplet_selector = HardestNegativeTripletSelector(marg)
        semi_hard_triplet_selector = SemihardNegativeTripletSelector(marg)
        trip_loss_fun = torch.nn.TripletMarginLoss(margin=marg, p=2)

        final_Clas = Classifier(OE_dim + OM_dim + OC_dim, 1, C_dr)
        final_Clas.to(device)
        Cl_optimizer = optim.Adagrad(final_Clas.parameters(), lr=lrCL, weight_decay=Cwd)

        # train each Supervised_Encoder with triplet loss
        for e_epoch in range(E_Supervised_Encoder_epoch):
            final_E_Supervised_Encoder.train()
            for i, (dataE, _, _, target) in enumerate(trainLoader):
                if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                    dataE = dataE.to(device)
                    encoded_E = final_E_Supervised_Encoder(dataE)

                    if e_epoch < (E_Supervised_Encoder_epoch - 2):
                        E_Triplets_list = semi_hard_triplet_selector(encoded_E, target)
                    else:
                        E_Triplets_list = hardest_triplet_selector(encoded_E, target)
                    E_loss = trip_loss_fun(encoded_E[E_Triplets_list[:, 0], :],
                                           encoded_E[E_Triplets_list[:, 1], :],
                                           encoded_E[E_Triplets_list[:, 2], :])

                    E_optimizer.zero_grad()
                    E_loss.backward()
                    E_optimizer.step()

        final_E_Supervised_Encoder.eval()

        for m_epoch in range(M_Supervised_Encoder_epoch):
            final_M_Supervised_Encoder.train().to(device)
            for i, (_, dataM, _, target) in enumerate(trainLoader):
                if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                    dataM = dataM.to(device)
                    encoded_M = final_M_Supervised_Encoder(dataM)
                    if m_epoch < (M_Supervised_Encoder_epoch - 2):
                        M_Triplets_list = semi_hard_triplet_selector(encoded_M, target)
                    else:
                        M_Triplets_list = hardest_triplet_selector(encoded_M, target)
                    M_loss = trip_loss_fun(encoded_M[M_Triplets_list[:, 0], :],
                                           encoded_M[M_Triplets_list[:, 1], :],
                                           encoded_M[M_Triplets_list[:, 2], :])

                    M_optimizer.zero_grad()
                    M_loss.backward()
                    M_optimizer.step()

        final_M_Supervised_Encoder.eval()

        for c_epoch in range(C_Supervised_Encoder_epoch):
            final_C_Supervised_Encoder.train()
            for i, (_, _, dataC, target) in enumerate(trainLoader):
                if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                    dataC = dataC.to(device)
                    encoded_C = final_C_Supervised_Encoder(dataC)

                    if c_epoch < (C_Supervised_Encoder_epoch - 2):
                        C_Triplets_list = semi_hard_triplet_selector(encoded_C, target)
                    else:
                        C_Triplets_list = hardest_triplet_selector(encoded_C, target)
                    C_loss = trip_loss_fun(encoded_C[C_Triplets_list[:, 0], :],
                                           encoded_C[C_Triplets_list[:, 1], :],
                                           encoded_C[C_Triplets_list[:, 2], :])

                    C_optimizer.zero_grad()
                    C_loss.backward()
                    C_optimizer.step()

        final_C_Supervised_Encoder.eval()

        # train classifier
        for cl_epoch in range(Classifier_epoch):
            final_Clas.train()
            for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                if torch.mean(target) != 0. and torch.mean(target) != 1.:
                    dataE = dataE.to(device)
                    dataM = dataM.to(device)
                    dataC = dataC.to(device)
                    target = target.to(device)
                    encoded_E = final_E_Supervised_Encoder(dataE)
                    encoded_M = final_M_Supervised_Encoder(dataM)
                    encoded_C = final_C_Supervised_Encoder(dataC)

                    Pred = final_Clas(encoded_E, encoded_M, encoded_C)

                    y_true = target.view(-1, 1).cpu()

                    cl_loss = BCE_loss_fun(Pred, target.view(-1, 1))
                    y_pred = Pred.cpu()
                    AUC = roc_auc_score(y_true.detach().numpy(), y_pred.detach().numpy())

                    Cl_optimizer.zero_grad()
                    cl_loss.backward()
                    Cl_optimizer.step()

            final_Clas.eval()

        # Test
        X_testE = torch.FloatTensor(final_scalerGDSC.transform(X_testE))
        encoded_test_E = final_E_Supervised_Encoder(torch.FloatTensor(X_testE).to(device))
        encoded_test_M = final_M_Supervised_Encoder(torch.FloatTensor(X_testM).to(device))
        encoded_test_C = final_C_Supervised_Encoder(torch.FloatTensor(X_testC).to(device))
        test_Pred = final_Clas(encoded_test_E, encoded_test_M, encoded_test_C)
        test_y_true = Y_test
        test_y_pred = test_Pred.cpu().detach().numpy()
        test_AUC = roc_auc_score(test_y_true, test_y_pred)
        test_AUCPR = average_precision_score(test_y_true, test_y_pred)

        # Extern
        ExternalE = torch.FloatTensor(final_scalerGDSC.transform(ExternalE))
        encoded_external_E = final_E_Supervised_Encoder(torch.FloatTensor(ExternalE).to(device))
        encoded_external_M = final_M_Supervised_Encoder(torch.FloatTensor(ExternalM.to_numpy()).to(device))
        encoded_external_C = final_C_Supervised_Encoder(torch.FloatTensor(ExternalC.to_numpy()).to(device))
        external_Pred = final_Clas(encoded_external_E, encoded_external_M, encoded_external_C)
        external_y_true = ExternalY
        external_y_pred = external_Pred.cpu().detach().numpy()
        external_AUC = roc_auc_score(external_y_true, external_y_pred)
        external_AUCPR = average_precision_score(external_y_true, external_y_pred)

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
    parser.add_argument('--drug', default='all', choices=['Gemcitabine_tcga', 'Gemcitabine_pdx', 'Cisplatin',
                                                          'Docetaxel', 'Erlotinib', 'Cetuximab', 'Paclitaxel'])
    args = parser.parse_args()

    for drug, extern_dataset in drugs.items():
        super_felt(args.experiment_name, drug, extern_dataset, args.gpu_number)
