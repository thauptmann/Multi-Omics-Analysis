import torch
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import trange

import encoder
from sklearn.feature_selection import VarianceThreshold
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from siamese_triplet.metrics import AverageNonzeroTripletsMetric
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

paclitaxel = {'drug': 'paclitaxel', 'mini_batch': 64, 'h_dim1': 512, 'h_dim2': 256, 'h_dim3': 1024, 'lrE': 0.0005,
              'lrM': 0.5, 'lrC': 0.5, 'lrCL': 0.5, 'dropout_rateE': 0.4, 'dropout_rateM': 0.6,
              'dropout_rateC': 0.3, 'weight_decay': 0.01,
              'dropout_rateClf': 0.6, 'gamma': 0.3, 'epochs': 5, 'margin': 1.5}

gemcitabine_pdx = {'drug': 'gemcitabine_pdx', 'mini_batch': 13, 'h_dim1': 256, 'h_dim2': 32, 'h_dim3': 64, 'lrE': 0.05,
                   'lrM': 0.00005, 'lrC': 0.0005, 'lrCL': 0.001, 'dropout_rateE': 0.4, 'dropout_rateM': 0.6,
                   'dropout_rateC': 0.5, 'weight_decay': 0.0001,
                   'dropout_rateClf': 0.3, 'gamma': 0.6, 'epochs': 10, 'margin': 0.5}

cetuximab = {'drug': 'cetuximab', 'mini_batch': 16, 'h_dim1': 32, 'h_dim2': 16, 'h_dim3': 256, 'lrE': 0.001,
             'lrM': 0.0001, 'lrC': 0.00005, 'lrCL': 0.005, 'dropout_rateE': 0.5, 'dropout_rateM': 0.8,
             'dropout_rateC': 0.5, 'weight_decay': 0.0001,
             'dropout_rateClf': 0.3, 'gamma': 0.5, 'epochs': 20, 'margin': 1.5,
             'expression':,
'response':
'mutation':,
'cna':,


             }

erlotinib = {'drug': 'erlotinib', 'mini_batch': 16, 'h_dim1': 32, 'h_dim2': 16, 'h_dim3': 256, 'lrE': 0.001,
             'lrM': 0.0001, 'lrC': 0.00005, 'lrCL': 0.005, 'dropout_rateE': 0.5, 'dropout_rateM': 0.8,
             'dropout_rateC': 0.5, 'weight_decay': 0.0001,
             'dropout_rateClf': 0.3, 'gamma': 0.5, 'epochs': 20, 'margin': 1.5}

docetaxel = {'drug': 'docetaxel', 'mini_batch': 8, 'h_dim1': 16, 'h_dim2': 16, 'h_dim3': 16, 'lrE': 0.0001,
             'lrM': 0.0005, 'lrC': 0.0005, 'lrCL': 0.001, 'dropout_rateE': 0.5, 'dropout_rateM': 0.5,
             'dropout_rateC': 0.5, 'weight_decay': 0.0001,
             'dropout_rateClf': 0.5, 'gamma': 0.4, 'epochs': 10, 'margin': 0.5}

cisplatin = {'drug': 'cisplatin', 'mini_batch': 15, 'h_dim1': 128, 'h_dim2': 128, 'h_dim3': 127, 'lrE': 0.05,
             'lrM': 0.005, 'lrC': 0.005, 'lrCL': 0.0005, 'dropout_rateE': 0.5, 'dropout_rateM': 0.6,
             'dropout_rateC': 0.8, 'weight_decay': 0.01,
             'dropout_rateClf': 0.6, 'gamma': 0.2, 'epochs': 20, 'margin': 0.5}

gemcitabine_tcga = {'drug': 'gemcitabine_tcga', 'mini_batch': 13, 'h_dim1': 16, 'h_dim2': 16, 'h_dim3': 16, 'lrE': 0.0001,
                    'lrM': 0.001, 'lrC': 0.01, 'lrCL': 0.05, 'dropout_rateE': 0.5, 'dropout_rateM': 0.5,
                    'dropout_rateC': 0.5, 'weight_decay': 0.001,
                    'dropout_rateClf': 0.5, 'gamma': 0.6, 'epochs': 10, 'margin': 2}


parameter = paclitaxel


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    data_path = Path('data/MOLI')
    expressions_path = data_path / 'exprs'
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'

    save_results_to = Path(f'./results/{parameter["drug"]}')
    save_results_to.mkdir(parents=True, exist_ok=True)

    GDSCE = pd.read_csv(expressions_homogenized_path / f'GDSC_exprs.{drug}.eb_with.PDX_exprs.{drug}.tsv',
                        sep="\t", index_col=0, decimal=',')
    GDSCE = pd.DataFrame.transpose(GDSCE)

    GDSCR = pd.read_csv(response_path / f'GDSC_response.{drug}.tsv',
                        sep="\t", index_col=0, decimal=',')

    PDXE = pd.read_csv(expressions_homogenized_path / f'PDX_exprs.{drug}.eb_with.GDSC_exprs.{drug}.tsv',
                       sep="\t", index_col=0, decimal=',')
    PDXE = pd.DataFrame.transpose(PDXE)

    PDXM = pd.read_csv(sna_binary_path / f'PDX_mutations.{drug}.tsv',
                       sep="\t", index_col=0, decimal='.')
    PDXM = pd.DataFrame.transpose(PDXM)

    PDXC = pd.read_csv(cna_binary_path / f'PDX_CNA.{drug}.tsv',
                       sep="\t", index_col=0, decimal='.')
    PDXC.drop_duplicates(keep='last')
    PDXC = pd.DataFrame.transpose(PDXC)

    GDSCM = pd.read_csv(sna_binary_path / f'GDSC_mutations.{drug}.tsv',
                        sep="\t", index_col=0, decimal='.')
    GDSCM = pd.DataFrame.transpose(GDSCM)

    GDSCC = pd.read_csv(cna_binary_path / f'GDSC_CNA.{drug}.tsv',
                        sep="\t", index_col=0, decimal='.')
    GDSCC.drop_duplicates(keep='last')
    GDSCC = pd.DataFrame.transpose(GDSCC)

    selector = VarianceThreshold(0.05)
    selector.fit_transform(GDSCE)
    GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

    PDXC = PDXC.fillna(0)
    PDXC[PDXC != 0.0] = 1
    PDXM = PDXM.fillna(0)
    PDXM[PDXM != 0.0] = 1
    GDSCM = GDSCM.fillna(0)
    GDSCM[GDSCM != 0.0] = 1
    GDSCC = GDSCC.fillna(0)
    GDSCC[GDSCC != 0.0] = 1

    ls = GDSCE.columns.intersection(GDSCM.columns)
    ls = ls.intersection(GDSCC.columns)
    ls = ls.intersection(PDXE.columns)
    ls = ls.intersection(PDXM.columns)
    ls = ls.intersection(PDXC.columns)
    ls2 = GDSCE.index.intersection(GDSCM.index)
    ls2 = ls2.intersection(GDSCC.index)
    ls = pd.unique(ls)

    GDSCE = GDSCE.loc[ls2, ls]
    GDSCM = GDSCM.loc[ls2, ls]
    GDSCC = GDSCC.loc[ls2, ls]

    GDSCR.loc[GDSCR.response == 'R'] = 0
    GDSCR.loc[GDSCR.response == 'S'] = 1

    y = GDSCR.response.to_numpy(dtype=np.int)

    skf = StratifiedKFold(n_splits=5)
    fold_max_aucs = []
    folds_aucs = []
    k = 0
    for train_index, test_index in skf.split(GDSCE.to_numpy(), y):
        k = k + 1
        X_trainE = GDSCE.values[train_index, :]
        X_testE = GDSCE.values[test_index, :]
        X_trainM = GDSCM.values[train_index, :]
        X_testM = GDSCM.values[test_index, :]
        X_trainC = GDSCC.values[train_index, :]
        X_testC = GDSCM.values[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]

        scalerGDSC = StandardScaler()
        X_trainE = scalerGDSC.fit_transform(X_trainE)
        X_testE = scalerGDSC.transform(X_testE)

        X_trainM = np.nan_to_num(X_trainM)
        X_trainC = np.nan_to_num(X_trainC)
        X_testM = np.nan_to_num(X_testM)
        X_testC = np.nan_to_num(X_testC)

        TX_testE = torch.FloatTensor(X_testE).to(device)
        TX_testM = torch.FloatTensor(X_testM).to(device)
        TX_testC = torch.FloatTensor(X_testC).to(device)
        ty_testE = torch.FloatTensor(y_test.astype(int)).to(device)

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

        trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                                  num_workers=8, sampler=sampler, pin_memory=True)

        n_sampE, IE_dim = X_trainE.shape
        n_sampM, IM_dim = X_trainM.shape

        Z_in = parameter.h_dim1 + parameter.h_dim2 + parameter.h_dim3

        costs_train = []
        aucs_train = []
        costs_test = []
        aucs_test = []

        random_negative_triplet_selector = RandomNegativeTripletSelector(marg)
        all_triplet_selector = AllTripletSelector()

        torch.cuda.manual_seed_all(42)
        AutoencoderE = encoder.Encoder(IE_dim, parameter['h_dim1'], parameter['dropout_rateE']).to(device)
        AutoencoderM = encoder.Encoder(IM_dim, parameter['h_dim2'], parameter['dropout_rateM']).to(device)
        AutoencoderC = encoder.Encoder(IM_dim, parameter['h_dim3'], parameter['dropout_rateC']).to(device)

        optimE = torch.optim.Adagrad(AutoencoderE.parameters(), lr=parameter['lrE'])
        optimM = torch.optim.Adagrad(AutoencoderM.parameters(), lr=parameter['lrM'])
        optimC = torch.optim.Adagrad(AutoencoderC.parameters(), lr=parameter['lrC'])

        trip_criterion = torch.nn.TripletMarginLoss(margin=parameter['marg'], p=2)

        Clas = encoder.Classifier(Z_in, parameter['dropout_rateClf']).to(device)
        optim_clas = torch.optim.Adagrad(Clas.parameters(), lr=parameter['lrCL'],
                                         weight_decay=parameter['weight_decay'])
        bce_loss = torch.nn.BCELoss()

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()
        for epoch in trange(parameter['epochs']):
            epoch_cost4 = 0
            epoch_cost3 = []
            num_minibatches = int(n_sampE / parameter['mb_size'])

            for (dataE, dataM, dataC, target) in trainLoader:

                for data in (dataE, dataM, dataC, target):
                    data.to(device)

                for optim in (optimE, optimM, optimC, optim_clas):
                    optim.zero_grad()

                AutoencoderE.train()
                AutoencoderM.train()
                AutoencoderC.train()
                Clas.train()
                if torch.mean(target) != 0. and torch.mean(target) != 1.:
                    # with autocast():
                    ZEX = AutoencoderE(dataE)
                    ZMX = AutoencoderM(dataM)
                    ZCX = AutoencoderC(dataC)

                    ZT = torch.cat((ZEX, ZMX, ZCX), 1)
                    ZT = F.normalize(ZT, p=2, dim=0)
                    y_pred = Clas(ZT)
                    Triplets = all_triplet_selector.get_triplets(ZT, target)
                    y_true = target.view(-1, 1).to(device)
                    loss = parameter['gamma'] * trip_criterion(ZT[Triplets[:, 0], :], ZT[Triplets[:, 1], :],
                                                               ZT[Triplets[:, 2], :]) \
                           + bce_loss(y_pred, y_true)

                    AUC = roc_auc_score(y_true.detach().cpu(), y_pred.detach().cpu())
                    # scaler.scale(loss).backward()
                    loss.backward()

                    for optim in (optim_clas, optimE, optimM, optimC):
                        optim.step()
                        # scaler.step(optim)

                    # Updates the scale for next iteration.
                    # scaler.update()

                    epoch_cost4 = epoch_cost4 + (loss / num_minibatches)
                    epoch_cost3.append(AUC)

            costs_train.append(torch.mean(epoch_cost4.cpu().detach()))
            aucs_train.append(np.mean(epoch_cost3))

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
                y_truet = ty_testE.view(-1, 1).to(device)
                TripletsT = all_triplet_selector.get_triplets(ZTT, ty_testE)
                lossT = parameter['gamma'] * trip_criterion(ZTT[TripletsT[:, 0], :], ZTT[TripletsT[:, 1], :],
                                                            ZTT[TripletsT[:, 2], :]) + bce_loss(PredT, y_truet)

                y_pred_test = PredT
                auc_test = roc_auc_score(y_truet.cpu().detach(), y_pred_test.cpu().detach())
                if auc_test > fold_max_auc:
                    fold_max_auc = auc_test

                costs_test.append(lossT.cpu().detach())
                aucs_test.append(auc_test)
        fold_max_aucs.append(fold_max_auc)
        folds_aucs.append(aucs_test)

        sns.set()
        plt.plot(costs_train, '-r', costs_test, '-b')
        plt.ylabel('Total cost')
        plt.xlabel('iterations (per tens)')

        filename = f'Cost_{parameter["drug"]}'

        plt.title('Cost')
        plt.savefig(str(save_results_to) + '/' + filename.replace(" ", "") + '.png', dpi=150)
        plt.close()

        plt.plot(aucs_train, '-r', aucs_test, '-b')
        plt.ylabel('AUC')
        plt.xlabel('iterations (per tens)')

        filename = f'auc_{parameter["drug"]}'

        plt.title('AUC')
        plt.savefig(str(save_results_to) + '/' + filename.replace(" ", "") + '.png', dpi=150)
        plt.close()

if __name__ == "__main__":
    # execute only if run as a script
    main()
