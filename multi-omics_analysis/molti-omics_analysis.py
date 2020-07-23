import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.cuda.amp
import seaborn as sns
import encoder
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from siamese_triplet.metrics import AverageNonzeroTripletsMetric
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import random
from sklearn.model_selection import StratifiedKFold


def main():
    save_results_to = '.'
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

    GDSCE = pd.read_csv(expressions_homogenized_path / 'GDSC_exprs.Cetuximab.eb_with.PDX_exprs.Cetuximab.tsv',
                        sep="\t", index_col=0, decimal=',')
    GDSCE = pd.DataFrame.transpose(GDSCE)

    GDSCR = pd.read_csv(response_path / 'GDSC_response.Cetuximab.tsv',
                        sep="\t", index_col=0, decimal=',')

    PDXE = pd.read_csv(expressions_homogenized_path / 'PDX_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv',
                       sep="\t", index_col=0, decimal=',')
    PDXE = pd.DataFrame.transpose(PDXE)

    PDXM = pd.read_csv(sna_binary_path / 'PDX_mutations.Cetuximab.tsv',
                       sep="\t", index_col=0, decimal='.')
    PDXM = pd.DataFrame.transpose(PDXM)

    PDXC = pd.read_csv(cna_binary_path / 'PDX_CNA.Cetuximab.tsv',
                       sep="\t", index_col=0, decimal='.')
    PDXC.drop_duplicates(keep='last')
    PDXC = pd.DataFrame.transpose(PDXC)

    GDSCM = pd.read_csv(sna_binary_path /'GDSC_mutations.Cetuximab.tsv',
                        sep="\t", index_col=0, decimal='.')
    GDSCM = pd.DataFrame.transpose(GDSCM)

    GDSCC = pd.read_csv(cna_binary_path / 'GDSC_CNA.Cetuximab.tsv',
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
    # GDSCR = GDSCR.loc[ls2, :]

    ls_h_dim = [1024, 512, 256, 128, 64]
    ls_marg = [0.5, 1, 1.5, 2, 2.5]
    ls_lr = [0.0005, 0.0001, 0.005, 0.001]
    ls_epoch = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]
    ls_rate = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ls_wd = [0.01, 0.001, 0.1, 0.0001]
    ls_lam = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    Y = GDSCR.response.values

    skf = StratifiedKFold(n_splits=5, random_state=42)

    max_iter = 100
    for iters in range(max_iter):
        k = 0
        mb_size = 30
        h_dim1 = random.choice(ls_h_dim)
        h_dim2 = random.choice(ls_h_dim)
        h_dim3 = random.choice(ls_h_dim)
        marg = random.choice(ls_marg)
        lrE = random.choice(ls_lr)
        lrM = random.choice(ls_lr)
        lrC = random.choice(ls_lr)
        lrCL = random.choice(ls_lr)
        epochs = random.choice(ls_epoch)
        rateE = random.choice(ls_rate)
        rateM = random.choice(ls_rate)
        rateC = random.choice(ls_rate)
        rateClf = random.choice(ls_rate)
        wd = random.choice(ls_wd)
        lam = random.choice(ls_lam)

        for train_index, test_index in skf.split(GDSCE.values, Y):
            k = k + 1
            X_trainE = GDSCE.values[train_index, :]
            X_testE = GDSCE.values[test_index, :]
            X_trainM = GDSCM.values[train_index, :]
            X_testM = GDSCM.values[test_index, :]
            X_trainC = GDSCC.values[train_index, :]
            X_testC = GDSCM.values[test_index, :]
            y_trainE = Y[train_index]
            y_testE = Y[test_index]

            scalerGDSC = sk.StandardScaler()
            scalerGDSC.fit(X_trainE)
            X_trainE = scalerGDSC.transform(X_trainE)
            X_testE = scalerGDSC.transform(X_testE)

            X_trainM = np.nan_to_num(X_trainM)
            X_trainC = np.nan_to_num(X_trainC)
            X_testM = np.nan_to_num(X_testM)
            X_testC = np.nan_to_num(X_testC)

            TX_testE = torch.FloatTensor(X_testE)
            TX_testM = torch.FloatTensor(X_testM)
            TX_testC = torch.FloatTensor(X_testC)
            ty_testE = torch.FloatTensor(y_testE.astype(int))

            # Train
            class_sample_count = np.array([len(np.where(y_trainE == t)[0]) for t in np.unique(y_trainE)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in y_trainE])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                            replacement=True)

            trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                          torch.FloatTensor(X_trainC),
                                                          torch.FloatTensor(y_trainE.astype(int)))

            trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                                      num_workers=1, sampler=sampler)

            n_sampE, IE_dim = X_trainE.shape
            n_sampM, IM_dim = X_trainM.shape

            Z_in = h_dim1 + h_dim2 + h_dim3


            costtr = []
            auctr = []
            costts = []
            aucts = []

            random_negative_triplet_selector = RandomNegativeTripletSelector(marg)
            all_triplet_selector = AllTripletSelector()

            class OnlineTriplet(nn.Module):
                def __init__(self, marg, triplet_selector):
                    super(OnlineTriplet, self).__init__()
                    self.marg = marg
                    self.triplet_selector = triplet_selector

                def forward(self, embeddings, target):
                    triplets = self.triplet_selector.get_triplets(embeddings, target)
                    return triplets

            torch.cuda.manual_seed_all(42)

            AutoencoderE = encoder.Encoder(IE_dim, h_dim1, rateE).to(device)
            AutoencoderM = encoder.Encoder(IM_dim, h_dim2, rateM).to(device)
            AutoencoderC = encoder.Encoder(IM_dim, h_dim3, rateC).to(device)

            optimE = torch.optim.Adagrad(AutoencoderE.parameters(), lr=lrE)
            optimM = torch.optim.Adagrad(AutoencoderM.parameters(), lr=lrM)
            optimC = torch.optim.Adagrad(AutoencoderC.parameters(), lr=lrC)

            trip_criterion = torch.nn.TripletMarginLoss(margin=marg, p=2)
            TripSel = OnlineTriplet(marg, random_negative_triplet_selector)
            TripSel2 = OnlineTriplet(marg, all_triplet_selector)

            Clas = encoder.Classifier(Z_in, rateClf).to(device)
            optim_clas = torch.optim.Adagrad(Clas.parameters(), lr=lrCL, weight_decay=wd)
            C_loss = torch.nn.BCELoss()

            # Creates a GradScaler once at the beginning of training.
            scaler = GradScaler()

            for epoch in range(epochs):

                epoch_cost4 = 0
                epoch_cost3 = []
                num_minibatches = int(n_sampE / mb_size)

                for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                    optimE.zero_grad()
                    optimM.zero_grad()
                    optimC.zero_grad()
                    optim_clas.zero_grad()

                    flag = 0
                    AutoencoderE.train()
                    AutoencoderM.train()
                    AutoencoderC.train()
                    Clas.train()

                    if torch.mean(target) != 0. and torch.mean(target) != 1.:
                        with autocast():
                            ZEX = AutoencoderE(dataE)
                            ZMX = AutoencoderM(dataM)
                            ZCX = AutoencoderC(dataC)

                            ZT = torch.cat((ZEX, ZMX, ZCX), 1)
                            ZT = F.normalize(ZT, p=2, dim=0)
                            Pred = Clas(ZT)

                            Triplets = TripSel2(ZT, target)
                            loss = lam * trip_criterion(ZT[Triplets[:, 0], :], ZT[Triplets[:, 1], :],
                                                    ZT[Triplets[:, 2], :]) + C_loss(Pred, target.view(-1, 1))

                        y_true = target.view(-1, 1)
                        y_pred = Pred
                        AUC = roc_auc_score(y_true.detach().numpy(), y_pred.detach().numpy())

                        scaler.scale(loss).backward()

                        scaler.step(optimE)
                        scaler.step(optimM)
                        scaler.step(optimC)
                        scaler.step(optim_clas)

                        epoch_cost4 = epoch_cost4 + (loss / num_minibatches)
                        epoch_cost3.append(AUC)
                        flag = 1

                if flag == 1:
                    costtr.append(torch.mean(epoch_cost4))
                    auctr.append(np.mean(epoch_cost3))
                    print('Iter-{}; Total loss: {:.4}'.format(epoch, loss))

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

                    TripletsT = TripSel2(ZTT, ty_testE)
                    lossT = lam * trip_criterion(ZTT[TripletsT[:, 0], :], ZTT[TripletsT[:, 1], :],
                                                 ZTT[TripletsT[:, 2], :]) + C_loss(PredT, ty_testE.view(-1, 1))

                    y_truet = ty_testE.view(-1, 1)
                    y_predt = PredT
                    AUCt = roc_auc_score(y_truet.detach().numpy(), y_predt.detach().numpy())

                    costts.append(lossT)
                    aucts.append(AUCt)

            sns.set()
            plt.plot(np.squeeze(costtr), '-r', np.squeeze(costts), '-b')
            plt.ylabel('Total cost')
            plt.xlabel('iterations (per tens)')

            title = 'Cost Cetuximab iter = {}, fold = {}, mb_size = {},  h_dim[1,2,3] = ({},{},{}), marg = {}, lr[E,M,C] = ({}, {}, {}), epoch = {}, rate[1,2,3,4] = ({},{},{},{}), wd = {}, lrCL = {}, lam = {}'. \
                format(iters, k, mbs, hdm1, hdm2, hdm3, mrg, lre, lrm, lrc, epch, rateE, rateM, rateC, rateClf, wd, lrCL,
                       lam)

            plt.suptitle(title)
            plt.savefig(save_results_to + title + '.png', dpi=150)
            plt.close()

            plt.plot(np.squeeze(auctr), '-r', np.squeeze(aucts), '-b')
            plt.ylabel('AUC')
            plt.xlabel('iterations (per tens)')

            title = 'AUC Cetuximab iter = {}, fold = {}, mb_size = {},  h_dim[1,2,3] = ({},{},{}), marg = {}, lr[E,M,C] = ({}, {}, {}), epoch = {}, rate[1,2,3,4] = ({},{},{},{}), wd = {}, lrCL = {}, lam = {}'. \
                format(iters, k, mbs, hdm1, hdm2, hdm3, mrg, lre, lrm, lrc, epch, rateE, rateM, rateC, rateClf, wd, lrCL,
                       lam)

            plt.suptitle(title)
            plt.savefig(save_results_to + title + '.png', dpi=150)
            plt.close()


if __name__ == "__main__":
    # execute only if run as a script
    main()
