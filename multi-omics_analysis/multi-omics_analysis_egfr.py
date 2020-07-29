from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import encoder
import pandas as pd
import sklearn.preprocessing as sk
from sklearn.feature_selection import VarianceThreshold
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from siamese_triplet.metrics import AverageNonzeroTripletsMetric
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score


def main():
    # reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    data_path = Path('data/MOLI')
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'
    expressions_path = data_path / 'exprs'

    GDSCE = pd.read_csv(expressions_path / "GDSC_exprs.z.EGFRi.tsv", sep="\t", index_col=0, decimal=",")
    GDSCE = pd.DataFrame.transpose(GDSCE)

    GDSCM = pd.read_csv(sna_binary_path / "GDSC_mutations.EGFRi.tsv", sep="\t", index_col=0, decimal=".")
    GDSCM = pd.DataFrame.transpose(GDSCM)
    GDSCM = GDSCM.loc[:, ~GDSCM.columns.duplicated()]

    GDSCC = pd.read_csv(cna_binary_path / "GDSC_CNA.EGFRi.tsv", sep="\t", index_col=0, decimal=".")
    GDSCC.drop_duplicates(keep='last')
    GDSCC = pd.DataFrame.transpose(GDSCC)
    GDSCC = GDSCC.loc[:, ~GDSCC.columns.duplicated()]

    PDXEerlo = pd.read_csv(expressions_homogenized_path / "PDX_exprs.Erlotinib.eb_with.GDSC_exprs.Erlotinib.tsv",
                           sep="\t", index_col=0, decimal=",")
    PDXEerlo = pd.DataFrame.transpose(PDXEerlo)

    PDXMerlo = pd.read_csv(sna_binary_path / "PDX_mutations.Erlotinib.tsv", sep="\t", index_col=0, decimal=",")
    PDXMerlo = pd.DataFrame.transpose(PDXMerlo)

    PDXCerlo = pd.read_csv(cna_binary_path / "PDX_CNA.Erlotinib.tsv", sep="\t", index_col=0, decimal=",")
    PDXCerlo.drop_duplicates(keep='last')
    PDXCerlo = pd.DataFrame.transpose(PDXCerlo)
    PDXCerlo = PDXCerlo.loc[:, ~PDXCerlo.columns.duplicated()]

    PDXEcet = pd.read_csv(expressions_homogenized_path / "PDX_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
    PDXEcet = pd.DataFrame.transpose(PDXEcet)

    PDXMcet = pd.read_csv(sna_binary_path / "PDX_mutations.Cetuximab.tsv", sep="\t", index_col=0, decimal=",")
    PDXMcet = pd.DataFrame.transpose(PDXMcet)

    PDXCcet = pd.read_csv(cna_binary_path / "PDX_CNA.Cetuximab.tsv", sep="\t", index_col=0, decimal=",")
    PDXCcet.drop_duplicates(keep='last')
    PDXCcet = pd.DataFrame.transpose(PDXCcet)
    PDXCcet = PDXCcet.loc[:, ~PDXCcet.columns.duplicated()]

    selector = VarianceThreshold(0.05)
    selector.fit_transform(GDSCE)
    GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

    GDSCM = GDSCM.fillna(0)
    GDSCM[GDSCM != 0.0] = 1
    GDSCC = GDSCC.fillna(0)
    GDSCC[GDSCC != 0.0] = 1

    PDXMcet = PDXMcet.fillna(0)
    PDXMcet[PDXMcet != 0.0] = 1
    PDXCcet = PDXCcet.fillna(0)
    PDXCcet[PDXCcet != 0.0] = 1

    PDXMerlo = PDXMerlo.fillna(0)
    PDXMerlo[PDXMerlo != 0.0] = 1
    PDXCerlo = PDXCerlo.fillna(0)
    PDXCerlo[PDXCerlo != 0.0] = 1

    ls = GDSCE.columns.intersection(GDSCM.columns)
    ls = ls.intersection(GDSCC.columns)
    ls = ls.intersection(PDXEerlo.columns)
    ls = ls.intersection(PDXMerlo.columns)
    ls = ls.intersection(PDXCerlo.columns)
    ls = ls.intersection(PDXEcet.columns)
    ls = ls.intersection(PDXMcet.columns)
    ls = ls.intersection(PDXCcet.columns)
    ls3 = PDXEerlo.index.intersection(PDXMerlo.index)
    ls3 = ls3.intersection(PDXCerlo.index)
    ls4 = PDXEcet.index.intersection(PDXMcet.index)
    ls4 = ls4.intersection(PDXCcet.index)
    ls = pd.unique(ls)

    PDXEerlo = PDXEerlo.loc[ls3, ls]
    PDXMerlo = PDXMerlo.loc[ls3, ls]
    PDXCerlo = PDXCerlo.loc[ls3, ls]
    PDXEcet = PDXEcet.loc[ls4, ls]
    PDXMcet = PDXMcet.loc[ls4, ls]
    PDXCcet = PDXCcet.loc[ls4, ls]
    GDSCE = GDSCE.loc[:, ls]
    GDSCM = GDSCM.loc[:, ls]
    GDSCC = GDSCC.loc[:, ls]

    GDSCR = pd.read_csv(response_path / "GDSC_response.EGFRi.tsv",
                        sep="\t", index_col=0, decimal=",")
    PDXRcet = pd.read_csv(response_path / "PDX_response.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
    PDXRerlo = pd.read_csv(response_path / "PDX_response.Erlotinib.tsv",
                           sep="\t", index_col=0, decimal=",")

    PDXRcet = PDXRcet.loc[ls4, :]
    PDXRerlo = PDXRerlo.loc[ls3, :]

    GDSCR.rename(mapper=str, axis='index', inplace=True)

    d = {"R": 0, "S": 1}
    GDSCR["response"] = GDSCR.loc[:, "response"].apply(lambda x: d[x])
    PDXRcet["response"] = PDXRcet.loc[:, "response"].apply(lambda x: d[x])
    PDXRerlo["response"] = PDXRerlo.loc[:, "response"].apply(lambda x: d[x])

    responses = GDSCR
    drugs = set(responses["drug"].values)
    exprs_z = GDSCE
    cna = GDSCC
    mut = GDSCM
    expression_zscores = []
    CNA = []
    mutations = []
    for drug in drugs:
        samples = responses.loc[responses["drug"] == drug, :].index.values
        e_z = exprs_z.loc[samples, :]
        c = cna.loc[samples, :]
        m = mut.loc[samples, :]
        m = mut.loc[samples, :]
        # next 3 rows if you want non-unique sample names
        e_z.rename(lambda x: str(x) + "_" + drug, axis="index", inplace=True)
        c.rename(lambda x: str(x) + "_" + drug, axis="index", inplace=True)
        m.rename(lambda x: str(x) + "_" + drug, axis="index", inplace=True)
        expression_zscores.append(e_z)
        CNA.append(c)
        mutations.append(m)
    responses.index = responses.index.values + "_" + responses["drug"].values
    GDSCEv2 = pd.concat(expression_zscores, axis=0)
    GDSCCv2 = pd.concat(CNA, axis=0)
    GDSCMv2 = pd.concat(mutations, axis=0)
    GDSCRv2 = responses

    ls2 = GDSCEv2.index.intersection(GDSCMv2.index)
    ls2 = ls2.intersection(GDSCCv2.index)
    GDSCEv2 = GDSCEv2.loc[ls2, :]
    GDSCMv2 = GDSCMv2.loc[ls2, :]
    GDSCCv2 = GDSCCv2.loc[ls2, :]
    GDSCRv2 = GDSCRv2.loc[ls2, :]

    mbs = 16
    hdm1 = 32
    hdm2 = 16
    hdm3 = 256
    margin = 1.5
    lre = 0.001
    lrm = 0.0001
    lrc = 0.00005
    lrCL = 0.005
    epochs = 40
    rate1 = 0.5
    rate2 = 0.8
    rate3 = 0.5
    rate4 = 0.3
    wd = 0.0001
    gamma = 0.5

    X_trainE = GDSCEv2.values
    X_trainM = GDSCMv2.values
    X_trainC = GDSCCv2.values
    Y = GDSCRv2['response'].values

    X_testEerlo = PDXEerlo.values
    X_testMerlo = PDXMerlo.values
    X_testCerlo = PDXCerlo.values
    Ytserlo = PDXRerlo['response'].values

    X_testEcet = PDXEcet.values
    X_testMcet = PDXMcet.values
    X_testCcet = PDXCcet.values
    Ytscet = PDXRcet['response'].values

    scalerGDSC = sk.StandardScaler()
    scalerGDSC.fit(X_trainE)
    X_trainE = scalerGDSC.transform(X_trainE)
    X_testEcet = scalerGDSC.transform(X_testEcet)
    X_testEerlo = scalerGDSC.transform(X_testEerlo)

    X_trainM = np.nan_to_num(X_trainM)
    X_trainC = np.nan_to_num(X_trainC)

    X_testEcet = torch.FloatTensor(X_testEcet)
    X_testMcet = torch.FloatTensor(X_testMcet)
    X_testCcet = torch.FloatTensor(X_testCcet)
    Ytscet = torch.FloatTensor(Ytscet.astype(int))

    X_testEerlo = torch.FloatTensor(X_testEerlo)
    X_testMerlo = torch.FloatTensor(X_testMerlo)
    X_testCerlo = torch.FloatTensor(X_testCerlo)
    Ytserlo = torch.FloatTensor(Ytserlo.astype(int))

    # Train
    class_sample_count = np.array([len(np.where(Y == t)[0]) for t in np.unique(Y)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in Y])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)

    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                  torch.FloatTensor(X_trainC),
                                                  torch.FloatTensor(Y))

    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mbs,
                                              shuffle=False,
                                              num_workers=8, sampler=sampler, pin_memory=True)

    n_sampE, IE_dim = X_trainE.shape
    n_sampM, IM_dim = X_trainM.shape
    n_sampC, IC_dim = X_trainC.shape

    Z_in = hdm1 + hdm2 + hdm3

    random_negative_triplet_selector = RandomNegativeTripletSelector(margin)
    all_triplet_selector = AllTripletSelector()

    AutoencoderE = encoder.Encoder(IE_dim, hdm1, rate1).to(device)
    AutoencoderM = encoder.Encoder(IM_dim, hdm2, rate2).to(device)
    AutoencoderC = encoder.Encoder(IC_dim, hdm3, rate3).to(device)

    optimE = torch.optim.Adagrad(AutoencoderE.parameters(), lr=lre)
    optimM = torch.optim.Adagrad(AutoencoderM.parameters(), lr=lrm)
    optimC = torch.optim.Adagrad(AutoencoderC.parameters(), lr=lrc)

    trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

    Clas = encoder.Classifier(Z_in, rate4).to(device)
    optim_clas = torch.optim.Adagrad(Clas.parameters(), lr=lrCL,
                                     weight_decay=wd)
    bce_loss = torch.nn.BCELoss()
    for epoch in trange(epochs):
        # train
        train(trainLoader, AutoencoderE, AutoencoderM, AutoencoderC, Clas, optimE, optimM, optimC, optim_clas,
              all_triplet_selector, trip_criterion, bce_loss, device, gamma)

    # test
    auc_test_cet = validate(AutoencoderE, AutoencoderM, AutoencoderC, Clas, X_testEcet, X_testMcet, X_testCcet, Ytscet,
                            device)
    print(f'EGFR Cetuximab: AUC = {auc_test_cet}')

    auc_test_erlo = validate(AutoencoderE, AutoencoderM, AutoencoderC, Clas, X_testEerlo, X_testMerlo, X_testCerlo,
                             Ytserlo, device)
    print(f'EGFR Erlotinib: AUC = {auc_test_erlo}')


def train(trainLoader, AutoencoderE, AutoencoderM, AutoencoderC, Clas, optimE, optimM, optimC, optim_clas,
          all_triplet_selector, trip_criterion, bce_with_logits_loss, device, gamma):
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
            loss = gamma * trip_criterion(ZT[Triplets[:, 0], :], ZT[Triplets[:, 1], :],
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


if __name__ == "__main__":
    main()
