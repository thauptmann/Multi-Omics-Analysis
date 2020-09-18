from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import pandas as pd
import sklearn.preprocessing as sk
from sklearn.feature_selection import VarianceThreshold
from siamese_triplet.utils import AllTripletSelector,RandomNegativeTripletSelector
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from torch import nn

def main():
    # reproducibility
    torch.manual_seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    data_path = Path('data/')
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'
    expressions_path = data_path / 'exprs'
    egfr_path = data_path / 'EGFR_experiments_data'

    GDSCE = pd.read_csv(egfr_path / "GDSC_exprs.z.EGFRi.tsv", sep="\t", index_col=0, decimal=",")
    GDSCE = pd.DataFrame.transpose(GDSCE)

    GDSCM = pd.read_csv(egfr_path / "GDSC_mutations.EGFRi.tsv", sep="\t", index_col=0, decimal=".")
    GDSCM = pd.DataFrame.transpose(GDSCM)
    GDSCM = GDSCM.loc[:, ~GDSCM.columns.duplicated()]

    GDSCC = pd.read_csv(egfr_path / "GDSC_CNA.EGFRi.tsv", sep="\t", index_col=0, decimal=".")
    GDSCC = GDSCC.drop_duplicates(keep='last')
    GDSCC = pd.DataFrame.transpose(GDSCC)
    GDSCC = GDSCC.loc[:, ~GDSCC.columns.duplicated()]

    PDXEerlo = pd.read_csv(egfr_path / "PDX_exprs.Erlotinib.eb_with.GDSC_exprs.Erlotinib.tsv",
                           sep="\t", index_col=0, decimal=",")
    PDXEerlo = pd.DataFrame.transpose(PDXEerlo)

    PDXMerlo = pd.read_csv(egfr_path / "PDX_mutations.Erlotinib.tsv", sep="\t", index_col=0, decimal=",")
    PDXMerlo = pd.DataFrame.transpose(PDXMerlo)

    PDXCerlo = pd.read_csv(egfr_path / "PDX_CNV.Erlotinib.tsv", sep="\t", index_col=0, decimal=",")
    PDXCerlo.drop_duplicates(keep='last')
    PDXCerlo = pd.DataFrame.transpose(PDXCerlo)
    PDXCerlo = PDXCerlo.loc[:, ~PDXCerlo.columns.duplicated()]

    PDXEcet = pd.read_csv(egfr_path / "PDX_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
    PDXEcet = pd.DataFrame.transpose(PDXEcet)

    PDXMcet = pd.read_csv(egfr_path / "PDX_mutations.Cetuximab.tsv", sep="\t", index_col=0, decimal=",")
    PDXMcet = pd.DataFrame.transpose(PDXMcet)

    PDXCcet = pd.read_csv(egfr_path / "PDX_CNV.Cetuximab.tsv", sep="\t", index_col=0, decimal=",")
    PDXCcet = PDXCcet.drop_duplicates(keep='last')
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

    GDSCR = pd.read_csv(egfr_path / "GDSC_response.EGFRi.tsv",
                        sep="\t", index_col=0, decimal=",")
    PDXRcet = pd.read_csv(egfr_path / "PDX_response.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
    PDXRerlo = pd.read_csv(egfr_path / "PDX_response.Erlotinib.tsv",
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

    mini_batch = 16
    h_dim1 = 32
    h_dim2 = 16
    h_dim3 = 256
    lr_e = 0.001
    lr_m = 0.0001
    lr_c = 5.00E-05
    lr_cl = 0.005
    dropout_rate_e = 0.5
    dropout_rate_m = 0.8
    dropout_rate_c = 0.5
    weight_decay = 0.0001
    dropout_rate_clf = 0.3
    gamma = 0.5
    epochs = 20
    margin = 1.5

    y_train = GDSCRv2.response.values.astype(int)
    x_train_e = GDSCEv2.values
    x_train_m = GDSCMv2.values
    x_train_c = GDSCCv2.values


    # Train
    x_test_eerlo = PDXEerlo.values
    x_test_merlo = torch.FloatTensor(PDXMerlo.values)
    x_test_cerlo = torch.FloatTensor(PDXCerlo.values)
    ytserlo = PDXRerlo['response'].values

    x_test_ecet = PDXEcet.values
    x_test_mcet = torch.FloatTensor(PDXMcet.values)
    x_test_ccet = torch.FloatTensor(PDXCcet.values)
    ytscet = PDXRcet['response'].values

    ytscet = torch.FloatTensor(ytscet.astype(int))
    ytserlo = torch.FloatTensor(ytserlo.astype(int))

    scaler_gdsc = sk.StandardScaler()
    scaler_gdsc.fit(x_train_e)
    x_train_e = torch.FloatTensor(scaler_gdsc.transform(x_train_e))
    x_test_ecet = torch.FloatTensor(scaler_gdsc.transform(x_test_ecet))
    x_test_eerlo = torch.FloatTensor(scaler_gdsc.transform(x_test_eerlo))

    x_train_m = torch.FloatTensor(np.nan_to_num(x_train_m))
    x_train_c = torch.FloatTensor(np.nan_to_num(x_train_c))

    N_samp_e, ie_dim = x_train_e.shape
    _, im_dim = x_train_m.shape
    _, ic_dim = x_train_m.shape
    z_in = h_dim1 + h_dim2 + h_dim3

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
                nn.Linear(z_in, 1),
                nn.Dropout(dropout_rate_clf),
                nn.Sigmoid())

        def forward(self, x):
            return self.FC(x)

    autoencoder_e = AEE().to(device)
    autoencoder_m = AEM().to(device)
    autoencoder_c = AEC().to(device)
    clas = Classifier().to(device)

    optim_e = torch.optim.Adagrad(autoencoder_e.parameters(), lr=lr_e)
    optim_m = torch.optim.Adagrad(autoencoder_m.parameters(), lr=lr_m)
    optim_c = torch.optim.Adagrad(autoencoder_c.parameters(), lr=lr_c)
    optim_clas = torch.optim.Adagrad(clas.parameters(), lr=lr_cl,
                                     weight_decay=weight_decay)

    trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    TripSel = OnlineTriplet(margin, triplet_selector)
    TripSel2 = OnlineTestTriplet(margin, triplet_selector2)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)
    train_dataset = torch.utils.data.TensorDataset(x_train_e, x_train_m,
                                                   x_train_c,
                                                   torch.FloatTensor(y_train))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=mini_batch,
                                               shuffle=False,
                                               num_workers=8, sampler=sampler, pin_memory=True, drop_last=True)

    test_dataset_erlo = torch.utils.data.TensorDataset(x_test_eerlo, x_test_merlo,
                                                       x_test_cerlo, torch.FloatTensor(ytserlo))
    test_loader_erlo = torch.utils.data.DataLoader(dataset=test_dataset_erlo, batch_size=mini_batch,
                                                   shuffle=False, num_workers=8, pin_memory=True)

    test_dataset_cet = torch.utils.data.TensorDataset(x_test_ecet, x_test_mcet,
                                                      x_test_ccet, torch.FloatTensor(ytscet))
    test_loader_cet = torch.utils.data.DataLoader(dataset=test_dataset_cet, batch_size=mini_batch, shuffle=False,
                                                  num_workers=8, pin_memory=True)

    auc_train = []
    for _ in trange(epochs):
        train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e,
                                      optim_m, optim_c, optim_clas, TripSel2, trip_criterion,
                                      bce_loss, device, auc_train, gamma)

    auc_test_erlo = validate(test_loader_erlo, autoencoder_e, autoencoder_m, autoencoder_c, clas, device)
    auc_test_cet = validate(test_loader_cet, autoencoder_e, autoencoder_m, autoencoder_c, clas, device)

    print(f'EGFR: AUROC Train = {auc_train[-1]}')
    print(f'EGFR Cetuximab: AUROC = {auc_test_cet}')
    print(f'EGFR Erlotinib: AUROC = {auc_test_erlo}')


def train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e, optim_m, optim_c, optim_clas,
          TripSel2, trip_criterion, C_loss, device, auc_train, gamma):
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
        auc_train.append(auc)
    return auc_train


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
    main()
