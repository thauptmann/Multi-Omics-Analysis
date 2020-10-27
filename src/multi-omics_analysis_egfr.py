from pathlib import Path
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import moli
import utils
import pandas as pd
import sklearn.preprocessing as sk
from sklearn.feature_selection import VarianceThreshold
from siamese_triplet.utils import AllTripletSelector,RandomNegativeTripletSelector
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import autocast
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler

mini_batch_list = [8, 16, 32, 64]
dim_list = [1024, 512, 256, 128, 64, 32, 16]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
epoch_list = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]
drop_rate_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.01, 0.001, 0.1, 0.0001]
gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
max_iter = 10

def main():
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    Y = GDSCRv2['response'].values

    skf = StratifiedKFold(n_splits=5)
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
        weight_decay = random.choice(weight_decay_list)
        dropout_rate_clf = random.choice(drop_rate_list)
        gamma = random.choice(gamma_list)
        epochs = random.choice(epoch_list)
        margin = random.choice(margin_list)

        aucs_validate = []
        for train_index, test_index in skf.split(GDSCEv2.to_numpy(), Y):
            x_train_e = GDSCEv2.values[train_index, :]
            x_train_m = GDSCMv2.values[train_index, :]
            x_train_c = GDSCCv2.values[train_index, :]

            x_test_e = GDSCEv2.values[test_index, :]
            x_test_m = GDSCMv2.values[test_index, :]
            x_test_c = GDSCCv2.values[test_index, :]

            y_train = Y[train_index]
            y_test = Y[test_index]

            scaler_gdsc = sk.StandardScaler()
            x_train_e = scaler_gdsc.fit_transform(x_train_e)
            x_test_e = scaler_gdsc.transform(x_test_e)
            x_train_m = np.nan_to_num(x_train_m)
            x_train_c = np.nan_to_num(x_train_c)

            # Initialisation
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

            test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e), torch.FloatTensor(x_test_m),
                                                          torch.FloatTensor(x_test_c),
                                                          torch.FloatTensor(y_test))
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=mini_batch,
                                                      shuffle=False, num_workers=8, pin_memory=True)

            n_samp_e, IE_dim = x_train_e.shape
            _, IM_dim = x_train_m.shape
            _, IC_dim = x_train_c.shape

            z_in = h_dim1 + h_dim2 + h_dim3

            random_negative_triplet_selector = RandomNegativeTripletSelector(margin)
            all_triplet_selector = AllTripletSelector()

            autoencoder_e = moli.Encoder(IE_dim, h_dim1, dropout_rate_e).to(device)
            autoencoder_m = moli.Encoder(IM_dim, h_dim2, dropout_rate_m).to(device)
            autoencoder_c = moli.Encoder(IC_dim, h_dim3, dropout_rate_c).to(device)

            optim_e = torch.optim.Adagrad(autoencoder_e.parameters(), lr=lr_e)
            optim_m = torch.optim.Adagrad(autoencoder_m.parameters(), lr=lr_m)
            optim_c = torch.optim.Adagrad(autoencoder_c.parameters(), lr=lr_c)

            trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

            clas = moli.Classifier(z_in, dropout_rate_clf).to(device)
            optim_clas = torch.optim.Adagrad(clas.parameters(), lr=lr_cl, weight_decay=weight_decay)
            bce_loss = torch.nn.BCEWithLogitsLoss()

            for epoch in range(epochs):
                cost_train = 0
                auc_train = []
                num_minibatches = int(n_samp_e / mini_batch)
                auc_train, cost_train = train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e,
                                              optim_m, optim_c, optim_clas,
                                              all_triplet_selector, trip_criterion, bce_loss, device, cost_train,
                                              num_minibatches,
                                              auc_train, gamma)

                # validate
                auc_validate = test(test_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas,
                                                   device)
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

    print(f'Best validation AUROC: {best_auc}')

    # Test
    x_test_eerlo = PDXEerlo.values
    x_test_merlo = torch.FloatTensor(PDXMerlo.values)
    x_test_cerlo = torch.FloatTensor(PDXCerlo.values)
    ytserlo = PDXRerlo['response'].values

    x_test_ecet = PDXEcet.values
    x_test_mcet = torch.FloatTensor(PDXMcet.values)
    x_test_ccet = torch.FloatTensor(PDXCcet.values)
    ytscet = PDXRcet['response'].values

    x_test_ecet = scaler_gdsc.transform(x_test_ecet)
    x_test_ecet = torch.FloatTensor(x_test_ecet)
    x_test_eerlo = scaler_gdsc.transform(x_test_eerlo)
    x_test_eerlo = torch.FloatTensor(x_test_eerlo)

    ytscet = torch.FloatTensor(ytscet.astype(int))
    ytserlo = torch.FloatTensor(ytserlo.astype(int))

    _, ie_dim = x_train_e.shape
    _, im_dim = x_train_m.shape
    _, ic_dim = x_test_m.shape
    z_in = best_h_dim1 + best_h_dim2 + best_h_dim3

    random_negative_triplet_selector = RandomNegativeTripletSelector(best_margin)
    all_triplet_selector = AllTripletSelector()

    autoencoder_e = moli.Encoder(ie_dim, best_h_dim1, best_dropout_rate_e).to(device)
    autoencoder_m = moli.Encoder(im_dim, best_h_dim2, best_dropout_rate_m).to(device)
    autoencoder_c = moli.Encoder(ic_dim, best_h_dim3, best_dropout_rate_c).to(device)
    clas = moli.Classifier(z_in, best_dropout_rate_clf).to(device)

    optim_e = torch.optim.Adagrad(autoencoder_e.parameters(), lr=best_lr_e)
    optim_m = torch.optim.Adagrad(autoencoder_m.parameters(), lr=best_lr_m)
    optim_c = torch.optim.Adagrad(autoencoder_c.parameters(), lr=best_lr_c)
    optim_clas = torch.optim.Adagrad(clas.parameters(), lr=best_lr_cl,
                                     weight_decay=best_weight_decay)

    trip_criterion = torch.nn.TripletMarginLoss(margin=best_margin, p=2)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    x_train_e = GDSCEv2.values
    x_train_m = GDSCMv2.values
    x_train_c = GDSCCv2.values

    y_train = Y

    scaler_gdsc = sk.StandardScaler()
    scaler_gdsc.fit(x_train_e)
    x_train_e = scaler_gdsc.transform(x_train_e)
    x_train_m = np.nan_to_num(x_train_m)
    x_train_c = np.nan_to_num(x_train_c)
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

    test_dataset_erlo = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_eerlo), torch.FloatTensor(x_test_merlo),
                                                       torch.FloatTensor(x_test_cerlo), torch.FloatTensor(ytserlo))
    test_loader_erlo = torch.utils.data.DataLoader(dataset=test_dataset_erlo, batch_size=mini_batch,
                                                   shuffle=False, num_workers=8, pin_memory=True)

    test_dataset_cet = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_ecet), torch.FloatTensor(x_test_mcet),
                                                      torch.FloatTensor(x_test_ccet), torch.FloatTensor(ytscet))
    test_loader_cet = torch.utils.data.DataLoader(dataset=test_dataset_cet, batch_size=mini_batch, shuffle=False,
                                                  num_workers=8, pin_memory=True)

    num_minibatches = int(n_samp_e / best_mini_batch)
    for _ in trange(best_epochs):
        auc_train, cost_train = train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e,
                                      optim_m, optim_c, optim_clas, all_triplet_selector, trip_criterion,
                                      bce_loss, device, cost_train, num_minibatches, auc_train, best_gamma)

    auc_test_erlo = test(test_loader_erlo, autoencoder_e, autoencoder_m, autoencoder_c, clas, device)
    auc_test_cet = test(test_loader_cet, autoencoder_e, autoencoder_m, autoencoder_c, clas, device)

    print(f'EGFR: AUROC Train = {auc_train[-1]}')
    print(f'EGFR Cetuximab: AUROCC = {auc_test_cet}')
    print(f'EGFR Erlotinib: AUROCC = {auc_test_erlo}')


def train(train_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, optim_e, optim_m, optim_c, optim_clas,
          all_triplet_selector, trip_criterion, bce_loss, device, cost_train, num_minibatches,
          auc_train, gamma):
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()
    for (dataE, dataM, dataC, target) in train_loader:
        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            dataE = dataE.to(device)
            dataM = dataM.to(device)
            dataC = dataC.to(device)
            target = target.to(device)

            for optimizer in (optim_e, optim_m, optim_c, optim_clas):
                optimizer.zero_grad()

            autoencoder_e.train()
            autoencoder_m.train()
            autoencoder_c.train()
            clas.train()

            with autocast():
                ZEX = autoencoder_e(dataE)
                ZMX = autoencoder_m(dataM)
                ZCX = autoencoder_c(dataC)

                ZT = torch.cat((ZEX, ZMX, ZCX), 1)
                ZT = F.normalize(ZT, p=2, dim=0)
                y_prediction = clas(ZT)
                triplets = all_triplet_selector.get_triplets(ZT, target)
                target = target.view(-1, 1)
                loss = gamma * trip_criterion(ZT[triplets[:, 0], :], ZT[triplets[:, 1], :],
                                              ZT[triplets[:, 2], :]) + bce_loss(y_prediction, target)

            auc = roc_auc_score(target.detach().cpu(), y_prediction.detach().cpu())
            scaler.scale(loss).backward()
            for optimizer in (optim_clas, optim_e, optim_m, optim_c):
                scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
            cost_train += (loss.item() / num_minibatches)
            auc_train.append(auc)
        return auc_train, cost_train


def test(data_loader, autoencoder_e, autoencoder_m, autoencoder_c, clas, device):
    y_true_test = []
    prediction_test = []
    for (dataE, dataM, dataC, target) in data_loader:
        tx_test_e = torch.FloatTensor(dataE).to(device)
        tx_test_m = torch.FloatTensor(dataM).to(device)
        tx_test_c = torch.FloatTensor(dataC).to(device)
        y_true_test.extend(target.view(-1, 1))
        with torch.no_grad():
            autoencoder_e.eval()
            autoencoder_m.eval()
            autoencoder_c.eval()
            clas.eval()
            ZET = autoencoder_e(tx_test_e)
            ZMT = autoencoder_m(tx_test_m)
            ZCT = autoencoder_c(tx_test_c)

            ztt = torch.cat((ZET, ZMT, ZCT), 1)
            ztt = F.normalize(ztt, p=2, dim=0)
            prediction_test.extend(clas(ztt).cpu().detach())

    auc_test = roc_auc_score(y_true_test, prediction_test)
    return auc_test


if __name__ == "__main__":
    main()
