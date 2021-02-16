from pathlib import Path
import torch
import numpy as np
from tqdm import trange
import pandas as pd
import sklearn.preprocessing as sk
from sklearn.feature_selection import VarianceThreshold
from siamese_triplet.utils import AllTripletSelector
from torch.utils.data.sampler import WeightedRandomSampler
from utils import network_training_util
from models.moli_model import Moli
from utils.network_training_util import create_dataloader


def main():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    data_path = Path('../../data/')
    egfr_path = data_path / 'EGFR_experiments_data'

    GDSC_E = pd.read_csv(egfr_path / "GDSC_exprs.z.EGFRi.tsv", sep="\t", index_col=0, decimal=",")
    GDSC_E = pd.DataFrame.transpose(GDSC_E)

    GDSC_M = pd.read_csv(egfr_path / "GDSC_mutations.EGFRi.tsv", sep="\t", index_col=0, decimal=".")
    GDSC_M = pd.DataFrame.transpose(GDSC_M)
    GDSC_M = GDSC_M.loc[:, ~GDSC_M.columns.duplicated()]

    GDSC_C = pd.read_csv(egfr_path / "GDSC_CNA.EGFRi.tsv", sep="\t", index_col=0, decimal=".")
    GDSC_C = GDSC_C.drop_duplicates(keep='last')
    GDSC_C = pd.DataFrame.transpose(GDSC_C)
    GDSC_C = GDSC_C.loc[:, ~GDSC_C.columns.duplicated()]

    PDX_E_erlo = pd.read_csv(egfr_path / "PDX_exprs.Erlotinib.eb_with.GDSC_exprs.Erlotinib.tsv",
                             sep="\t", index_col=0, decimal=",")
    PDX_E_erlo = pd.DataFrame.transpose(PDX_E_erlo)

    PDX_M_erlo = pd.read_csv(egfr_path / "PDX_mutations.Erlotinib.tsv", sep="\t", index_col=0, decimal=",")
    PDX_M_erlo = pd.DataFrame.transpose(PDX_M_erlo)

    PDX_C_erlo = pd.read_csv(egfr_path / "PDX_CNV.Erlotinib.tsv", sep="\t", index_col=0, decimal=",")
    PDX_C_erlo.drop_duplicates(keep='last')
    PDX_C_erlo = pd.DataFrame.transpose(PDX_C_erlo)
    PDX_C_erlo = PDX_C_erlo.loc[:, ~PDX_C_erlo.columns.duplicated()]

    PDX_E_cet = pd.read_csv(egfr_path / "PDX_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv",
                            sep="\t", index_col=0, decimal=",")
    PDX_E_cet = pd.DataFrame.transpose(PDX_E_cet)

    PDX_M_cet = pd.read_csv(egfr_path / "PDX_mutations.Cetuximab.tsv", sep="\t", index_col=0, decimal=",")
    PDX_M_cet = pd.DataFrame.transpose(PDX_M_cet)

    PDX_C_cet = pd.read_csv(egfr_path / "PDX_CNV.Cetuximab.tsv", sep="\t", index_col=0, decimal=",")
    PDX_C_cet = PDX_C_cet.drop_duplicates(keep='last')
    PDX_C_cet = pd.DataFrame.transpose(PDX_C_cet)
    PDX_C_cet = PDX_C_cet.loc[:, ~PDX_C_cet.columns.duplicated()]

    selector = VarianceThreshold(0.05)
    selector.fit_transform(GDSC_E)
    GDSC_E = GDSC_E[GDSC_E.columns[selector.get_support(indices=True)]]

    GDSC_M = GDSC_M.fillna(0)
    GDSC_M[GDSC_M != 0.0] = 1
    GDSC_C = GDSC_C.fillna(0)
    GDSC_C[GDSC_C != 0.0] = 1

    PDX_M_cet = PDX_M_cet.fillna(0)
    PDX_M_cet[PDX_M_cet != 0.0] = 1
    PDX_C_cet = PDX_C_cet.fillna(0)
    PDX_C_cet[PDX_C_cet != 0.0] = 1

    PDX_M_erlo = PDX_M_erlo.fillna(0)
    PDX_M_erlo[PDX_M_erlo != 0.0] = 1
    PDX_C_erlo = PDX_C_erlo.fillna(0)
    PDX_C_erlo[PDX_C_erlo != 0.0] = 1

    ls = GDSC_E.columns.intersection(GDSC_M.columns)
    ls = ls.intersection(GDSC_C.columns)
    ls = ls.intersection(PDX_E_erlo.columns)
    ls = ls.intersection(PDX_M_erlo.columns)
    ls = ls.intersection(PDX_C_erlo.columns)
    ls = ls.intersection(PDX_E_cet.columns)
    ls = ls.intersection(PDX_M_cet.columns)
    ls = ls.intersection(PDX_C_cet.columns)
    ls3 = PDX_E_erlo.index.intersection(PDX_M_erlo.index)
    ls3 = ls3.intersection(PDX_C_erlo.index)
    ls4 = PDX_E_cet.index.intersection(PDX_M_cet.index)
    ls4 = ls4.intersection(PDX_C_cet.index)
    ls = pd.unique(ls)

    PDX_E_erlo = PDX_E_erlo.loc[ls3, ls]
    PDX_M_erlo = PDX_M_erlo.loc[ls3, ls]
    PDX_C_erlo = PDX_C_erlo.loc[ls3, ls]
    PDX_E_cet = PDX_E_cet.loc[ls4, ls]
    PDX_M_cet = PDX_M_cet.loc[ls4, ls]
    PDX_C_cet = PDX_C_cet.loc[ls4, ls]
    GDSC_E = GDSC_E.loc[:, ls]
    GDSC_M = GDSC_M.loc[:, ls]
    GDSC_C = GDSC_C.loc[:, ls]

    GDSC_R = pd.read_csv(egfr_path / "GDSC_response.EGFRi.tsv",
                         sep="\t", index_col=0, decimal=",")
    PDX_R_cet = pd.read_csv(egfr_path / "PDX_response.Cetuximab.tsv",
                            sep="\t", index_col=0, decimal=",")
    PDX_R_erlo = pd.read_csv(egfr_path / "PDX_response.Erlotinib.tsv",
                             sep="\t", index_col=0, decimal=",")

    PDX_R_cet = PDX_R_cet.loc[ls4, :]
    PDX_R_erlo = PDX_R_erlo.loc[ls3, :]

    GDSC_R.rename(mapper=str, axis='index', inplace=True)

    d = {"R": 0, "S": 1}
    GDSC_R["response"] = GDSC_R.loc[:, "response"].apply(lambda x: d[x])
    PDX_R_cet["response"] = PDX_R_cet.loc[:, "response"].apply(lambda x: d[x])
    PDX_R_erlo["response"] = PDX_R_erlo.loc[:, "response"].apply(lambda x: d[x])

    responses = GDSC_R
    drugs = set(responses["drug"].values)
    exprs_z = GDSC_E
    cna = GDSC_C
    mut = GDSC_M
    expression_z_scores = []
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
        expression_z_scores.append(e_z)
        CNA.append(c)
        mutations.append(m)
    responses.index = responses.index.values + "_" + responses["drug"].values
    GDSCEv2 = pd.concat(expression_z_scores, axis=0)
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
    x_test_e_erlo = PDX_E_erlo.values
    x_test_m_erlo = torch.FloatTensor(PDX_M_erlo.values)
    x_test_c_erlo = torch.FloatTensor(PDX_C_erlo.values)
    y_test_erlo = PDX_R_erlo['response'].values

    x_test_e_cet = PDX_E_cet.values
    x_test_m_cet = torch.FloatTensor(PDX_M_cet.values)
    x_test_c_cet = torch.FloatTensor(PDX_C_cet.values)
    y_test_cet = PDX_R_cet['response'].values

    y_test_cet = torch.FloatTensor(y_test_cet.astype(int))
    y_test_erlo = torch.FloatTensor(y_test_erlo.astype(int))

    scaler_gdsc = sk.StandardScaler()
    scaler_gdsc.fit(x_train_e)
    x_train_e = torch.FloatTensor(scaler_gdsc.transform(x_train_e))
    x_test_e_cet = torch.FloatTensor(scaler_gdsc.transform(x_test_e_cet))
    x_test_e_erlo = torch.FloatTensor(scaler_gdsc.transform(x_test_e_erlo))

    x_train_m = torch.FloatTensor(np.nan_to_num(x_train_m))
    x_train_c = torch.FloatTensor(np.nan_to_num(x_train_c))

    n_sample_e, ie_dim = x_train_e.shape
    _, im_dim = x_train_m.shape
    _, ic_dim = x_train_m.shape

    triplet_selector2 = AllTripletSelector()
    moli = Moli([ie_dim, im_dim, ic_dim], [h_dim1, h_dim2, h_dim3],
                           [dropout_rate_e, dropout_rate_m, dropout_rate_c,
                            dropout_rate_clf]).to(device)

    moli_optimiser = torch.optim.Adagrad([
        {'params': moli.expression_encoder.parameters(), 'lr': lr_e},
        {'params': moli.mutation_encoder.parameters(), 'lr': lr_m},
        {'params': moli.cna_encoder.parameters(), 'lr': lr_c},
        {'params': moli.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
    ])

    trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

    train_loader = create_dataloader(x_train_e, x_train_m, x_train_c, y_train, mini_batch, pin_memory, sampler, True)
    test_loader_erlo = create_dataloader(x_test_e_erlo, x_test_m_erlo, x_test_c_erlo, y_test_erlo, mini_batch,
                                         pin_memory)
    test_loader_cet = create_dataloader(x_test_e_cet, x_test_m_cet, x_test_c_cet, y_test_cet, mini_batch, pin_memory)

    auc_train = 0
    for _ in trange(epochs):
        auc_train, _ = network_training_util.train(train_loader, moli, moli_optimiser, triplet_selector2,
                                                   trip_criterion, bce_loss, device, gamma)

    auc_test_erlo = network_training_util.validate(test_loader_erlo, moli, device)
    auc_test_cet = network_training_util.validate(test_loader_cet, moli, device)

    print(f'EGFR: AUROC Train = {auc_train}')
    print(f'EGFR Cetuximab: AUROC Test = {auc_test_cet}')
    print(f'EGFR Erlotinib: AUROC Test = {auc_test_erlo}')


if __name__ == "__main__":
    main()
