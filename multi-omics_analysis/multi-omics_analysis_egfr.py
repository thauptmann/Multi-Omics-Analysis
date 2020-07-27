import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import encoder
from pathlib import Path
import sklearn as sk
import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from siamese_triplet.metrics import AverageNonzeroTripletsMetric
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import random
from random import randint
from sklearn.model_selection import StratifiedKFold
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import statsmodels.api as sm
from mne.stats import bonferroni_correction

def main():
    save_results_to = Path('.')
    torch.manual_seed(42)
    random.seed(42)

    GDSCE = pd.read_csv("GDSC_exprs.z.EGFRi.tsv",
                    sep="\t", index_col=0, decimal=",")
    GDSCE = pd.DataFrame.transpose(GDSCE)

    GDSCM = pd.read_csv("GDSC_mutations.EGFRi.tsv",
                        sep="\t", index_col=0, decimal=".")
    GDSCM = pd.DataFrame.transpose(GDSCM)
    GDSCM = GDSCM.loc[:, ~GDSCM.columns.duplicated()]

    GDSCC = pd.read_csv("GDSC_CNA.EGFRi.tsv",
                        sep="\t", index_col=0, decimal=".")
    GDSCC.drop_duplicates(keep='last')
    GDSCC = pd.DataFrame.transpose(GDSCC)
    GDSCC = GDSCC.loc[:, ~GDSCC.columns.duplicated()]

    PDXEerlo = pd.read_csv("PDX_exprs.Erlotinib.eb_with.GDSC_exprs.Erlotinib.tsv",
                           sep="\t", index_col=0, decimal=",")
    PDXEerlo = pd.DataFrame.transpose(PDXEerlo)
    PDXMerlo = pd.read_csv("PDX_mutations.Erlotinib.tsv",
                           sep="\t", index_col=0, decimal=",")
    PDXMerlo = pd.DataFrame.transpose(PDXMerlo)
    PDXCerlo = pd.read_csv("PDX_CNA.Erlotinib.tsv",
                           sep="\t", index_col=0, decimal=",")
    PDXCerlo.drop_duplicates(keep='last')
    PDXCerlo = pd.DataFrame.transpose(PDXCerlo)
    PDXCerlo = PDXCerlo.loc[:, ~PDXCerlo.columns.duplicated()]

    PDXEcet = pd.read_csv("PDX_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
    PDXEcet = pd.DataFrame.transpose(PDXEcet)
    PDXMcet = pd.read_csv("PDX_mutations.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
    PDXMcet = pd.DataFrame.transpose(PDXMcet)
    PDXCcet = pd.read_csv("PDX_CNA.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
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

    ls = GDSCE.columns.intersection(GDSCM.columns)
    ls = ls.intersection(GDSCC.columns)
    ls = ls.intersection(PDXEerlo.columns)
    ls = ls.intersection(PDXMerlo.columns)
    ls = ls.intersection(PDXCerlo.columns)
    ls = ls.intersection(PDXEcet.columns)
    ls = ls.intersection(PDXMcet.columns)
    ls = ls.intersection(PDXCcet.columns)
    ls2 = GDSCE.index.intersection(GDSCM.index)
    ls2 = ls2.intersection(GDSCC.index)
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

    GDSCR = pd.read_csv("GDSC_response.EGFRi.tsv",
                        sep="\t", index_col=0, decimal=",")

    GDSCR.rename(mapper=str, axis='index', inplace=True)

    d = {"R": 0, "S": 1}
    GDSCR["response"] = GDSCR.loc[:, "response"].apply(lambda x: d[x])

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

    Y = GDSCRv2['response'].values

    PDXRcet = pd.read_csv("PDX_response.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
    PDXRcet.loc[PDXRcet.iloc[:, 0] == 'R'] = 0
    PDXRcet.loc[PDXRcet.iloc[:, 0] == 'S'] = 1
    PDXRcet = PDXRcet.loc[ls4, :]
    Ytscet = PDXRcet['response'].values

    PDXRerlo = pd.read_csv("PDX_response.Erlotinib.tsv",
                           sep="\t", index_col=0, decimal=",")
    PDXRerlo.loc[PDXRerlo.iloc[:, 0] == 'R'] = 0
    PDXRerlo.loc[PDXRerlo.iloc[:, 0] == 'S'] = 1
    PDXRerlo = PDXRerlo.loc[ls3, :]
    Ytserlo = PDXRerlo['response'].values

    hdm1 = 32
    hdm2 = 16
    hdm3 = 256
    rate1 = 0.5
    rate2 = 0.8
    rate3 = 0.5
    rate4 = 0.3

    scalerGDSC = sk.StandardScaler()
    scalerGDSC.fit(GDSCEv2.values)
    X_trainE = scalerGDSC.transform(GDSCEv2.values)
    X_testEerlo = scalerGDSC.transform(PDXEerlo.values)
    X_testEcet = scalerGDSC.transform(PDXEcet.values)

    X_trainM = np.nan_to_num(GDSCMv2.values)
    X_trainC = np.nan_to_num(GDSCCv2.values)
    X_testMerlo = np.nan_to_num(PDXMerlo.values)
    X_testCerlo = np.nan_to_num(PDXCerlo.values)
    X_testMcet = np.nan_to_num(PDXMcet.values)
    X_testCcet = np.nan_to_num(PDXCcet.values)

    TX_testEerlo = torch.FloatTensor(X_testEerlo)
    TX_testMerlo = torch.FloatTensor(X_testMerlo)
    TX_testCerlo = torch.FloatTensor(X_testCerlo)
    ty_testEerlo = torch.FloatTensor(Ytserlo.astype(int))

    TX_testEcet = torch.FloatTensor(X_testEcet)
    TX_testMcet = torch.FloatTensor(X_testMcet)
    TX_testCcet = torch.FloatTensor(X_testCcet)
    ty_testEcet = torch.FloatTensor(Ytscet.astype(int))

    n_sampE, IE_dim = X_trainE.shape
    n_sampM, IM_dim = X_trainM.shape
    n_sampC, IC_dim = X_trainC.shape

    h_dim1 = hdm1
    h_dim2 = hdm2
    h_dim3 = hdm3
    Z_in = h_dim1 + h_dim2 + h_dim3

    costtr = []
    auctr = []
    costts = []
    aucts = []

    torch.cuda.manual_seed_all(42)

    AutoencoderE = torch.load('EGFRv2Exprs.pt')
    AutoencoderM = torch.load('EGFRv2Mut.pt')
    AutoencoderC = torch.load('EGFRv2CNA.pt')

    Clas = torch.load('EGFRv2Class.pt')

    AutoencoderE.eval()
    AutoencoderM.eval()
    AutoencoderC.eval()
    Clas.eval()

    ZEX = AutoencoderE(torch.FloatTensor(X_trainE))
    ZMX = AutoencoderM(torch.FloatTensor(X_trainM))
    ZCX = AutoencoderC(torch.FloatTensor(X_trainC))
    ZTX = torch.cat((ZEX, ZMX, ZCX), 1)
    ZTX = F.normalize(ZTX, p=2, dim=0)
    PredX = Clas(ZTX)
    AUCt = roc_auc_score(Y, PredX.detach().numpy())
    print(AUCt)

    ZETerlo = AutoencoderE(TX_testEerlo)
    ZMTerlo = AutoencoderM(TX_testMerlo)
    ZCTerlo = AutoencoderC(TX_testCerlo)
    ZTTerlo = torch.cat((ZETerlo, ZMTerlo, ZCTerlo), 1)
    ZTTerlo = F.normalize(ZTTerlo, p=2, dim=0)
    PredTerlo = Clas(ZTTerlo)
    AUCterlo = roc_auc_score(Ytserlo, PredTerlo.detach().numpy())
    print(AUCterlo)

    ZETcet = AutoencoderE(TX_testEcet)
    ZMTcet = AutoencoderM(TX_testMcet)
    ZCTcet = AutoencoderC(TX_testCcet)
    ZTTcet = torch.cat((ZETcet, ZMTcet, ZCTcet), 1)
    ZTTcet = F.normalize(ZTTcet, p=2, dim=0)
    PredTcet = Clas(ZTTcet)
    AUCtcet = roc_auc_score(Ytscet, PredTcet.detach().numpy())
    print(AUCtcet)

    PRADE = pd.read_csv("TCGA-PRAD_exprs.tsv",
                        sep="\t", index_col=0, decimal=".")
    PRADE = pd.DataFrame.transpose(PRADE)

    PRADM = pd.read_csv("TCGA-PRAD_mutations.tsv",
                        sep="\t", index_col=0, decimal=".")
    PRADM = pd.DataFrame.transpose(PRADM)
    PRADM = PRADM.loc[:, ~PRADM.columns.duplicated()]

    PRADC = pd.read_csv("TCGA-PRAD_CNA.tsv",
                        sep="\t", index_col=0, decimal=".")
    PRADC = pd.DataFrame.transpose(PRADC)
    PRADC = PRADC.loc[:, ~PRADC.columns.duplicated()]

    PRADM = PRADM.fillna(0)
    PRADM[PRADM != 0.0] = 1
    PRADC = PRADC.fillna(0)
    PRADC[PRADC != 0.0] = 1

    # PRADE.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # PRADM.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # PRADC.rename(lambda x : x[0:11], axis = "index", inplace=True)

    lsPRAD = PRADE.index.intersection(PRADM.index)
    lsPRAD = lsPRAD.intersection(PRADC.index)
    lsPRAD = pd.unique(lsPRAD)

    PRADE = PRADE.loc[lsPRAD, ls]
    PRADM = PRADM.loc[lsPRAD, ls]
    PRADC = PRADC.loc[lsPRAD, ls]

    print(PRADE.shape)
    print(PRADM.shape)
    print(PRADC.shape)

    AutoencoderE.eval()
    AutoencoderM.eval()
    AutoencoderC.eval()
    Clas.eval()

    PRADE2 = np.nan_to_num(PRADE.values)
    PRADM2 = np.nan_to_num(PRADM.values)
    PRADC2 = np.nan_to_num(PRADC.values)

    NPRADE2 = scalerGDSC.transform(PRADE2)

    PRADexprs = torch.FloatTensor(NPRADE2)
    PRADmut = torch.FloatTensor(PRADM2)
    PRADcna = torch.FloatTensor(PRADC2)

    PRADZE = AutoencoderE(PRADexprs)
    PRADZM = AutoencoderM(PRADmut)
    PRADZC = AutoencoderC(PRADcna)

    PRADZT = torch.cat((PRADZE, PRADZM, PRADZC), 1)
    PRADZTX = F.normalize(PRADZT, p=2, dim=0)
    PredPRAD = Clas(PRADZTX)

    KIRPE = pd.read_csv("TCGA-KIRP_exprs.tsv",
                        sep="\t", index_col=0, decimal=".")
    KIRPE = pd.DataFrame.transpose(KIRPE)

    KIRPM = pd.read_csv("TCGA-KIRP_mutations.tsv",
                        sep="\t", index_col=0, decimal=".")
    KIRPM = pd.DataFrame.transpose(KIRPM)
    KIRPM = KIRPM.loc[:, ~KIRPM.columns.duplicated()]

    KIRPC = pd.read_csv("TCGA-KIRP_CNA.tsv",
                        sep="\t", index_col=0, decimal=".")
    KIRPC = pd.DataFrame.transpose(KIRPC)
    KIRPC = KIRPC.loc[:, ~KIRPC.columns.duplicated()]

    KIRPM = KIRPM.fillna(0)
    KIRPM[KIRPM != 0.0] = 1
    KIRPC = KIRPC.fillna(0)
    KIRPC[KIRPC != 0.0] = 1

    # KIRPE.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # KIRPM.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # KIRPC.rename(lambda x : x[0:11], axis = "index", inplace=True)

    lsKIRP = KIRPE.index.intersection(KIRPM.index)
    lsKIRP = lsKIRP.intersection(KIRPC.index)
    lsKIRP = pd.unique(lsKIRP)

    KIRPE = KIRPE.loc[lsKIRP, ls]
    KIRPM = KIRPM.loc[lsKIRP, ls]
    KIRPC = KIRPC.loc[lsKIRP, ls]

    print(KIRPE.shape)
    print(KIRPM.shape)
    print(KIRPC.shape)

    AutoencoderE.eval()
    AutoencoderM.eval()
    AutoencoderC.eval()
    Clas.eval()

    KIRPE2 = np.nan_to_num(KIRPE.values)
    KIRPM2 = np.nan_to_num(KIRPM.values)
    KIRPC2 = np.nan_to_num(KIRPC.values)

    NKIRPE2 = scalerGDSC.transform(KIRPE2)

    KIRPexprs = torch.FloatTensor(NKIRPE2)
    KIRPmut = torch.FloatTensor(KIRPM2)
    KIRPcna = torch.FloatTensor(KIRPC2)

    KIRPZE = AutoencoderE(KIRPexprs)
    KIRPZM = AutoencoderM(KIRPmut)
    KIRPZC = AutoencoderC(KIRPcna)

    KIRPZT = torch.cat((KIRPZE, KIRPZM, KIRPZC), 1)
    KIRPZTX = F.normalize(KIRPZT, p=2, dim=0)
    PredKIRP = Clas(KIRPZTX)

    BLCAE = pd.read_csv("TCGA-BLCA_exprs.tsv",
                        sep="\t", index_col=0, decimal=".")
    BLCAE = pd.DataFrame.transpose(BLCAE)

    BLCAM = pd.read_csv("TCGA-BLCA_mutations.tsv",
                        sep="\t", index_col=0, decimal=".")
    BLCAM = pd.DataFrame.transpose(BLCAM)
    BLCAM = BLCAM.loc[:, ~BLCAM.columns.duplicated()]

    BLCAC = pd.read_csv("TCGA-BLCA_CNA.tsv",
                        sep="\t", index_col=0, decimal=".")
    BLCAC = pd.DataFrame.transpose(BLCAC)
    BLCAC = BLCAC.loc[:, ~BLCAC.columns.duplicated()]

    BLCAM = BLCAM.fillna(0)
    BLCAM[BLCAM != 0.0] = 1
    BLCAC = BLCAC.fillna(0)
    BLCAC[BLCAC != 0.0] = 1

    # BLCAE.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # BLCAM.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # BLCAC.rename(lambda x : x[0:11], axis = "index", inplace=True)

    lsBLCA = BLCAE.index.intersection(BLCAM.index)
    lsBLCA = lsBLCA.intersection(BLCAC.index)
    lsBLCA = pd.unique(lsBLCA)

    BLCAE = BLCAE.loc[lsBLCA, ls]
    BLCAM = BLCAM.loc[lsBLCA, ls]
    BLCAC = BLCAC.loc[lsBLCA, ls]

    print(BLCAE.shape)
    print(BLCAM.shape)
    print(BLCAC.shape)

    AutoencoderE.eval()
    AutoencoderM.eval()
    AutoencoderC.eval()
    Clas.eval()

    BLCAE2 = np.nan_to_num(BLCAE.values)
    BLCAM2 = np.nan_to_num(BLCAM.values)
    BLCAC2 = np.nan_to_num(BLCAC.values)

    NBLCAE2 = scalerGDSC.transform(BLCAE2)

    BLCAexprs = torch.FloatTensor(NBLCAE2)
    BLCAmut = torch.FloatTensor(BLCAM2)
    BLCAcna = torch.FloatTensor(BLCAC2)

    BLCAZE = AutoencoderE(BLCAexprs)
    BLCAZM = AutoencoderM(BLCAmut)
    BLCAZC = AutoencoderC(BLCAcna)

    BLCAZT = torch.cat((BLCAZE, BLCAZM, BLCAZC), 1)
    BLCAZTX = F.normalize(BLCAZT, p=2, dim=0)
    PredBLCA = Clas(BLCAZTX)

    BRCAE = pd.read_csv("TCGA-BRCA_exprs.tsv",
                        sep="\t", index_col=0, decimal=".")
    BRCAE = pd.DataFrame.transpose(BRCAE)

    BRCAM = pd.read_csv("TCGA-BRCA_mutations.tsv",
                        sep="\t", index_col=0, decimal=".")
    BRCAM = pd.DataFrame.transpose(BRCAM)
    BRCAM = BRCAM.loc[:, ~BRCAM.columns.duplicated()]

    BRCAC = pd.read_csv("TCGA-BRCA_CNA.tsv",
                        sep="\t", index_col=0, decimal=".")
    BRCAC = pd.DataFrame.transpose(BRCAC)
    BRCAC = BRCAC.loc[:, ~BRCAC.columns.duplicated()]

    BRCAM = BRCAM.fillna(0)
    BRCAM[BRCAM != 0.0] = 1
    BRCAC = BRCAC.fillna(0)
    BRCAC[BRCAC != 0.0] = 1

    # BRCAE.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # BRCAM.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # BRCAC.rename(lambda x : x[0:11], axis = "index", inplace=True)

    lsBRCA = BRCAE.index.intersection(BRCAM.index)
    lsBRCA = lsBRCA.intersection(BRCAC.index)
    lsBRCA = pd.unique(lsBRCA)

    BRCAE = BRCAE.loc[lsBRCA, ls]
    BRCAM = BRCAM.loc[lsBRCA, ls]
    BRCAC = BRCAC.loc[lsBRCA, ls]

    print(BRCAE.shape)
    print(BRCAM.shape)
    print(BRCAC.shape)

    AutoencoderE.eval()
    AutoencoderM.eval()
    AutoencoderC.eval()
    Clas.eval()

    BRCAE2 = np.nan_to_num(BRCAE.values)
    BRCAM2 = np.nan_to_num(BRCAM.values)
    BRCAC2 = np.nan_to_num(BRCAC.values)

    NBRCAE2 = scalerGDSC.transform(BRCAE2)

    BRCAexprs = torch.FloatTensor(NBRCAE2)
    BRCAmut = torch.FloatTensor(BRCAM2)
    BRCAcna = torch.FloatTensor(BRCAC2)

    BRCAZE = AutoencoderE(BRCAexprs)
    BRCAZM = AutoencoderM(BRCAmut)
    BRCAZC = AutoencoderC(BRCAcna)

    BRCAZT = torch.cat((BRCAZE, BRCAZM, BRCAZC), 1)
    BRCAZTX = F.normalize(BRCAZT, p=2, dim=0)
    PredBRCA = Clas(BRCAZTX)

    PAADE = pd.read_csv("TCGA-PAAD_exprs.tsv",
                        sep="\t", index_col=0, decimal=".")
    PAADE = pd.DataFrame.transpose(PAADE)

    PAADM = pd.read_csv("TCGA-PAAD_mutations.tsv",
                        sep="\t", index_col=0, decimal=".")
    PAADM = pd.DataFrame.transpose(PAADM)
    PAADM = PAADM.loc[:, ~PAADM.columns.duplicated()]

    PAADC = pd.read_csv("TCGA-PAAD_CNA.tsv",
                        sep="\t", index_col=0, decimal=".")
    PAADC = pd.DataFrame.transpose(PAADC)
    PAADC = PAADC.loc[:, ~PAADC.columns.duplicated()]

    PAADM = PAADM.fillna(0)
    PAADM[PAADM != 0.0] = 1
    PAADC = PAADC.fillna(0)
    PAADC[PAADC != 0.0] = 1

    # PAADE.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # PAADM.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # PAADC.rename(lambda x : x[0:11], axis = "index", inplace=True)

    lsPAAD = PAADE.index.intersection(PAADM.index)
    lsPAAD = lsPAAD.intersection(PAADC.index)
    lsPAAD = pd.unique(lsPAAD)

    PAADE = PAADE.loc[lsPAAD, ls]
    PAADM = PAADM.loc[lsPAAD, ls]
    PAADC = PAADC.loc[lsPAAD, ls]

    print(PAADE.shape)
    print(PAADM.shape)
    print(PAADC.shape)

    AutoencoderE.eval()
    AutoencoderM.eval()
    AutoencoderC.eval()
    Clas.eval()

    PAADE2 = np.nan_to_num(PAADE.values)
    PAADM2 = np.nan_to_num(PAADM.values)
    PAADC2 = np.nan_to_num(PAADC.values)

    NPAADE2 = scalerGDSC.transform(PAADE2)

    PAADexprs = torch.FloatTensor(NPAADE2)
    PAADmut = torch.FloatTensor(PAADM2)
    PAADcna = torch.FloatTensor(PAADC2)

    PAADZE = AutoencoderE(PAADexprs)
    PAADZM = AutoencoderM(PAADmut)
    PAADZC = AutoencoderC(PAADcna)

    PAADZT = torch.cat((PAADZE, PAADZM, PAADZC), 1)
    PAADZTX = F.normalize(PAADZT, p=2, dim=0)
    PredPAAD = Clas(PAADZTX)

    LUADE = pd.read_csv("TCGA-LUAD_exprs.tsv",
                        sep="\t", index_col=0, decimal=".")
    LUADE = pd.DataFrame.transpose(LUADE)

    LUADM = pd.read_csv("TCGA-LUAD_mutations.tsv",
                        sep="\t", index_col=0, decimal=".")
    LUADM = pd.DataFrame.transpose(LUADM)
    LUADM = LUADM.loc[:, ~LUADM.columns.duplicated()]

    LUADC = pd.read_csv("TCGA-LUAD_CNA.tsv",
                        sep="\t", index_col=0, decimal=".")
    LUADC = pd.DataFrame.transpose(LUADC)
    LUADC = LUADC.loc[:, ~LUADC.columns.duplicated()]

    LUADM = LUADM.fillna(0)
    LUADM[LUADM != 0.0] = 1
    LUADC = LUADC.fillna(0)
    LUADC[LUADC != 0.0] = 1

    # LUADE.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # LUADM.rename(lambda x : x[0:11], axis = "index", inplace=True)
    # LUADC.rename(lambda x : x[0:11], axis = "index", inplace=True)

    lsLUAD = LUADE.index.intersection(LUADM.index)
    lsLUAD = lsLUAD.intersection(LUADC.index)
    lsLUAD = pd.unique(lsLUAD)

    LUADE = LUADE.loc[lsLUAD, ls]
    LUADM = LUADM.loc[lsLUAD, ls]
    LUADC = LUADC.loc[lsLUAD, ls]

    print(LUADE.shape)
    print(LUADM.shape)
    print(LUADC.shape)

    AutoencoderE.eval()
    AutoencoderM.eval()
    AutoencoderC.eval()
    Clas.eval()

    LUADE2 = np.nan_to_num(LUADE.values)
    LUADM2 = np.nan_to_num(LUADM.values)
    LUADC2 = np.nan_to_num(LUADC.values)

    NLUADE2 = scalerGDSC.transform(LUADE2)

    LUADexprs = torch.FloatTensor(NLUADE2)
    LUADmut = torch.FloatTensor(LUADM2)
    LUADcna = torch.FloatTensor(LUADC2)

    LUADZE = AutoencoderE(LUADexprs)
    LUADZM = AutoencoderM(LUADmut)
    LUADZC = AutoencoderC(LUADcna)

    LUADZT = torch.cat((LUADZE, LUADZM, LUADZC), 1)
    LUADZTX = F.normalize(LUADZT, p=2, dim=0)
    PredLUAD = Clas(LUADZTX)

    lsEGFR = [10000, 102, 10252, 10253, 10254, 1026, 1027, 107, 108, 109, 111, 11140, 112, 113, 114, 1147, 115, 117145,
              1173, 1175, 1211, 1213, 1385, 1445, 156, 160, 161, 163, 1950, 1956, 196883, 2060, 207, 208, 2308, 2309, 23239,
              2475, 253260, 2549, 26018, 2885, 2931, 29924, 30011, 3164, 3265, 3320, 3709, 3710, 3845, 4193, 4303, 4893,
              5136, 5153, 5170, 5290, 5295, 5335, 5566, 5567, 5568, 5573, 5575, 5576, 5577, 5578, 5580, 5581, 5582, 55824,
              5594, 5595, 5604, 5605, 572, 5728, 57761, 58513, 5894, 6199, 6233, 64223, 6456, 6464, 6654, 6714, 6868, 7249,
              728590, 729120, 730418, 7311, 731292, 7529, 79109, 801, 8027, 8038, 805, 808, 814, 842, 84335, 867, 9146, 983,
              998]

    listEGFR = PRADE.columns.intersection(lsEGFR)
    PRADEEGFR = PRADE[listEGFR]
    PRADMEGFR = PRADM[listEGFR]
    PRADCEGFR = PRADC[listEGFR]

    X = PRADEEGFR
    y = PredPRAD.detach().numpy()

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)  # make the predictions by the model

    # Print out the statistics
    model.summary()

    print(bonferroni_correction(model.pvalues, alpha=0.05))

    listEGFR = KIRPE.columns.intersection(lsEGFR)
    KIRPEEGFR = KIRPE[listEGFR]
    KIRPMEGFR = KIRPM[listEGFR]
    KIRPCEGFR = KIRPC[listEGFR]

    X = KIRPEEGFR
    y = PredKIRP.detach().numpy()

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)  # make the predictions by the model

    # Print out the statistics
    model.summary()
    print(bonferroni_correction(model.pvalues, alpha=0.05))

    listEGFR = BLCAE.columns.intersection(lsEGFR)
    BLCAEEGFR = BLCAE[listEGFR]
    BLCAMEGFR = BLCAM[listEGFR]
    BLCACEGFR = BLCAC[listEGFR]

    X = BLCAEEGFR
    y = PredBLCA.detach().numpy()

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)  # make the predictions by the model

    # Print out the statistics
    model.summary()
    print(bonferroni_correction(model.pvalues, alpha=0.05))
    listEGFR = BRCAE.columns.intersection(lsEGFR)
    BRCAEEGFR = BRCAE[listEGFR]
    BRCAMEGFR = BRCAM[listEGFR]
    BRCACEGFR = BRCAC[listEGFR]
    X = BRCAEEGFR
    y = PredBRCA.detach().numpy()

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)  # make the predictions by the model

    # Print out the statistics
    model.summary()
    print(bonferroni_correction(model.pvalues, alpha=0.05))
    listEGFR = PAADE.columns.intersection(lsEGFR)
    PAADEEGFR = PAADE[listEGFR]
    PAADMEGFR = PAADM[listEGFR]
    PAADCEGFR = PAADC[listEGFR]

    X = PAADEEGFR
    y = PredPAAD.detach().numpy()

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)  # make the predictions by the model

    # Print out the statistics
    model.summary()

    print(bonferroni_correction(model.pvalues, alpha=0.05))

    listEGFR = LUADE.columns.intersection(lsEGFR)
    LUADEEGFR = LUADE[listEGFR]
    LUADMEGFR = LUADM[listEGFR]
    LUADCEGFR = LUADC[listEGFR]

    X = LUADEEGFR
    y = PredLUAD.detach().numpy()

    # Note the difference in argument order
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)  # make the predictions by the model

    # Print out the statistics
    model.summary()
    print(bonferroni_correction(model.pvalues, alpha=0.05))

if __name__ == "__main__":
    # execute only if run as a script
    main()
