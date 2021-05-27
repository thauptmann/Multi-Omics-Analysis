import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from utils.network_training_util import read_and_transpose_csv


def load_data(data_path):
    expression_path = data_path / 'exprs_homogenized'
    mutation_path = data_path / 'SNA_binary'
    cna_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'

    GDSCE = read_and_transpose_csv(expression_path / "GDSC_exprs.EGFRi.eb_with.PDX_exprs.EGFRi.tsv")
    GDSCM = pd.read_csv(mutation_path / "GDSC_mutations.EGFRi.tsv", sep="\t", index_col=0, decimal=".")
    GDSCM = pd.DataFrame.transpose(GDSCM)
    GDSCM = GDSCM.loc[:, ~GDSCM.columns.duplicated()]

    GDSCC = pd.read_csv(cna_path / "GDSC_CNA.EGFRi.tsv", sep="\t", index_col=0, decimal=".")
    GDSCC = pd.DataFrame.transpose(GDSCC)
    GDSCC = GDSCC.loc[:, ~GDSCC.columns.duplicated()]

    selector = VarianceThreshold(0.05)
    selector.fit(GDSCE)
    GDSCE = GDSCE[GDSCE.columns[selector.get_support(indices=True)]]

    GDSCM = GDSCM.fillna(0)
    GDSCM[GDSCM != 0.0] = 1
    GDSCC = GDSCC.fillna(0)
    GDSCC[GDSCC != 0.0] = 1

    ls = GDSCE.columns.intersection(GDSCM.columns)
    ls = ls.intersection(GDSCC.columns)
    ls = pd.unique(ls)

    GDSCE = GDSCE.loc[:, ls]
    GDSCM = GDSCM.loc[:, ls]
    GDSCC = GDSCC.loc[:, ls]

    GDSCR = pd.read_csv(response_path / "GDSC_response.EGFRi.tsv", sep="\t", index_col=0, decimal=",")
    GDSCR.rename(mapper=str, axis='index', inplace=True)

    d = {"R": 0, "S": 1}
    GDSCR["response"] = GDSCR.loc[:, "response"].apply(lambda x: d[x])

    responses = GDSCR
    drugs = set(responses["drug"].values)
    exprs_z = GDSCE
    cna = GDSCC
    mut = GDSCM
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
    GDSCRv2 = GDSCRv2['response'].values

    GDSCMv2 = np.nan_to_num(GDSCMv2)
    GDSCCv2 = np.nan_to_num(GDSCCv2)
    GDSCEv2 = GDSCEv2.to_numpy()

    return GDSCEv2, GDSCMv2, GDSCCv2, GDSCRv2
