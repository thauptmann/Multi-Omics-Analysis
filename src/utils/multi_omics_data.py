import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from utils.network_training_util import read_and_transpose_csv


def load_egfr_data(data_path):
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

    PDXEerlo = pd.read_csv(expression_path / "PDX_exprs.Erlotinib.eb_with.GDSC_exprs.Erlotinib.tsv",
                           sep="\t", index_col=0, decimal=",")
    PDXEerlo = pd.DataFrame.transpose(PDXEerlo)

    PDXMerlo = pd.read_csv(mutation_path / "PDX_mutations.Erlotinib.tsv", sep="\t", index_col=0, decimal=",")
    PDXMerlo = pd.DataFrame.transpose(PDXMerlo)

    PDXCerlo = pd.read_csv(cna_path / "PDX_CNA.Erlotinib.tsv", sep="\t", index_col=0, decimal=",")
    PDXCerlo = pd.DataFrame.transpose(PDXCerlo)
    PDXCerlo = PDXCerlo.loc[:, ~PDXCerlo.columns.duplicated()]

    PDXEcet = pd.read_csv(expression_path / "PDX_exprs.Cetuximab.eb_with.GDSC_exprs.Cetuximab.tsv",
                          sep="\t", index_col=0, decimal=",")
    PDXEcet = pd.DataFrame.transpose(PDXEcet)

    PDXMcet = pd.read_csv(mutation_path / "PDX_mutations.Cetuximab.tsv", sep="\t", index_col=0, decimal=",")
    PDXMcet = pd.DataFrame.transpose(PDXMcet)

    PDXCcet = pd.read_csv(cna_path / "PDX_CNA.Cetuximab.tsv", sep="\t", index_col=0, decimal=",")
    PDXCcet = pd.DataFrame.transpose(PDXCcet)
    PDXCcet = PDXCcet.loc[:, ~PDXCcet.columns.duplicated()]

    selector = VarianceThreshold(0.05)
    selector.fit(GDSCE)
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
    PDXRerlo = PDXRerlo['response'].values
    PDXRcet = PDXRcet['response'].values

    GDSCMv2 = np.nan_to_num(GDSCMv2)
    GDSCCv2 = np.nan_to_num(GDSCCv2)
    GDSCEv2 = GDSCEv2.to_numpy()

    PDXMcet = np.nan_to_num(PDXMcet)
    PDXCcet = np.nan_to_num(PDXCcet)
    PDXEcet = PDXEcet.to_numpy()

    PDXMerlo = np.nan_to_num(PDXMerlo)
    PDXCerlo = np.nan_to_num(PDXCerlo)
    PDXEerlo = PDXEerlo.to_numpy()

    return GDSCEv2, GDSCMv2, GDSCCv2, GDSCRv2, PDXEerlo, PDXMerlo, PDXCerlo, PDXRerlo, PDXEcet, PDXMcet, PDXCcet, \
           PDXRcet


def load_drug_data(data_path, drug, dataset):
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'
    expression_train = read_and_transpose_csv(expressions_homogenized_path / parameter['expression_train'])
    response_train = pd.read_csv(response_path / parameter['response_train'], sep="\t", index_col=0, decimal=',')
    mutation_train = read_and_transpose_csv(sna_binary_path / parameter['mutation_train'])
    cna_train = read_and_transpose_csv(cna_binary_path / parameter['cna_train'])
    cna_train = cna_train.loc[:, ~cna_train.columns.duplicated()]
    expression_test = read_and_transpose_csv(expressions_homogenized_path / parameter['expression_test'])
    mutation_test = read_and_transpose_csv(sna_binary_path / parameter['mutation_test'])
    cna_test = read_and_transpose_csv(cna_binary_path / parameter['cna_test'])
    response_test = pd.read_csv(response_path / parameter['response_test'], sep="\t", index_col=0, decimal=',')
    response_train.loc[response_train.response == 'R'] = 0
    response_train.loc[response_train.response == 'S'] = 1
    response_test.loc[response_test.response == 'R'] = 0
    response_test.loc[response_test.response == 'S'] = 1
    response_test.rename(mapper=str, axis='index', inplace=True)
    response_train.rename(mapper=str, axis='index', inplace=True)
    selector = VarianceThreshold(0.05)
    selector.fit(expression_train)
    expression_train = expression_train[expression_train.columns[selector.get_support(indices=True)]]
    cna_test = cna_test.fillna(0)
    cna_test[cna_test != 0.0] = 1
    cna_train = cna_train.fillna(0)
    cna_train[cna_train != 0.0] = 1
    mutation_test = mutation_test.fillna(0)
    mutation_test[mutation_test != 0.0] = 1
    mutation_train = mutation_train.fillna(0)
    mutation_train[mutation_train != 0.0] = 1
    ls = expression_train.columns.intersection(mutation_train.columns)
    ls = ls.intersection(cna_train.columns)
    ls = ls.intersection(expression_test.columns)
    ls = ls.intersection(mutation_test.columns)
    ls = ls.intersection(cna_test.columns)
    ls = pd.unique(ls)
    ls2 = expression_train.index.intersection(mutation_train.index)
    ls2 = ls2.intersection(cna_train.index)
    ls3 = expression_test.index.intersection(mutation_test.index)
    ls3 = ls3.intersection(cna_test.index)
    expression_test = expression_test.loc[ls3, ls]
    mutation_test = mutation_test.loc[ls3, ls]
    cna_test = cna_test.loc[ls3, ls]
    response_test = response_test.loc[ls3, :]
    expression_train = expression_train.loc[ls2, ls]
    mutation_train = mutation_train.loc[ls2, ls]
    cna_train = cna_train.loc[ls2, ls]
    response_train = response_train.loc[ls2, :]
    y = response_train.response.to_numpy(dtype=int)
    return cna_test, cna_train, expression_test, expression_train, mutation_test, mutation_train, response_test, \
           response_train, y
