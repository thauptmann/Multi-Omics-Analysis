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

    GDSCE = GDSCE[GDSCE.columns[get_high_variance_gen_indices(GDSCE)]]
    GDSCM = GDSCM[GDSCM.columns[get_high_variance_gen_indices(GDSCM)]]
    GDSCC = GDSCC[GDSCC.columns[get_high_variance_gen_indices(GDSCC)]]

    expression_intersection_genes_index = GDSCE.columns.intersection(PDXEcet.columns)
    expression_intersection_genes_index = expression_intersection_genes_index.intersection(PDXEerlo.columns)

    mutation_intersection_genes_index = GDSCM.columns.intersection(PDXMcet.columns)
    mutation_intersection_genes_index = mutation_intersection_genes_index.intersection(PDXMerlo.columns)

    cna_intersection_genes_index = GDSCC.columns.intersection(PDXCcet.columns)
    cna_intersection_genes_index = cna_intersection_genes_index.intersection(PDXCerlo.columns)

    extern_erlo_sample_intersection = PDXEerlo.index.intersection(PDXMerlo.index)
    extern_erlo_sample_intersection = extern_erlo_sample_intersection.intersection(PDXCerlo.index)

    extern_cet_sample_intersection = PDXEcet.index.intersection(PDXMcet.index)
    extern_cet_sample_intersection = extern_cet_sample_intersection.intersection(PDXCcet.index)

    train_samples_intersection = GDSCE.index.intersection(GDSCM.index)
    train_samples_intersection = train_samples_intersection.intersection(GDSCC.index)

    PDXEerlo = PDXEerlo.loc[extern_erlo_sample_intersection, expression_intersection_genes_index]
    PDXMerlo = PDXMerlo.loc[extern_erlo_sample_intersection, mutation_intersection_genes_index]
    PDXCerlo = PDXCerlo.loc[extern_erlo_sample_intersection, cna_intersection_genes_index]

    PDXEcet = PDXEcet.loc[extern_cet_sample_intersection, expression_intersection_genes_index]
    PDXMcet = PDXMcet.loc[extern_cet_sample_intersection, mutation_intersection_genes_index]
    PDXCcet = PDXCcet.loc[extern_cet_sample_intersection, cna_intersection_genes_index]

    GDSCE = GDSCE.loc[train_samples_intersection, expression_intersection_genes_index]
    GDSCM = GDSCM.loc[train_samples_intersection, mutation_intersection_genes_index]
    GDSCC = GDSCC.loc[train_samples_intersection, cna_intersection_genes_index]

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

    pdx_e_both = pd.concat([PDXEcet, PDXEerlo])
    pdx_m_both = pd.concat([PDXMcet, PDXMerlo])
    pdx_c_both = pd.concat([PDXCcet, PDXCerlo])
    pdx_r_both = np.concatenate([PDXRcet, PDXRerlo])
    return GDSCEv2.to_numpy(), GDSCMv2.to_numpy(), GDSCCv2.to_numpy(), GDSCRv2, \
           pdx_e_both.to_numpy(), pdx_m_both.to_numpy(), pdx_c_both.to_numpy(), pdx_r_both


def get_high_variance_gen_indices(data):
    selector = VarianceThreshold(0)
    return selector.fit(data).get_support(indices=True)


def load_drug_data(data_path, drug, dataset):
    drug = drug.split('_')[0]
    cna_binary_path = data_path / 'CNA_binary'
    response_path = data_path / 'response'
    sna_binary_path = data_path / 'SNA_binary'
    expressions_homogenized_path = data_path / 'exprs_homogenized'
    expression_train = read_and_transpose_csv(expressions_homogenized_path
                                              / f'GDSC_exprs.{drug}.eb_with.{dataset}_exprs.{drug}.tsv')
    response_train = pd.read_csv(response_path / f"GDSC_response.{drug}.tsv",
                                 sep="\t", index_col=0, decimal=',')
    mutation_train = read_and_transpose_csv(sna_binary_path / f"GDSC_mutations.{drug}.tsv")
    cna_train = read_and_transpose_csv(cna_binary_path / f"GDSC_CNA.{drug}.tsv")
    cna_train = cna_train.loc[:, ~cna_train.columns.duplicated()]
    expression_extern = read_and_transpose_csv(expressions_homogenized_path /
                                               f"{dataset}_exprs.{drug}.eb_with.GDSC_exprs.{drug}.tsv")
    mutation_extern = read_and_transpose_csv(sna_binary_path / f"{dataset}_mutations.{drug}.tsv")
    cna_extern = read_and_transpose_csv(cna_binary_path / f"{dataset}_CNA.{drug}.tsv")
    cna_extern = cna_extern.loc[:, ~cna_extern.columns.duplicated()]
    response_extern = pd.read_csv(response_path / f"{dataset}_response.{drug}.tsv",
                                  sep="\t", index_col=0, decimal=',')

    response_train.loc[response_train.response == 'R'] = 0
    response_train.loc[response_train.response == 'S'] = 1
    response_train.rename(mapper=str, axis='index', inplace=True)
    response_extern.loc[response_extern.response == 'R'] = 0
    response_extern.loc[response_extern.response == 'S'] = 1
    response_extern.rename(mapper=str, axis='index', inplace=True)

    cna_extern = cna_extern.fillna(0)
    cna_extern[cna_extern != 0.0] = 1
    cna_train = cna_train.fillna(0)
    cna_train[cna_train != 0.0] = 1
    mutation_extern = mutation_extern.fillna(0)
    mutation_extern[mutation_extern != 0.0] = 1
    mutation_train = mutation_train.fillna(0)
    mutation_train[mutation_train != 0.0] = 1

    expression_train = expression_train[expression_train.columns[get_high_variance_gen_indices(expression_train)]]
    mutation_train = mutation_train[mutation_train.columns[get_high_variance_gen_indices(mutation_train)]]
    cna_train = cna_train[cna_train.columns[get_high_variance_gen_indices(cna_train)]]

    expression_intersection_genes_index = expression_train.columns.intersection(expression_extern.columns)
    mutation_intersection_genes_index = mutation_train.columns.intersection(mutation_extern.columns)
    cna_intersection_genes_index = cna_train.columns.intersection(cna_extern.columns)

    extern_sample_intersection = expression_extern.index.intersection(mutation_extern.index)
    extern_sample_intersection = extern_sample_intersection.intersection(cna_extern.index)
    train_samples_intersection = expression_train.index.intersection(mutation_train.index)
    train_samples_intersection = train_samples_intersection.intersection(cna_train.index)

    expression_extern = expression_extern.loc[extern_sample_intersection, expression_intersection_genes_index]
    mutation_extern = mutation_extern.loc[extern_sample_intersection, mutation_intersection_genes_index]
    cna_extern = cna_extern.loc[extern_sample_intersection, cna_intersection_genes_index]
    response_extern = response_extern.loc[extern_sample_intersection, :]
    expression_train = expression_train.loc[train_samples_intersection, expression_intersection_genes_index]
    mutation_train = mutation_train.loc[train_samples_intersection, mutation_intersection_genes_index]
    cna_train = cna_train.loc[train_samples_intersection, cna_intersection_genes_index]
    response_train = response_train.loc[train_samples_intersection, :]

    y_train = response_train.response.to_numpy(dtype=int)
    y_extern = response_extern.response.to_numpy(dtype=int)
    return expression_train.to_numpy(), mutation_train.to_numpy(), cna_train.to_numpy(), y_train, \
           expression_extern.to_numpy(), mutation_extern.to_numpy(), cna_extern.to_numpy(), y_extern
