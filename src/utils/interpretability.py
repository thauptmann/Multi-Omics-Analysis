import torch
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from Bio import Entrez

Entrez.email = os.environ.get("MAIL")


def compute_importances_values_single_input(X, explainer, baseline):
    mean_attributions = torch.zeros_like(X)
    for sample in tqdm(baseline):
        all_attributions = explainer.attribute(
            X,
            baselines=sample[None, :],
        )
        mean_attributions += all_attributions.detach()
    return (all_attributions / len(baseline)).detach().cpu().numpy()


def compute_importances_values_multiple_inputs(X, explainer, baseline):
    expression_attributions = torch.zeros_like(X[0])
    mutation_attributions = torch.zeros_like(X[1])
    cna_attributions = torch.zeros_like(X[2])
    for e, m, c in tqdm(zip(baseline[0], baseline[1], baseline[2])):
        all_attributions = explainer.attribute(
            X,
            baselines=(e[None, :], m[None, :], c[None, :]),
        )
        expression_attributions += all_attributions[0].detach()
        mutation_attributions += all_attributions[1].detach()
        cna_attributions += all_attributions[2].detach()
    expression_attributions = (
        (expression_attributions / len(baseline[0])).detach().cpu().numpy()
    )
    mutation_attributions = (
        (mutation_attributions / len(baseline[0])).detach().cpu().numpy()
    )
    cna_attributions = (cna_attributions / len(baseline[0])).detach().cpu().numpy()
    result_attributions = np.concatenate(
        [expression_attributions, mutation_attributions, cna_attributions], axis=1
    )
    return result_attributions


def save_importance_results(importances, feature_names, path, dataset):
    mean_importances = np.mean(importances, axis=0)
    sd_importances = np.std(importances, axis=0)

    absolute_sorted_indices = (np.abs(mean_importances)).argsort()
    absolute_most_important_features = feature_names[absolute_sorted_indices]
    absolute_highest_importances_mean = mean_importances[absolute_sorted_indices]
    absolute_highest_importance_sd = sd_importances[absolute_sorted_indices]
    feature_names = convert_genez_id_to_name(feature_names)
    data = {
        "feature_names": absolute_most_important_features,
        "mean importances": absolute_highest_importances_mean,
        "sd_importances": absolute_highest_importance_sd,
    }
    df = pd.DataFrame(data)
    df.to_csv(str(path / dataset) + ".csv")


def convert_genez_id_to_name(feature_names):
    names = []
    types = ids = [feature.split(" ")[0] for feature in feature_names]
    ids = [feature.split(" ")[1] for feature in feature_names]

    # Rest call to get names for ids
    request = Entrez.epost("gene", id=",".join(ids))
    result = Entrez.read(request)
    webEnv = result["WebEnv"]
    queryKey = result["QueryKey"]
    data = Entrez.esummary(db="gene", webenv=webEnv, query_key=queryKey)
    annotations = Entrez.read(data)
    for annotation in annotations.items():
        document_summary = annotation[1]["DocumentSummary"]
        for gene_data in document_summary:
            gene_name = gene_data["Name"]
            names.append(gene_name)

    # return converted features
    return [type + f" {name}" for type, name in zip(types, names)]
