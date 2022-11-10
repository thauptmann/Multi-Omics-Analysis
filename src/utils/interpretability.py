import torch
from tqdm import tqdm
import numpy as np


def compute_importances_values_single_input(X, explainer, baseline):
    # mean_attributions = torch.zeros_like(X)
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
    expression_attributions = (expression_attributions / len(baseline[0])).detach().cpu().numpy()
    mutation_attributions = (mutation_attributions / len(baseline[0])).detach().cpu().numpy()
    cna_attributions = (cna_attributions / len(baseline[0])).detach().cpu().numpy()
    result_attributions = np.concatenate(
        [expression_attributions, mutation_attributions, cna_attributions]
    )
    return result_attributions


def save_importance_results(
    importances, feature_names, predictions, targets, path, dataset
):
    df = None
    df.to_csv(path / dataset + ".csv")
