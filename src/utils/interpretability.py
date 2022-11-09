import torch
from tqdm import tqdm

def compute_importances_values(X, explainer, baseline):
    mean_attributions = torch.zeros_like(X)
    for sample in tqdm(baseline):
        all_attributions = explainer.attribute(
            X,
            sample[None, :],
        )
        mean_attributions += all_attributions.detach()
    return (all_attributions / len(baseline)).detach().cpu().numpy()

def save_importance_results(importances, feature_names, predictions, targets, path, dataset):
    df = None
    df.to_csv(path / dataset + ".csv")
