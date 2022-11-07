import torch
from tqdm import tqdm

def compute_importances_values(X, explainer, baseline):
    mean_attributions = torch.zeros_like(X)
    for sample in tqdm(baseline):
        all_attributions = explainer.attribute(
            X,
            sample[None, :],
        )
        mean_attributions += all_attributions.detach().cpu()
    return all_attributions / len(baseline)
