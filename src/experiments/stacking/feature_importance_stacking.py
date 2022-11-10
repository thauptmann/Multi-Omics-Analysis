from sklearn.preprocessing import StandardScaler
import yaml
import torch
from pathlib import Path
import numpy as np
import sys
from captum.attr import IntegratedGradients, GradientShap

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from models.stacking_model import StackingModel
from utils.network_training_util import create_sampler, get_loss_fn
from utils.input_arguments import get_cmd_arguments
from utils import multi_omics_data
from utils.interpretability import compute_importances_values_single_input
from utils.network_training_util import (
    get_loss_fn,
    create_data_loader,
    create_sampler,
    train,
    test,
)
from utils.visualisation import visualize_importances

file_directory = Path(__file__).parent
with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)


mini_batch = 16
h_dim = 512
lr = 0.001
dropout_rate = 0.1
weight_decay = 0.0001
margin = 1.0
epochs = 2
gamma = 0

torch.manual_seed(parameter["random_seed"])
np.random.seed(parameter["random_seed"])


def stacking_feature_importance(
    experiment_name,
    drug_name,
    extern_dataset_name,
):
    device = torch.device("cpu")
    pin_memory = False
    result_path = Path(
        file_directory,
        "..",
        "..",
        "..",
        "results",
        "early_integration",
        "explanation",
        experiment_name,
        drug_name,
    )
    result_path.mkdir(exist_ok=True, parents=True)
    data_path = Path(file_directory, "..", "..", "..", "data")
    (
        gdsc_e,
        gdsc_m,
        gdsc_c,
        gdsc_r,
        extern_e,
        extern_m,
        extern_c,
        extern_r,
    ) = multi_omics_data.load_drug_data_with_elbow(
        data_path, drug_name, extern_dataset_name, return_data_frames=True
    )
    # get columns names
    expression_columns = gdsc_e.columns
    expression_columns = [
        f"expression_{expression_gene}" for expression_gene in expression_columns
    ]

    mutation_columns = gdsc_m.columns
    mutation_columns = [
        f"mutation_{mutation_gene}" for mutation_gene in mutation_columns
    ]

    cna_columns = gdsc_c.columns
    cna_columns = [f"cna_{cna_gene}" for cna_gene in cna_columns]

    all_columns = np.concatenate([expression_columns, mutation_columns, cna_columns])

    gdsc_e = gdsc_e.to_numpy()
    gdsc_m = gdsc_m.to_numpy()
    gdsc_c = gdsc_c.to_numpy()
    extern_e = extern_e.to_numpy()
    extern_m = extern_m.to_numpy()
    extern_c = extern_c.to_numpy()

    gdsc_concat = np.concatenate([gdsc_e, gdsc_m, gdsc_c], axis=1)
    extern_concat = np.concatenate([extern_e, extern_m, extern_c], axis=1)

    baseline = torch.zeros_like(torch.FloatTensor([gdsc_concat[0]]))

    baseline = gdsc_concat

    scaler_gdsc = StandardScaler()
    gdsc_concat = torch.FloatTensor(scaler_gdsc.fit_transform(gdsc_concat))
    extern_concat = torch.FloatTensor(scaler_gdsc.transform(extern_concat))
    scaled_baseline = torch.FloatTensor(scaler_gdsc.transform(baseline))

    # Initialisation
    sampler = create_sampler(gdsc_r)
    dataset = torch.utils.data.TensorDataset(gdsc_concat, torch.FloatTensor(gdsc_r))
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=mini_batch,
        shuffle=False,
        num_workers=8,
        pin_memory=pin_memory,
        drop_last=True,
        sampler=sampler,
    )
    _, ie_dim = gdsc_concat.shape

    loss_fn = get_loss_fn(margin, gamma)

    stacking_model = StackingModel(
        ie_dim,
        h_dim,
        dropout_rate,
    ).to(device)

    stacking_optimiser = torch.optim.Adagrad(
        stacking_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    for _ in range(epochs):
        train(
            train_loader,
            stacking_model,
            stacking_optimiser,
            loss_fn,
            device,
            gamma,
        )
    stacking_model.eval()

    train_predictions = stacking_model(gdsc_concat)
    gdsc_concat.requires_grad_()
    integradet_gradients = GradientShap(stacking_model)

    all_attributions_test = compute_importances_values_single_input(
        gdsc_concat,
        gdsc_r,
        train_predictions,
        all_columns,
        integradet_gradients,
        scaled_baseline,
    )
    visualize_importances(
        all_columns,
        all_attributions_test.detach().numpy(),
        path=result_path,
        file_name="all_attributions_test",
    )

    extern_predictions = stacking_model(extern_concat)
    extern_concat.requires_grad_()
    all_attributions_extern = compute_importances_values_single_input(
        extern_concat,
        extern_r,
        extern_predictions,
        all_columns,
        integradet_gradients,
        scaled_baseline,
    )
    visualize_importances(
        all_columns,
        all_attributions_extern.detach().numpy(),
        path=result_path,
        file_name="all_attributions_extern",
    )


if __name__ == "__main__":
    args = get_cmd_arguments()

    if args.drug == "all":
        for drug, extern_dataset in parameter["drugs"].items():
            stacking_feature_importance(
                args.experiment_name,
                drug,
                extern_dataset,
            )
    else:
        extern_dataset = parameter["drugs"][args.drug]
        stacking_feature_importance(
            args.experiment_name,
            args.drug,
            extern_dataset,
        )
