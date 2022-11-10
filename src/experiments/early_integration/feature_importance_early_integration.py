from sklearn.preprocessing import StandardScaler
import yaml
import torch
from pathlib import Path
import numpy as np
import sys
from captum.attr import DeepLift, ShapleyValueSampling

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from models.early_integration_model import EarlyIntegration
from utils.network_training_util import create_sampler, get_loss_fn
from utils.input_arguments import get_cmd_arguments
from utils import multi_omics_data
from utils.interpretability import (
    compute_importances_values_single_input,
    save_importance_results,
)
from train_early_integration import train_early_integration
from utils.visualisation import visualize_importances
from utils.choose_gpu import create_device

file_directory = Path(__file__).parent
with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)

best_hyperparameter = {
    "Cetuximab": {
        "mini_batch": 16,
        "h_dim": 512,
        "lr": 0.001,
        "dropout_rate": 0.1,
        "weight_decay": 0.0001,
        "margin": 1.0,
        "epochs": 2,
        "gamma": 0,
    },
    "Docetaxel": {
        "mini_batch": 8,
        "h_dim": 64,
        "lr": 0.001,
        "dropout_rate": 0.3,
        "weight_decay": 0.0001,
        "margin": 1.0,
        "epochs": 13,
        "gamma": 0,
    },
}

torch.manual_seed(parameter["random_seed"])
np.random.seed(parameter["random_seed"])


def early_integration_feature_importance(
    experiment_name,
    drug_name,
    extern_dataset_name,
    convert_ids,
    gpu_number,
):
    hyperparameter = best_hyperparameter[drug_name]
    mini_batch = hyperparameter["mini_batch"]
    h_dim = hyperparameter["h_dim"]
    lr = hyperparameter["lr"]
    dropout_rate = hyperparameter["dropout_rate"]
    weight_decay = hyperparameter["weight_decay"]
    margin = hyperparameter["margin"]
    epochs = hyperparameter["epochs"]
    gamma = hyperparameter["gamma"]

    device, pin_memory = create_device(gpu_number)
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
        f"Expression {expression_gene}" for expression_gene in expression_columns
    ]

    mutation_columns = gdsc_m.columns
    mutation_columns = [
        f"Mutation {mutation_gene}" for mutation_gene in mutation_columns
    ]

    cna_columns = gdsc_c.columns
    cna_columns = [f"CNA {cna_gene}" for cna_gene in cna_columns]

    all_columns = np.concatenate([expression_columns, mutation_columns, cna_columns])

    gdsc_e = gdsc_e.to_numpy()
    gdsc_m = gdsc_m.to_numpy()
    gdsc_c = gdsc_c.to_numpy()
    extern_e = extern_e.to_numpy()
    extern_m = extern_m.to_numpy()
    extern_c = extern_c.to_numpy()

    number_of_expression_features = gdsc_e.shape[1]
    number_of_mutation_features = gdsc_m.shape[1]

    gdsc_concat = np.concatenate([gdsc_e, gdsc_m, gdsc_c], axis=1)
    extern_concat = np.concatenate([extern_e, extern_m, extern_c], axis=1)

    # baseline = torch.zeros_like(torch.FloatTensor([gdsc_concat[0]]))

    scaler_gdsc = StandardScaler()
    gdsc_concat_scaled = torch.Tensor(scaler_gdsc.fit_transform(gdsc_concat))
    extern_concat_scaled = torch.Tensor(scaler_gdsc.transform(extern_concat))
    scaled_baseline = torch.Tensor(gdsc_concat_scaled).to(device)

    # Initialisation
    sampler = create_sampler(gdsc_r)
    dataset = torch.utils.data.TensorDataset(gdsc_concat_scaled, torch.Tensor(gdsc_r))
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=mini_batch,
        shuffle=False,
        num_workers=8,
        pin_memory=pin_memory,
        drop_last=True,
        sampler=sampler,
    )
    _, ie_dim = gdsc_concat_scaled.shape

    loss_fn = get_loss_fn(margin, gamma)

    early_integration_model = EarlyIntegration(
        ie_dim,
        h_dim,
        dropout_rate,
    ).to(device)

    moli_optimiser = torch.optim.Adagrad(
        early_integration_model.parameters(), lr=lr, weight_decay=weight_decay
    )

    for _ in range(epochs):
        train_early_integration(
            train_loader,
            early_integration_model,
            moli_optimiser,
            loss_fn,
            device,
            gamma,
        )
    early_integration_model.eval()

    gdsc_concat_scaled = gdsc_concat_scaled.to(device)
    train_predictions = early_integration_model(gdsc_concat_scaled)
    gdsc_concat_scaled.requires_grad_()
    integradet_gradients = DeepLift(early_integration_model)

    all_attributions_test = compute_importances_values_single_input(
        gdsc_concat_scaled,
        integradet_gradients,
        scaled_baseline,
    )

    # prepare data for visualization
    min_value = np.min(gdsc_concat)
    max_value = np.max(gdsc_concat)
    homogenized_mutation = np.ones_like(gdsc_m)
    homogenized_mutation[gdsc_m == 1] = max_value
    homogenized_mutation[gdsc_m == 0] = min_value

    homogenized_cna = np.ones_like(gdsc_c)
    homogenized_cna[gdsc_c == 1] = max_value
    homogenized_cna[gdsc_c == 0] = min_value

    gdsc_visualize = np.concatenate(
        [gdsc_e, homogenized_mutation, homogenized_cna], axis=1
    )

    visualize_importances(
        all_columns,
        all_attributions_test,
        gdsc_r,
        train_predictions,
        gdsc_visualize,
        path=result_path,
        file_name="all_attributions_test",
        convert_ids=convert_ids,
        number_of_expression_features=number_of_expression_features,
        number_of_mutation_features=number_of_mutation_features,
    )

    extern_concat_scaled = extern_concat_scaled.to(device)
    extern_predictions = early_integration_model(extern_concat_scaled)
    extern_concat_scaled.requires_grad_()
    all_attributions_extern = compute_importances_values_single_input(
        extern_concat_scaled, integradet_gradients, scaled_baseline
    )

    homogenized_mutation = np.ones_like(extern_m, dtype=np.float32)
    homogenized_mutation[extern_m == 1] = max_value
    homogenized_mutation[extern_m == 0] = min_value

    homogenized_cna = np.ones_like(extern_c, dtype=np.float32)
    homogenized_cna[extern_c == 1] = max_value
    homogenized_cna[extern_c == 0] = min_value

    extern_visualization = None
    extern_visualization = np.concatenate(
        [extern_e, homogenized_mutation, homogenized_cna], axis=1
    )

    visualize_importances(
        all_columns,
        all_attributions_extern,
        extern_r,
        extern_predictions,
        extern_visualization,
        path=result_path,
        file_name="all_attributions_extern",
        convert_ids=convert_ids,
        number_of_expression_features=number_of_expression_features,
        number_of_mutation_features=number_of_mutation_features,
    )

    save_importance_results(
        all_attributions_test, all_columns, extern_predictions, gdsc_r, "extern"
    )
    save_importance_results(
        all_attributions_extern, all_columns, train_predictions, extern_r, "test"
    )


if __name__ == "__main__":
    args = get_cmd_arguments()

    if args.drug == "all":
        for drug, extern_dataset in parameter["drugs"].items():
            early_integration_feature_importance(
                args.experiment_name,
                drug,
                extern_dataset,
                args.convert_ids,
                args.gpu_number,
            )
    else:
        extern_dataset = parameter["drugs"][args.drug]
        early_integration_feature_importance(
            args.experiment_name,
            args.drug,
            extern_dataset,
            args.convert_ids,
            args.gpu_number,
        )
