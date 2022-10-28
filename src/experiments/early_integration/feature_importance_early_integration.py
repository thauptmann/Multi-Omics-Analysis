import shap
import yaml
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.input_arguments import get_cmd_arguments
from utils import multi_omics_data

file_directory = Path(__file__).parent
with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)


parameterization = {
    "mini_batch": 16,
    "h_dim": 512,
    "lr": 0.001,
    "dropout_rate": 0.1,
    "weight_decay": 0.0001,
    "margin": 1.0,
    "epochs": 2,
    "gamma": 0,
}


def early_integration_feature_importance(
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
        "explain_cetuximab",
    )
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
        data_path, drug_name, extern_dataset_name
    )


if __name__ == "__main__":
    args = get_cmd_arguments()

    extern_dataset = parameter["drugs"]["Cetuximab"]
    early_integration_feature_importance(
        args.drug,
        extern_dataset,
        args.gpu_number,
    )
