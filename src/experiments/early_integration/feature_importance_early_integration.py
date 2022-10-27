import shap
import yaml
from pathlib import Path

from src.utils.input_arguments import get_cmd_arguments

file_directory = Path(__file__).parent
with open((file_directory / "../../config/hyperparameter.yaml"), "r") as stream:
    parameter = yaml.safe_load(stream)

def early_integration_feature_importance():
    pass

if __name__ == "__main__":
    args = get_cmd_arguments()

    extern_dataset = parameter["drugs"]["Cetuximab"]
    early_integration_feature_importance(
        args.search_iterations,
        args.experiment_name,
        args.drug,
        extern_dataset,
        args.gpu_number,
        args.deactivate_triplet_loss,
    )