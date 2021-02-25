import sys
from pathlib import Path
import torch
from ax import (
    ParameterType,
    RangeParameter,
    ChoiceParameter,
    SearchSpace,
    SimpleExperiment,
    save,
    load,
    FixedParameter
)
from ax.modelbridge.registry import Models

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
import argparse
from pathlib import Path
import numpy as np
import auto_moli_egfr
from utils import egfr_data
from utils.visualisation import save_auroc_plots

mini_batch_list = [8, 16, 32, 64]
dim_list = [1024, 512, 256, 128, 64, 32, 16, 8]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
epoch_list = [10, 20, 50, 15, 30, 40, 60, 70, 80, 90, 100]
drop_rate_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.1, 0.01, 0.001, 0.0001]
gamma_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
combination_list = [0, 1, 2, 3]
depth_list = [1, 2, 3]


def bo_moli(search_iterations, run_test, sobol_iterations, load_checkpoint, experiment_name, combination):
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")

    result_path = Path('..', '..', '..', 'results', 'egfr', 'bayesian_optimisation', experiment_name)
    checkpoint_path = result_path / 'checkpoint.json'
    result_path.mkdir(parents=True, exist_ok=True)

    data_path = Path('..', '..', '..', 'data')
    egfr_path = Path(data_path, 'EGFR_experiments_data')
    GDSCE, GDSCM, GDSCC, GDSCR, PDXEerlo, PDXMerlo, PDXCerlo, PDXRerlo, PDXEcet, PDXMcet, PDXCcet, PDXRcet = \
        egfr_data.load_data(egfr_path)

    moli_search_space = create_search_space(combination)
    # load or set up experiment with initial sobel runs
    if load_checkpoint & checkpoint_path.exists():
        print("Load checkpoint")
        experiment = load(str(checkpoint_path))
        experiment.evaluation_function = lambda parameterization: auto_moli_egfr.train_evaluate(parameterization,
                                                                                                GDSCE, GDSCM, GDSCC,
                                                                                                GDSCR, device)
        print(f"Resuming with iteration {len(experiment.trials.values()) + 1}")
    else:
        experiment = SimpleExperiment(
            name="BO-MOLI",
            search_space=moli_search_space,
            evaluation_function=lambda parameterization: auto_moli_egfr.train_evaluate(parameterization,
                                                                                       GDSCE, GDSCM, GDSCC, GDSCR,
                                                                                       device),
            objective_name="AUROC",
            minimize=False,
        )

        print(f"Running Sobol initialization trials...")
        sobol = Models.SOBOL(experiment.search_space, seed=random_seed)
        for i in range(sobol_iterations):
            experiment.new_trial(generator_run=sobol.gen(1))

    best_arm = None
    for i in range(len(experiment.trials.values()), search_iterations + 1):
        print(f"Running GP+EI optimization trial {i + 1 - sobol_iterations}/{search_iterations}...")
        # Reinitialize GP+EI model at each step with updated data.
        gpei = Models.BOTORCH(experiment=experiment, data=experiment.eval())
        generator_run = gpei.gen(1)
        best_arm, _ = generator_run.best_arm_predictions
        experiment.new_trial(generator_run=generator_run)
        save(experiment, str(checkpoint_path))

        if i % 10 == 0:
            best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
            save_auroc_plots(best_objectives, result_path, sobol_iterations)
            best_parameters = best_arm.parameters
            print(best_parameters)

    best_parameters = best_arm.parameters

    # save all results
    print(best_parameters)
    print("Done!")

    best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
    save(experiment, str(checkpoint_path))
    np.save(best_objectives, result_path / 'best_objectives')
    save_auroc_plots(best_objectives, result_path, sobol_iterations)

    if run_test:
        auc_train, auc_test_erlo, auc_test_cet = auto_moli_egfr.train_and_test(best_parameters, GDSCE, GDSCM, GDSCC,
                                                                               GDSCR, PDXEerlo, PDXMerlo, PDXCerlo,
                                                                               PDXRerlo, PDXEcet, PDXMcet, PDXCcet,
                                                                               PDXRcet, device)

        print(f'EGFR: AUROC Train = {auc_train}')
        print(f'EGFR Cetuximab: AUROC = {auc_test_cet}')
        print(f'EGFR Erlotinib: AUROC = {auc_test_erlo}')


def create_search_space(combination):
    if combination is None:
        combination_parameter = ChoiceParameter(name='combination', values=combination_list,
                                                parameter_type=ParameterType.INT)
    else:
        combination_parameter = FixedParameter(name='combination', value=combination,
                                                parameter_type=ParameterType.INT)
    return SearchSpace(
        parameters=[
            RangeParameter(name='mini_batch', lower=8, upper=64, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim1", lower=8, upper=1024, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim2", lower=8, upper=1024, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim3", lower=8, upper=1024, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim4", lower=8, upper=1024, parameter_type=ParameterType.INT),
            RangeParameter(name="h_dim5", lower=8, upper=1024, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_1", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_2", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_3", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_4", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="depth_5", lower=1, upper=3, parameter_type=ParameterType.INT),
            RangeParameter(name="lr_e", lower=0.00001, upper=0.1, log_scale=True, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="lr_m", lower=0.00001, upper=0.1, log_scale=True, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="lr_c", lower=0.00001, upper=0.1, log_scale=True, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="lr_cl", lower=0.00001, upper=0.1, log_scale=True, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="lr_middle", lower=0.00001, upper=0.1, log_scale=True,
                           parameter_type=ParameterType.FLOAT),
            RangeParameter(name="dropout_rate_e", lower=0.0, upper=0.8, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="dropout_rate_m", lower=0.0, upper=0.8, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="dropout_rate_c", lower=0.0, upper=0.8, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="dropout_rate_clf", lower=0.0, upper=0.8, parameter_type=ParameterType.FLOAT),
            RangeParameter(name="dropout_rate_middle", lower=0.0, upper=0.8, parameter_type=ParameterType.FLOAT),
            RangeParameter(name='weight_decay', lower=0.001, upper=0.1, log_scale=True,
                           parameter_type=ParameterType.FLOAT),
            RangeParameter(name='gamma', lower=0.0, upper=0.6, parameter_type=ParameterType.FLOAT),
            # RangeParameter(name='epochs', lower=10, upper=100, parameter_type=ParameterType.INT),
            combination_parameter,
            RangeParameter(name='epochs', lower=1, upper=2, parameter_type=ParameterType.INT),
            RangeParameter(name='margin', lower=0.5, upper=2.5, parameter_type=ParameterType.FLOAT),

        ]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--search_iterations', default=1, type=int)
    parser.add_argument('--sobol_iterations', default=5, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--combination', default=None, type=int)
    args = parser.parse_args()
    bo_moli(args.search_iterations, args.run_test, args.sobol_iterations, args.load_checkpoint, args.experiment_name,
            args.combination)
