import sys
from datetime import datetime
from pathlib import Path
import torch
import pickle
import time
from ax import (
    ParameterType,
    RangeParameter,
    SearchSpace,
    SimpleExperiment,
    FixedParameter
)
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.modelbridge_utils import get_pending_observation_features
from ax.storage.json_store.load import load_experiment
from ax.storage.json_store.save import save_experiment

from sklearn.model_selection import StratifiedKFold
from ax.modelbridge.registry import Models
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
import argparse
from pathlib import Path
import numpy as np
from training_bo_holi_moli import train_and_validate, train_final
from utils import multi_omics_data
from utils.visualisation import save_auroc_plots, save_auroc_with_variance_plots
from utils.network_training_util import calculate_mean_and_std_auc, test

depth_lower = 1
depth_upper = 4
drop_rate_lower = 0.0
drop_rate_upper = 0.9
weight_decay_lower = 0.0001
weight_decay_upper = 0.1
gamma_lower = 0.0
gamma_upper = 0.6
dim_lower = 8
dim_upper = 256
margin_lower = 0.5
margin_upper = 3.5
learning_rate_lower = 0.00001
learning_rate_upper = 0.01
combination_lower = 0
combination_upper = 4
batch_size_lower = 16
batch_size_upper = 32
epoch_lower = 10
epoch_upper = 50

drugs = {
    'Gemcitabine_tcga': 'TCGA',
    'Gemcitabine_pdx': 'PDX',
    'Cisplatin': 'TCGA',
    'Docetaxel': 'TCGA',
    'Erlotinib': 'PDX',
    'Cetuximab': 'PDX',
    'Paclitaxel': 'PDX',
    # 'EGFR': 'PDX'
}

random_seed = 42


def bo_moli(search_iterations, sobol_iterations, load_checkpoint, experiment_name, combination,
            sampling_method, drug_name, extern_dataset_name, gpu_number, small_search_space):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
        pin_memory = False
    else:
        device = torch.device("cpu")
        pin_memory = False

    result_path = Path('..', '..', '..', 'results', 'bayesian_optimisation', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)

    file_mode = 'a' if load_checkpoint else 'w'
    result_file = open(result_path / 'logs.txt', file_mode)
    checkpoint_path = result_path / 'checkpoint.json'
    result_file.write(f"Start for {drug_name}\n")
    print(f"Start for {drug_name}")

    data_path = Path('..', '..', '..', 'data')
    if drug_name == 'EGFR':
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_egfr_data(data_path)
    else:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name)
    moli_search_space = create_search_space(combination, small_search_space)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    max_objective_list = []
    test_auc_list = []
    extern_auc_list = []
    objectives_list = []
    now = datetime.now()
    result_file.write(f'Start experiment at {now}\n')
    cv_splits = 5
    skf = StratifiedKFold(n_splits=cv_splits, random_state=random_seed, shuffle=True)
    iteration = 0

    start_time = time.time()
    for train_index, test_index in tqdm(skf.split(gdsc_e, gdsc_r), total=skf.get_n_splits(), desc=" Outer k-fold"):
        result_file.write(f'\t{iteration = }. \n')
        x_train_e = gdsc_e[train_index]
        x_train_m = gdsc_m[train_index]
        x_train_c = gdsc_c[train_index]
        y_train = gdsc_r[train_index]
        x_test_e = gdsc_e[test_index]
        x_test_m = gdsc_m[test_index]
        x_test_c = gdsc_c[test_index]
        y_test = gdsc_r[test_index]

        # load or set up experiment with initial sobel runs
        evaluation_function = lambda parameterization: train_and_validate(parameterization,
                                                                          x_train_e, x_train_m,
                                                                          x_train_c,
                                                                          y_train, device, pin_memory)
        if load_checkpoint & checkpoint_path.exists():
            print("Load checkpoint")
            experiment = load_experiment(str(checkpoint_path))
            experiment.evaluation_function = evaluation_function
            print(f"Resuming after iteration {len(experiment.trials.values())}")

        else:
            experiment = SimpleExperiment(
                name="BO-MOLI",
                search_space=moli_search_space,
                evaluation_function=evaluation_function,
                objective_name="auroc",
                minimize=False,
            )

        if sampling_method == 'gp':
            print('Using sobol+GPEI')
            generation_strategy = GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL,
                                   num_trials=sobol_iterations,
                                   model_kwargs={"seed": random_seed}),
                    GenerationStep(
                        model=Models.BOTORCH,
                        num_trials=-1,
                    ),
                ],
                name="Sobol+GPEI"
            )
        elif sampling_method == 'saasbo':
            print('Using sobol+SAASBO')
            generation_strategy = GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL,
                                   num_trials=sobol_iterations
                                   ),
                    GenerationStep(
                        model=Models.FULLYBAYESIAN,
                        num_trials=-1,
                        model_kwargs={
                            "num_samples": 256,
                            "warmup_steps": 512,
                            "disable_progbar": True,
                            "torch_device": torch.device('cpu'),
                            "torch_dtype": torch.double,
                        },
                    ),
                ],
                name="SAASBO"
            )
        else:
            print('Using only sobol')
            sobol_iterations = search_iterations
            generation_strategy = GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_trials=search_iterations),
                ],
                name="Sobol"
            )

        for i in range(0, search_iterations):
            if i < sobol_iterations:
                print(f"Running sobol optimization trial {i + 1} ...")
            else:
                print(f"Running {sampling_method} optimization trial {i - sobol_iterations + 1} ...")

            # Reinitialize GP+EI model at each step with updated data.
            generator_run = generation_strategy.gen(
                experiment=experiment, n=1, pending_observations=get_pending_observation_features(experiment)
            )
            experiment.new_trial(generator_run)
            experiment.eval()
            save_experiment(experiment, str(checkpoint_path))

            if i % 10 == 0 and i != 0:
                best_parameters = extract_best_parameter(experiment)
                objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
                save_auroc_plots(objectives, result_path, iteration, sobol_iterations)
                print(best_parameters)

        # save results
        best_parameters = extract_best_parameter(experiment)
        max_objective = max(np.array([trial.objective_mean for trial in experiment.trials.values()]))
        objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
        save_experiment(experiment, str(checkpoint_path))
        pickle.dump(objectives, open(result_path / 'objectives', "wb"))
        pickle.dump(best_parameters, open(result_path / 'best_parameters', "wb"))
        save_auroc_plots(objectives, result_path, iteration, sobol_iterations)

        iteration += 1

        result_file.write(f'\t\t{str(best_parameters) = }\n')

        model_test, scaler_test = train_final(best_parameters, x_train_e, x_train_m, x_train_c, y_train, device,
                                              pin_memory)
        auc_test = test(model_test, scaler_test, x_test_e, x_test_m, x_test_c, y_test, device, pin_memory)

        model_extern, scaler_extern = train_final(best_parameters, gdsc_e, gdsc_m, gdsc_c, gdsc_r, device, pin_memory)
        auc_extern = test(model_extern, scaler_extern, extern_e, extern_m, extern_c, extern_r, device, pin_memory)

        result_file.write(f'\t\tBest {drug} validation Auroc = {max_objective}\n')
        result_file.write(f'\t\t{drug} test Auroc = {auc_test}\n')
        result_file.write(f'\t\t{drug} extern AUROC = {auc_extern}\n')
        objectives_list.append(objectives)
        max_objective_list.append(max_objective)
        test_auc_list.append(auc_test)
        extern_auc_list.append(auc_extern)

    print("Done!")

    result_dict = {
        'validation': max_objective_list,
        'test': test_auc_list,
        'extern': extern_auc_list
    }
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)
    save_auroc_with_variance_plots(objectives_list, result_path, 'final', sobol_iterations)

    end_time = time.time()
    result_file.write(f'\tMinutes needed: {round((end_time - start_time) / 60)}')
    result_file.close()


def extract_best_parameter(experiment):
    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    best_parameters = best_arm.parameters
    return best_parameters


def create_search_space(combination, small_search_space):
    if combination is None:
        combination_parameter = RangeParameter(name='combination', lower=combination_lower, upper=combination_upper,
                                               parameter_type=ParameterType.INT)
    else:
        combination_parameter = FixedParameter(name='combination', value=combination,
                                               parameter_type=ParameterType.INT)

    if combination is None and not small_search_space:
        search_space = SearchSpace(
            parameters=[
                RangeParameter(name='mini_batch', lower=batch_size_lower, upper=batch_size_upper,
                               parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim1", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim2", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim3", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim4", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim5", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_1", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_2", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_3", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_4", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_5", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="lr_e", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="lr_m", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="lr_c", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="lr_cl", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="lr_middle", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="dropout_rate_e", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name="dropout_rate_m", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name="dropout_rate_c", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name="dropout_rate_clf", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name="dropout_rate_middle", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name='weight_decay', lower=weight_decay_lower, upper=weight_decay_upper, log_scale=True,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name='gamma', lower=gamma_lower, upper=gamma_upper, parameter_type=ParameterType.FLOAT),
                RangeParameter(name='epochs', lower=epoch_lower, upper=epoch_upper, parameter_type=ParameterType.INT),
                combination_parameter,
                RangeParameter(name='margin', lower=margin_lower, upper=margin_upper,
                               parameter_type=ParameterType.FLOAT),
            ]
        )
    # moli
    elif combination is not None and not small_search_space:
        search_space = SearchSpace(
            parameters=[
                RangeParameter(name='mini_batch', lower=batch_size_lower, upper=batch_size_upper,
                               parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim1", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim2", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim3", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim5", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_1", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_2", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_3", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_5", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="lr_e", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="lr_m", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="lr_c", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="lr_cl", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="dropout_rate_e", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name="dropout_rate_m", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name="dropout_rate_c", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name="dropout_rate_clf", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name='weight_decay', lower=weight_decay_lower, upper=weight_decay_upper, log_scale=True,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name='gamma', lower=gamma_lower, upper=gamma_upper, parameter_type=ParameterType.FLOAT),
                RangeParameter(name='epochs', lower=epoch_lower, upper=epoch_upper, parameter_type=ParameterType.INT),
                combination_parameter,
                RangeParameter(name='margin', lower=margin_lower, upper=margin_upper,
                               parameter_type=ParameterType.FLOAT),
            ]
        )
    else:
        search_space = SearchSpace(
            parameters=[
                RangeParameter(name='mini_batch', lower=batch_size_lower, upper=batch_size_upper,
                               parameter_type=ParameterType.INT),
                RangeParameter(name="h_dim1", lower=dim_lower, upper=dim_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="depth_1", lower=depth_lower, upper=depth_upper, parameter_type=ParameterType.INT),
                RangeParameter(name="lr_e", lower=learning_rate_lower, upper=learning_rate_upper,
                               parameter_type=ParameterType.FLOAT, log_scale=True),
                RangeParameter(name="dropout_rate_e", lower=drop_rate_lower, upper=drop_rate_upper,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name='weight_decay', lower=weight_decay_lower, upper=weight_decay_upper, log_scale=True,
                               parameter_type=ParameterType.FLOAT),
                RangeParameter(name='gamma', lower=gamma_lower, upper=gamma_upper, parameter_type=ParameterType.FLOAT),
                RangeParameter(name='epochs', lower=epoch_lower, upper=epoch_upper, parameter_type=ParameterType.INT),
                combination_parameter,
                RangeParameter(name='margin', lower=margin_lower, upper=margin_upper,
                               parameter_type=ParameterType.FLOAT),
            ]
        )
    return search_space


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_iterations', default=200, type=int)
    parser.add_argument('--sobol_iterations', default=20, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--combination', default=None, type=int)
    parser.add_argument('--sampling_method', default='gp', choices=['gp', 'sobol', 'saasbo'])
    parser.add_argument('--gpu_number', type=int)
    parser.add_argument('--small_search_space', default=False, action='store_true')
    args = parser.parse_args()

    for drug, extern_dataset in drugs.items():
        bo_moli(args.search_iterations, args.sobol_iterations, args.load_checkpoint, args.experiment_name,
                args.combination, args.sampling_method, drug, extern_dataset, args.gpu_number, args.small_search_space)
