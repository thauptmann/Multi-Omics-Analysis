import sys
from datetime import datetime
from pathlib import Path
import torch
import pickle
import time
from ax import (
    ParameterType,
    RangeParameter,
    ChoiceParameter,
    SearchSpace,
    SimpleExperiment,
    FixedParameter
)
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
from training_bo_holi_moli import train_and_validate, train_final, test
from utils import multi_omics_data
from utils.visualisation import save_auroc_plots, save_auroc_with_variance_plots

depth_lower = 1
depth_upper = 5
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
combination_list = [0, 1, 2, 3, 4]
batch_size_list = [32, 64]
epoch_lower = 10
epoch_upper = 50

drugs = {'Gemcitabine_tcga': 'TCGA',
         'Gemcitabine_pdx': 'PDX',
         'Cisplatin': 'TCGA',
         'Docetaxel': 'TCGA',
         'Erlotinib': 'PDX',
         'Cetuximab': 'PDX',
         'Paclitaxel': 'PDX',
         'EGFR': 'PDX'
         }


def bo_moli(search_iterations, sobol_iterations, load_checkpoint, experiment_name, combination,
            sampling_method, drug_name, extern_dataset_name, gpu_number):
    if sampling_method == 'sobol':
        sobol_iterations = 0

    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    result_path = Path('..', '..', '..', 'results', 'bayesian_optimisation', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)

    file_mode = 'a' if load_checkpoint else 'w'
    result_file = open(result_path / 'logs.txt', file_mode)
    checkpoint_path = result_path / 'checkpoint.json'
    result_file.write(f"Start for {drug_name}")
    print(f"Start for {drug_name}")

    data_path = Path('..', '..', '..', 'data')
    if drug_name == 'EGFR':
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_egfr_data(data_path)
    else:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name)
    moli_search_space = create_search_space(combination)

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    sobol = Models.SOBOL(moli_search_space, seed=random_seed)

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
        if load_checkpoint & checkpoint_path.exists():
            print("Load checkpoint")
            experiment = load_experiment(str(checkpoint_path))
            experiment.evaluation_function = lambda parameterization: train_and_validate(parameterization,
                                                                                         x_train_e, x_train_m,
                                                                                         x_train_c,
                                                                                         y_train,
                                                                                         device, pin_memory)
            print(f"Resuming after iteration {len(experiment.trials.values())}")

        else:
            experiment = SimpleExperiment(
                name="BO-MOLI",
                search_space=moli_search_space,
                evaluation_function=lambda parameterization: train_and_validate(parameterization,
                                                                                x_train_e, x_train_m,
                                                                                x_train_c,
                                                                                y_train, device, pin_memory),
                objective_name="auroc",
                minimize=False,
            )

            print(f"Running Sobol initialization trials...")
            for i in range(sobol_iterations):
                print(f"Running Sobol initialisation {i + 1}/{sobol_iterations}")
                experiment.new_trial(generator_run=sobol.gen(1))
                experiment.eval()
            save_experiment(experiment, str(checkpoint_path))

        for i in range(len(experiment.trials.values()), search_iterations):
            print(f"Running GP+EI optimization trial {i + 1} ...")

            # Reinitialize GP+EI model at each step with updated data.
            if sampling_method == 'gp':
                gp_ei = Models.BOTORCH(experiment=experiment, data=experiment.fetch_data())
                generator_run = gp_ei.gen(1)
            else:
                generator_run = sobol.gen(1)

            experiment.new_trial(generator_run=generator_run)
            experiment.eval()
            experiment.evaluation_function = lambda parameterization: train_and_validate(parameterization,
                                                                                         x_train_e, x_train_m,
                                                                                         x_train_c,
                                                                                         y_train,
                                                                                         device, pin_memory)
            save_experiment(experiment, str(checkpoint_path))

            if i % 10 == 0:
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
        aux_extern = test(model_extern, scaler_extern, extern_e, extern_m, extern_c, extern_r, device, pin_memory)

        result_file.write(f'\t\tBest {drug} validation Auroc = {max_objective}\n')
        result_file.write(f'\t\t{drug} test Auroc = {auc_test}\n')
        result_file.write(f'\t\t{drug} extern AUROC = {aux_extern}\n')
        objectives_list.append(objectives)
        max_objective_list.append(max_objective)
        test_auc_list.append(auc_test)
        extern_auc_list.append(aux_extern)

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


def calculate_mean_and_std_auc(result_dict, result_file, drug_name):
    result_file.write(f'\tMean Result for {drug_name}:\n')
    for result_name, result_value in result_dict.items():
        mean = np.mean(result_value)
        std = np.std(result_value)
        max_value = np.max(result_value)
        min_value = np.min(result_value)
        result_file.write(f'\t\t{result_name} mean: {mean}\n')
        result_file.write(f'\t\t{result_name} std: {std}\n')
        result_file.write(f'\t\t{result_name} max: {max_value}\n')
        result_file.write(f'\t\t{result_name} min: {min_value}\n')


def create_search_space(combination):
    if combination is None:
        combination_parameter = RangeParameter(name='combination', lower=combination_lower, upper=combination_upper,
                                               parameter_type=ParameterType.INT)
    else:
        combination_parameter = FixedParameter(name='combination', value=combination,
                                               parameter_type=ParameterType.INT)
    return SearchSpace(
        parameters=[
            ChoiceParameter(name='mini_batch', values=batch_size_list, parameter_type=ParameterType.INT),
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
            RangeParameter(name='margin', lower=margin_lower, upper=margin_upper, parameter_type=ParameterType.FLOAT),
        ]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_iterations', default=200, type=int)
    parser.add_argument('--sobol_iterations', default=20, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--combination', default=None, type=int)
    parser.add_argument('--sampling_method', default='gp', choices=['gp', 'sobol'])
    parser.add_argument('--gpu_number', type=int)
    args = parser.parse_args()

    for drug, extern_dataset in drugs.items():
        bo_moli(args.search_iterations, args.sobol_iterations, args.load_checkpoint, args.experiment_name,
                args.combination, args.sampling_method, drug, extern_dataset, args.gpu_number)
