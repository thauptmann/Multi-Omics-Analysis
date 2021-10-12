import sys
from datetime import datetime
from pathlib import Path
import torch
import pickle
import time
import numpy as np
import argparse
from tqdm import tqdm

from ax import optimize

from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.storage.json_store.save import save_experiment

from sklearn.model_selection import StratifiedKFold
from ax.modelbridge.registry import Models

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
from pathlib import Path
from training_bo_holi_moli import train_final, train_and_validate, reset_best_auroc
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
    'Paclitaxel': 'PDX'
}

random_seed = 42


def bo_moli(search_iterations, sobol_iterations, load_checkpoint, experiment_name, combination,
            sampling_method, drug_name, extern_dataset_name, gpu_number, small_search_space,
            deactivate_skip_bad_iterations, triplet_selector_type):
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
    result_file = open(result_path / 'results.txt', file_mode)
    log_file = open(result_path / 'logs.txt', file_mode)
    checkpoint_path = result_path / 'checkpoint.json'
    log_file.write(f"Start for {drug_name}\n")

    data_path = Path('..', '..', '..', 'data')
    if drug_name == 'EGFR':
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_egfr_data(data_path)
    else:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name)
    moli_search_space = create_search_space(combination, small_search_space, triplet_selector_type)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    max_objective_list = []
    test_auc_list = []
    extern_auc_list = []
    test_auprc_list = []
    extern_auprc_list = []
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

        reset_best_auroc()
        evaluation_function = lambda parameterization: train_and_validate(parameterization,
                                                                          x_train_e, x_train_m,
                                                                          x_train_c,
                                                                          y_train, device, pin_memory,
                                                                          deactivate_skip_bad_iterations,
                                                                          triplet_selector_type)

        if sampling_method == 'gp':
            log_file.write('Using sobol+GPEI')
            generation_strategy = GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL,
                                   num_trials=sobol_iterations,
                                   max_parallelism=1,
                                   model_kwargs={"seed": random_seed}),
                    GenerationStep(
                        model=Models.BOTORCH,
                        max_parallelism=1,
                        num_trials=-1,
                    ),
                ],
                name="Sobol+GPEI"
            )
        elif sampling_method == 'saasbo':
            log_file.write('Using sobol+SAASBO')
            generation_strategy = GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL,
                                   num_trials=sobol_iterations,
                                   max_parallelism=1,
                                   model_kwargs={"seed": random_seed}),
                    GenerationStep(
                        model=Models.FULLYBAYESIAN,
                        num_trials=-1,
                        max_parallelism=1,
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
            log_file.write('Using only sobol')
            sobol_iterations = search_iterations
            generation_strategy = GenerationStrategy(
                steps=[
                    GenerationStep(model=Models.SOBOL, num_trials=-1),
                ],
                name="Sobol"
            )

        best_parameters, values, experiment, model = optimize(
            total_trials=search_iterations,
            experiment_name='Holi-Moli',
            objective_name='auroc',
            parameters=moli_search_space,
            evaluation_function=evaluation_function,
            minimize=False,
            generation_strategy=generation_strategy
        )

        # save results
        max_objective = max(np.array([trial.objective_mean for trial in experiment.trials.values()]))
        objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
        save_experiment(experiment, str(checkpoint_path))
        pickle.dump(objectives, open(result_path / 'objectives', "wb"))
        pickle.dump(best_parameters, open(result_path / 'best_parameters', "wb"))
        save_auroc_plots(objectives, result_path, iteration, sobol_iterations)

        iteration += 1

        result_file.write(f'\t\t{str(best_parameters) = }\n')

        model_final, scaler_final = train_final(best_parameters, x_train_e, x_train_m, x_train_c, y_train, device,
                                                pin_memory)
        auc_test, auprc_test = test(model_final, scaler_final, x_test_e, x_test_m, x_test_c, y_test, device, pin_memory)
        auc_extern, auprc_extern = test(model_final, scaler_final, extern_e, extern_m, extern_c, extern_r, device,
                                        pin_memory)

        result_file.write(f'\t\tBest {drug} validation Auroc = {max_objective}\n')
        result_file.write(f'\t\t{drug} test Auroc = {auc_test}\n')
        result_file.write(f'\t\t{drug} test AUPRC = {auprc_test}\n')
        result_file.write(f'\t\t{drug} extern AUROC = {auc_extern}\n')
        result_file.write(f'\t\t{drug} extern AUPRC = {auprc_extern}\n')
        objectives_list.append(objectives)
        max_objective_list.append(max_objective)
        test_auc_list.append(auc_test)
        extern_auc_list.append(auc_extern)
        test_auprc_list.append(auprc_test)
        extern_auprc_list.append(auprc_extern)

    print("Done!")

    result_dict = {
        'validation auroc': max_objective_list,
        'test auroc': test_auc_list,
        'test auprc': test_auprc_list,
        'extern auroc': extern_auc_list,
        'extern auprc': extern_auprc_list
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


def create_search_space(combination, small_search_space, triplet_selector_type):
    if combination is None:
        combination_parameter = {'name': 'combination', "bounds": [combination_lower, combination_upper],
                                 "value_type": "int", 'type': 'range'}
    else:
        combination_parameter = {'name': 'combination', 'value': combination, 'type': 'fixed', "value_type": "int"}

    if triplet_selector_type == 'none':
        gamma = {'name': 'gamma', "value": 0, "value_type": "float", 'type': 'fixed'}
        margin = {'name': 'margin', "value": 0, "value_type": "float", 'type': 'fixed'}
    else:
        gamma = {'name': 'gamma', "bounds": [gamma_lower, gamma_upper], "value_type": "float", 'type': 'range'}
        margin = {'name': 'margin', "bounds": [margin_lower, margin_upper], "value_type": "float", 'type': 'range'}

    if combination is None and not small_search_space:
        search_space = [
            {'name': 'mini_batch', 'bounds': [batch_size_lower, batch_size_upper], 'value_type': 'int',
             'type': 'range'},
            {'name': 'h_dim1', "bounds": [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
            {'name': "h_dim2", "bounds": [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
            {'name': "h_dim3", "bounds": [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
            {'name': "h_dim4", "bounds": [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
            {'name': "h_dim5", "bounds": [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
            {'name': "depth_1", "bounds": [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
            {'name': "depth_2", "bounds": [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
            {'name': "depth_3", "bounds": [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
            {'name': "depth_4", "bounds": [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
            {'name': "depth_5", "bounds": [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
            {'name': "lr_e", "bounds": [learning_rate_lower, learning_rate_upper], "value_type": "float",
             'log_scale': True, 'type': 'range'},
            {'name': "lr_m", "bounds": [learning_rate_lower, learning_rate_upper], "value_type": "float",
             'log_scale': True, 'type': 'range'},
            {'name': "lr_c", "bounds": [learning_rate_lower, learning_rate_upper], "value_type": "float",
             'log_scale': True, 'type': 'range'},
            {'name': "lr_cl", "bounds": [learning_rate_lower, learning_rate_upper], "value_type": "float",
             'log_scale': True, 'type': 'range'},
            {'name': "lr_middle", "bounds": [learning_rate_lower, learning_rate_upper],
             "value_type": "float", 'log_scale': True, 'type': 'range'},
            {'name': "dropout_rate_e", "bounds": [drop_rate_lower, drop_rate_upper], "value_type": "float",
             'type': 'range'},
            {'name': "dropout_rate_m", "bounds": [drop_rate_lower, drop_rate_upper], "value_type": "float",
             'type': 'range'},
            {'name': "dropout_rate_c", "bounds": [drop_rate_lower, drop_rate_upper], "value_type": "float",
             'type': 'range'},
            {'name': "dropout_rate_clf", "bounds": [drop_rate_lower, drop_rate_upper], "value_type": "float",
             'type': 'range'},
            {'name': "dropout_rate_middle", "bounds": [drop_rate_lower, drop_rate_upper], "value_type": "float",
             'type': 'range'},
            {'name': 'weight_decay', "bounds": [weight_decay_lower, weight_decay_upper], 'log_scale': True,
             "value_type": "float", 'type': 'range'},
            gamma,
            margin,
            {'name': 'epochs', "bounds": [epoch_lower, epoch_upper], "value_type": "int", 'type': 'range'},
            combination_parameter
        ]

    # moli
    elif combination is not None and not small_search_space:
        search_space = [{'name': 'mini_batch', 'bounds': [batch_size_lower, batch_size_upper],
                         'type': 'range', 'value_type': 'int', 'log_scale': True},
                        {'name': "h_dim1", 'bounds': [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
                        {'name': "h_dim2", 'bounds': [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
                        {'name': "h_dim3", 'bounds': [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
                        {'name': "h_dim5", 'bounds': [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
                        {'name': "depth_1", 'bounds': [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
                        {'name': "depth_2", 'bounds': [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
                        {'name': "depth_3", 'bounds': [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
                        {'name': "depth_5", 'bounds': [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
                        {'name': "lr_e", 'bounds': [learning_rate_lower, learning_rate_upper],
                         "value_type": "float", 'log_scale': True, 'type': 'range'},
                        {'name': "lr_m", 'bounds': [learning_rate_lower, learning_rate_upper],
                         "value_type": "float", 'log_scale': True, 'type': 'range'},
                        {'name': "lr_c", 'bounds': [learning_rate_lower, learning_rate_upper],
                         "value_type": "float", 'log_scale': True, 'type': 'range'},
                        {'name': "lr_cl", 'bounds': [learning_rate_lower, learning_rate_upper],
                         "value_type": "float", 'log_scale': True, 'type': 'range'},
                        {'name': "dropout_rate_e", 'bounds': [drop_rate_lower, drop_rate_upper],
                         "value_type": "float", 'type': 'range'},
                        {'name': "dropout_rate_m", 'bounds': [drop_rate_lower, drop_rate_upper],
                         "value_type": "float", 'type': 'range'},
                        {'name': "dropout_rate_c", 'bounds': [drop_rate_lower, drop_rate_upper],
                         "value_type": "float", 'type': 'range'},
                        {'name': "dropout_rate_clf", 'bounds': [drop_rate_lower, drop_rate_upper],
                         "value_type": "float", 'type': 'range'},
                        {'name': 'weight_decay', 'bounds': [weight_decay_lower, weight_decay_upper], 'log_scale': True,
                         "value_type": "float", 'type': 'range'},
                       gamma,
                        margin,
                        {'name': 'epochs', 'bounds': [epoch_lower, epoch_upper], "value_type": "int", 'type': 'range'},
                        combination_parameter
                        ]
    else:
        search_space = [
            {'name': 'mini_batch', 'bounds': [batch_size_lower, batch_size_upper], "value_type": "int",
             'type': 'range'},
            {'name': "h_dim1", 'bounds': [dim_lower, dim_upper], "value_type": "int", 'type': 'range'},
            {'name': "depth_1", 'bounds': [depth_lower, depth_upper], "value_type": "int", 'type': 'range'},
            {'name': "lr_e", 'bounds': [learning_rate_lower, learning_rate_upper],
             "value_type": "float", 'log_scale': True, 'type': 'range'},
            {'name': "dropout_rate_e", 'bounds': [drop_rate_lower, drop_rate_upper], "value_type": "float",
             'type': 'range'},
            {'name': 'weight_decay', 'bounds': [weight_decay_lower, weight_decay_upper], 'log_scale': True,
             "value_type": "float", 'type': 'range'},
            gamma,
            {'name': 'epochs', 'bounds': [epoch_lower, epoch_upper], "value_type": "int", 'type': 'range'},
            combination_parameter,
            margin
        ]
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
    parser.add_argument('--deactivate_skip_bad_iterations', default=False, action='store_true')
    parser.add_argument('--drug', default='all', choices=['Gemcitabine_tcga', 'Gemcitabine_pdx', 'Cisplatin',
                                                          'Docetaxel', 'Erlotinib', 'Cetuximab', 'Paclitaxel'])
    parser.add_argument('--triplet_selector_type', default='all', choices=['all', 'hardest', 'random', 'semi_hard',
                                                                           'none'])
    args = parser.parse_args()

    if args.drug == 'all':
        for drug, extern_dataset in drugs.items():
            bo_moli(args.search_iterations, args.sobol_iterations, args.load_checkpoint, args.experiment_name,
                    args.combination, args.sampling_method, drug, extern_dataset, args.gpu_number,
                    args.small_search_space, args.deactivate_skip_bad_iterations, args.triplet_selector_type)
    else:
        drug, extern_dataset = drugs[args.drug]
        bo_moli(args.search_iterations, args.sobol_iterations, args.load_checkpoint, args.experiment_name,
                args.combination, args.sampling_method, drug, extern_dataset, args.gpu_number,
                args.small_search_space, args.deactivate_skip_bad_iterations, args.triplet_selector_type)
