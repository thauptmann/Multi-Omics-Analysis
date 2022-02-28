import sys
from datetime import datetime
from pathlib import Path
import torch
import pickle
import time
import numpy as np
import yaml
from tqdm import tqdm
from ax import optimize
from ax.storage.json_store.save import save_experiment
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.experiment_utils import create_generation_strategy
from utils.input_arguments import get_cmd_arguments
from utils.searchspaces import create_holi_moli_search_space
from utils.choose_gpu import get_free_gpu
from training_bo_holi_moli import train_final, train_and_validate, reset_best_auroc
from utils import multi_omics_data
from utils.visualisation import save_auroc_plots, save_auroc_with_variance_plots
from utils.network_training_util import calculate_mean_and_std_auc, test

with open(Path('../../config/hyperparameter.yaml'), 'r') as stream:
    parameter = yaml.safe_load(stream)


def bo_moli(search_iterations, sobol_iterations, load_checkpoint, experiment_name, combination,
            sampling_method, drug_name, extern_dataset_name, gpu_number, small_search_space,
            deactivate_skip_bad_iterations, semi_hard_triplet, deactivate_elbow_method, architecture,
            noisy):
    device, pin_memory = create_device(gpu_number)

    result_path = Path('..', '..', '..', 'results', 'bayesian_optimisation', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)

    file_mode = 'a' if load_checkpoint else 'w'
    result_file = open(result_path / 'results.txt', file_mode)
    log_file = open(result_path / 'logs.txt', file_mode)
    checkpoint_path = result_path / 'checkpoint.json'
    log_file.write(f"Start for {drug_name}\n")

    data_path = Path('..', '..', '..', 'data')
    if deactivate_elbow_method:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name)
    else:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_drug_data_with_elbow(data_path, drug_name, extern_dataset_name)

    moli_search_space = create_holi_moli_search_space(combination, small_search_space, semi_hard_triplet)

    torch.manual_seed(parameter['random_seed'])
    np.random.seed(parameter['random_seed'])

    max_objective_list = []
    test_auc_list = []
    extern_auc_list = []
    test_auprc_list = []
    extern_auprc_list = []
    objectives_list = []
    now = datetime.now()
    result_file.write(f'Start experiment at {now}\n')
    log_file.write(f'Using {sampling_method}')
    skf = StratifiedKFold(n_splits=parameter['cv_splits'], random_state=parameter['random_seed'], shuffle=True)
    iteration = 0
    sobol_iterations = search_iterations if sampling_method == 'sobol' else sobol_iterations

    start_time = time.time()
    for train_index, test_index in tqdm(skf.split(gdsc_e, gdsc_r), total=skf.get_n_splits(), desc="Outer k-fold"):
        result_file.write(f'\t{iteration = }. \n')
        x_train_validate_e = gdsc_e[train_index]
        x_train_validate_m = gdsc_m[train_index]
        x_train_validate_c = gdsc_c[train_index]
        y_train_validate = gdsc_r[train_index]
        x_test_e = gdsc_e[test_index]
        x_test_m = gdsc_m[test_index]
        x_test_c = gdsc_c[test_index]
        y_test = gdsc_r[test_index]

        reset_best_auroc()
        evaluation_function = lambda parameterization: train_and_validate(parameterization,
                                                                          x_train_validate_e, x_train_validate_m,
                                                                          x_train_validate_c,
                                                                          y_train_validate, device, pin_memory,
                                                                          deactivate_skip_bad_iterations,
                                                                          semi_hard_triplet, architecture, noisy)
        generation_strategy = create_generation_strategy(sampling_method, sobol_iterations, parameter['random_seed'])

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

        model_final, scaler_final = train_final(best_parameters, x_train_validate_e, x_train_validate_m,
                                                x_train_validate_c, y_train_validate, device,
                                                pin_memory, semi_hard_triplet, architecture, noisy)
        auc_test, auprc_test = test(model_final, scaler_final, x_test_e, x_test_m, x_test_c, y_test, device)
        auc_extern, auprc_extern = test(model_final, scaler_final, extern_e, extern_m, extern_c, extern_r, device)

        result_file.write(f'\t\tBest {drug_name} validation Auroc = {max_objective}\n')
        objectives_list.append(objectives)
        max_objective_list.append(max_objective)
        test_auc_list.append(auc_test)
        extern_auc_list.append(auc_extern)
        test_auprc_list.append(auprc_test)
        extern_auprc_list.append(auprc_extern)

    print("Done!")
    end_time = time.time()
    result_file.write(f'\tMinutes needed: {round((end_time - start_time) / 60)}')
    result_dict = {
        'validation auroc': max_objective_list,
        'test auroc': test_auc_list,
        'test auprc': test_auprc_list,
        'extern auroc': extern_auc_list,
        'extern auprc': extern_auprc_list
    }
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)
    save_auroc_with_variance_plots(objectives_list, result_path, 'final', sobol_iterations)
    positive_extern = np.count_nonzero(extern_r == 1)
    negative_extern = np.count_nonzero(extern_r == 0)
    no_skill_prediction_auprc = positive_extern / (positive_extern + negative_extern)
    result_file.write(f'\n No skill predictor extern AUPRC: {no_skill_prediction_auprc} \n')

    result_file.write(f'\n test auroc list: {test_auc_list} \n')
    result_file.write(f'\n test auprc list: {test_auprc_list} \n')
    result_file.write(f'\n extern auroc list: {extern_auc_list} \n')
    result_file.write(f'\n extern auprc list: {extern_auprc_list} \n')
    result_file.write(f'\n validation auroc list: {max_objective_list} \n')

    result_file.close()


def create_device(gpu_number):
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
    return device, pin_memory


def extract_best_parameter(experiment):
    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    best_parameters = best_arm.parameters
    return best_parameters


if __name__ == '__main__':
    args = get_cmd_arguments()
    if args.drug == 'all':
        for drug, extern_dataset in parameter['drugs'].items():
            bo_moli(args.search_iterations, args.sobol_iterations, args.load_checkpoint, args.experiment_name,
                    args.combination, args.sampling_method, drug, extern_dataset, args.gpu_number,
                    args.small_search_space, args.deactivate_skip_bad_iterations, args.semi_hard_triplet,
                    args.deactivate_elbow_method, args.architecture, args.noisy)
    else:
        extern_dataset = parameter['drugs'][args.drug]
        bo_moli(args.search_iterations, args.sobol_iterations, args.load_checkpoint, args.experiment_name,
                args.combination, args.sampling_method, args.drug, extern_dataset, args.gpu_number,
                args.small_search_space, args.deactivate_skip_bad_iterations, args.semi_hard_triplet,
                args.deactivate_elbow_method, args.architecture, args.noisy)
