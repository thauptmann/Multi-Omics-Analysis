import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from utils import multi_omics_data
from utils.network_training_util import test, test_ensemble, calculate_mean_and_std_auc

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
from training_bo_holi_moli import train_final

drugs = {
    'Gemcitabine_tcga': 'TCGA',
    'Gemcitabine_pdx': 'PDX',
    'Cisplatin': 'TCGA',
    'Docetaxel': 'TCGA',
    'Erlotinib': 'PDX',
    'Cetuximab': 'PDX',
    'Paclitaxel': 'PDX',
}

random_seed = 42


def train_and_validate_ensemble(experiment_name, gpu_number, drug_name, extern_dataset_name,
                                best_parameters_list):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    cv_splits = 5
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

    result_path = Path('..', '..', '..', 'results', 'bayesian_optimisation', 'ensemble', experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)

    result_file = open(result_path / 'logs.txt', 'w')
    result_file.write(f"Start for {drug_name}\n")
    print(f"Start for {drug_name}")

    data_path = Path('..', '..', '..', 'data')
    gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
        = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name)

    iteration = 0
    auc_list = []
    auc_extern_list = []
    skf = StratifiedKFold(n_splits=cv_splits, random_state=random_seed, shuffle=True)
    for train_index, test_index in tqdm(skf.split(gdsc_e, gdsc_r), total=skf.get_n_splits(), desc=" Outer k-fold"):
        x_train_e = gdsc_e[train_index]
        x_train_m = gdsc_m[train_index]
        x_train_c = gdsc_c[train_index]
        y_train = gdsc_r[train_index]
        x_test_e = gdsc_e[test_index]
        x_test_m = gdsc_m[test_index]
        x_test_c = gdsc_c[test_index]
        y_test = gdsc_r[test_index]

        model_test, scaler_test = train_final(best_parameters_list[iteration], x_train_e, x_train_m, x_train_c,
                                              y_train, device,
                                              pin_memory)
        auc_test = test(model_test, scaler_test, x_test_e, x_test_m, x_test_c, y_test, device, pin_memory)
        model_extern, scaler_extern = train_final(best_parameters_list[iteration], gdsc_e, gdsc_m, gdsc_c, gdsc_r,
                                                  device, pin_memory)
        auc_extern = test(model_extern, scaler_extern, extern_e, extern_m, extern_c, extern_r, device, pin_memory)
        auc_list.append(auc_test)
        auc_extern_list.append(auc_extern)

        iteration += 1

    model_list = []
    scaler_list = []
    #for best_parameters in best_parameters_list:

       # model_extern, scaler_extern = train_final(best_parameters, gdsc_e, gdsc_m, gdsc_c, gdsc_r, device, pin_memory)
       # y_true_list, prediction_lists = test_ensemble(model_extern, scaler_extern, extern_e, extern_m, extern_c,
         #                                             extern_r, device, pin_memory)
    # todo soft vote
    hard_voting_auroc = 0

    # todo hard vote
    soft_voting_auroc = 0

    # todo weighted vote
    weighted_voting_auroc = 0
    print(f'{auc_list =}\n')
    print(f'{auc_extern_list =}\n')
    result_dict = {
        'test': auc_list,
        'extern': auc_extern_list
    }
    calculate_mean_and_std_auctmp(result_dict, result_file, drug_name)

    # print(f'\t\t{drug_name} test Auroc = {auc_test}\n')
    # print(f'\t\t{drug_name} extern AUROC = {auc_extern}\n')


def calculate_mean_and_std_auctmp(result_dict, result_file, drug_name):
    result_file.write(f'\tMean Result for {drug_name}:\n')
    for result_name, result_value in result_dict.items():
        mean = np.mean(result_value)
        std = np.std(result_value)
        max_value = np.max(result_value)
        min_value = np.min(result_value)
        print(f'\t\t{result_name} mean: {mean}\n')
        print(f'\t\t{result_name} std: {std}\n')
        print(f'\t\t{result_name} max: {max_value}\n')
        print(f'\t\t{result_name} min: {min_value}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--gpu_number', type=int)
    parser.add_argument('--method_name', required=True)
    args = parser.parse_args()

    p = Path('../results')
    logfile_name = 'logs.txt'
    cv_result_path = Path('..', '..', '..', 'results', 'bayesian_optimisation')

    drug_paths = [x for x in cv_result_path.iterdir()]
    for drug_path in drug_paths:
        best_parameters_list = []
        drug_name = drug_path.stem
        log_path = drug_path / args.method_name / logfile_name
        if log_path.is_file():
            with open(log_path, 'r') as log_file:
                test_auroc = []
                extern_auroc = []
                for line in log_file:
                    if 'best_parameters' in line:
                        best_parameter_string = line.split("=")[-1].strip()
                        # strip the string literals
                        best_parameters_list.append(eval(best_parameter_string[1:-1]))

        train_and_validate_ensemble(args.experiment_name, args.gpu_number, drug_name, drugs[drug_name],
                                    best_parameters_list)
