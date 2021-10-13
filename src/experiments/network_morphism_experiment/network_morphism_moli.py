import argparse
import sys
import time
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from datetime import datetime
from utils import multi_omics_data
from utils.visualisation import save_auroc_plots, save_auroc_with_variance_plots

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.network_morphism_experiment.autokeras.nn.metric import Auroc
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.choose_gpu import get_free_gpu
from utils.network_training_util import BceWithTripletsToss, calculate_mean_and_std_auc, test, test_ensemble, \
    get_triplet_selector, get_loss_fn, create_data_loader
from autokeras.search import BayesianSearcher
from autokeras.nn.generator import DenseNetGenerator

drugs = {'Gemcitabine_tcga': 'TCGA',
         'Gemcitabine_pdx': 'PDX',
         'Cisplatin': 'TCGA',
         'Docetaxel': 'TCGA',
         'Erlotinib': 'PDX',
         'Cetuximab': 'PDX',
         'Paclitaxel': 'PDX'
         }
MAX_EPOCH = 100


def bo_network_morphism_moli(search_iterations, experiment_name, drug_name, extern_dataset_name, gpu_number,
                             triplet_selector_type):
    gamma = 0.2
    margin = 1
    batch_size = 32

    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

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

    result_path = Path('..', '..', '..', 'results', 'network_morphism', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)

    result_file = open(result_path / 'logs.txt', 'w')
    result_file.write(f"Start for {drug_name}\n")
    print(f"Start for {drug_name}")

    data_path = Path('..', '..', '..', 'data')
    if drug_name == 'EGFR':
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_egfr_data(data_path)
    else:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name)

    test_auc_list = []
    extern_auc_list = []
    objectives_list = []
    max_objective_list = []
    path = Path('tmp')
    auroc_metric = Auroc
    triplet_selector = get_triplet_selector(margin, triplet_selector_type)
    loss_fn = get_loss_fn(margin, gamma, triplet_selector)
    path.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    result_file.write(f'Start experiment at {now}\n')
    cv_splits = 5
    skf = StratifiedKFold(n_splits=cv_splits, random_state=random_seed, shuffle=True)

    start_time = time.time()
    iteration = 0
    input_shape_list = [gdsc_e.shape[-1], gdsc_m.shape[-1], gdsc_c.shape[-1]]
    for index, test_index in tqdm(skf.split(gdsc_e, gdsc_r), total=skf.get_n_splits(), desc="Outer k-fold"):
        x_e = gdsc_e[index]
        x_m = gdsc_m[index]
        x_c = gdsc_c[index]
        y = gdsc_r[index]

        x_test_e = gdsc_e[test_index]
        x_test_m = gdsc_m[test_index]
        x_test_c = gdsc_c[test_index]
        y_test = gdsc_r[test_index]

        x_train_e, x_validate_e, x_train_m, x_validate_m, x_train_c, x_validate_c, y_train, y_validate \
            = train_test_split((x_e, x_m, x_c, y), stratify=y, test_size=0.20, random_state=random_seed)
        best_model_list = []

        scaler_gdsc = StandardScaler()
        x_train_e = scaler_gdsc.fit_transform(x_train_e)
        x_validate_e = scaler_gdsc.transform(x_validate_e)
        # Initialisation
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        train_loader = create_data_loader(torch.FloatTensor(x_train_e),
                                          torch.FloatTensor(x_train_m),
                                          torch.FloatTensor(x_train_c),
                                          torch.FloatTensor(y_train), batch_size, True,
                                          pin_memory, sampler=sampler)
        validation_loader = create_data_loader(torch.FloatTensor(x_validate_e),
                                               torch.FloatTensor(x_validate_m),
                                               torch.FloatTensor(x_validate_c),
                                               torch.FloatTensor(y_validate), batch_size * 4, False,
                                               pin_memory)

        dense_generator_list = [DenseNetGenerator]
        searcher = BayesianSearcher(1, input_shape_list, path, auroc_metric, loss_fn, verbose=True,
                                    generators=dense_generator_list, skip_conn=False)
        for _ in range(search_iterations):
            searcher.search(train_loader, validation_loader)

        best_model = searcher.load_best_model()
        best_model_list.append(best_model)
        max_objective = searcher.get_metric_value_by_id(searcher.get_best_model_id())

        objectives = []
        for iteration_id in range(search_iterations):
            objectives.append(searcher.get_metric_value_by_id(iteration_id))

        save_auroc_plots(objectives, result_path, iteration)

        # Test
        auc_test, auprc_test = test(best_model, scaler_gdsc, x_test_e, x_test_m, x_test_c, y_test, device)
        auc_extern, auprc_extern = test(best_model, scaler_gdsc, extern_e, extern_m, extern_c, extern_r, device)

        result_file.write(f'\t\tBest {drug} validation Auroc = {max_objective}\n')
        result_file.write(f'\t\t{drug} test Auroc = {auc_test}\n')
        result_file.write(f'\t\t{drug} extern AUROC = {auc_extern}\n')
        objectives_list.append(objectives)
        max_objective_list.append(max_objective)
        test_auc_list.append(auc_test)
        extern_auc_list.append(auc_extern)

        iteration += 1

    print("Done!")

    result_dict = {
        'validation': max_objective_list,
        'test': test_auc_list,
        'extern': extern_auc_list
    }
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)
    save_auroc_with_variance_plots(objectives_list, result_path, 'final')

    end_time = time.time()
    result_file.write(f'\tMinutes needed: {round((end_time - start_time) / 60)}')
    result_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_iterations', default=200, type=int)
    parser.add_argument('--sobol_iterations', default=20, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--gpu_number', type=int)
    parser.add_argument('--triplet_selector_type', default='all', choices=['all', 'hardest', 'random', 'semi_hard'])
    args = parser.parse_args()
    for drug, extern_dataset in drugs.items():
        bo_network_morphism_moli(args.search_iterations, args.experiment_name, drug, extern_dataset, args.gpu_number,
                                 args.triplet_selector_type)
