import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
from utils import egfr_data
from autokeras.search import BayesianSearcher


def bo_network_morphism_moli(search_iterations, run_test, sobol_iterations, load_checkpoint, experiment_name):
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")

    result_path = Path('..', '..', '..', 'results', 'egfr', 'network_morphism', experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = open(result_path / 'logs.txt', "a")
    checkpoint_path = result_path / 'checkpoint.json'

    data_path = Path('..', '..', '..', 'data')
    gdsc_e, gdsc_m, gdsc_c, gdsc_r = egfr_data.load_data(data_path)
    stratified_shuffle_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
    train_index, test_index = next(stratified_shuffle_splitter.split(gdsc_e, gdsc_r))
    x_train_e = gdsc_e[train_index]
    x_train_m = gdsc_m[train_index]
    x_train_c = gdsc_c[train_index]
    y_train = gdsc_r[train_index]
    x_test_e = gdsc_e[test_index]
    x_test_m = gdsc_m[test_index]
    x_test_c = gdsc_c[test_index]
    y_test = gdsc_r[test_index]
    validation_aurocs = []
    best_validation = 0.5
    _, ie_dim = x_train_e.shape
    _, im_dim = x_train_m.shape
    _, ic_dim = x_train_c.shape
    searcher = BayesianSearcher(1, (ie_dim, im_dim, ic_dim), skip_conn=False)
    for i in range(search_iterations):
        model = get_next_model()
        validation_auroc = train_model(model, x_train_e, x_train_m, x_train_c, y_train, early_stopping=True)
        validation_aurocs.append(validation_auroc)

        if validation_auroc > best_validation:
            best_validation = validation_auroc
            best_model = model
    test_model(model, x_train_e, x_train_m, x_train_c, y_train, x_test_e, x_test_m, x_test_c, y_test)


def train_model(model, x_train_e, x_train_m, x_train_c, y_train, early_stopping):
    pass


def test_model(model, x_train_e, x_train_m, x_train_c, y_train, x_train_e1, x_train_m1, x_train_c1, y_train1):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--search_iterations', default=200, type=int)
    parser.add_argument('--sobol_iterations', default=20, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    args = parser.parse_args()
    bo_network_morphism_moli(args.search_iterations, args.run_test, args.sobol_iterations, args.load_checkpoint,
                             args.experiment_name)
