import argparse
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
from utils import egfr_data


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
    gdsc_e, gdsc_m, gdsc_c, gdsc_r, pdx_e_erlo, pdx_m_erlo, pdx_c_erlo, pdx_r_erlo, pdx_e_cet, dpx_m_cet, \
    pdx_c_cet, pdx_r_cet = egfr_data.load_data(data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--search_iterations', default=1, type=int)
    parser.add_argument('--sobol_iterations', default=5, type=int)
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--combination', default=None, type=int)
    parser.add_argument('--sampling_method', default='gp')
    args = parser.parse_args()
    bo_network_morphism_moli(args.search_iterations, args.run_test, args.sobol_iterations, args.load_checkpoint,
                             args.experiment_name)
