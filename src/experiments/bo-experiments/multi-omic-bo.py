import argparse
from pathlib import Path
import numpy as np
from ax import optimize
import moli_egfr_bo
from utils import egfr_data
from utils.visualisation import save_auroc_plots

mini_batch_list = [8, 16, 32, 64]
dim_list = [1024, 512, 256, 128, 64, 32, 16, 8]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
epoch_list = [10, 20, 50, 15, 30, 40, 60, 70, 80, 90, 100]
drop_rate_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.01, 0.001, 0.1, 0.0001]
gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
combination_list = [0, 1, 2, 3]
depth_list = [1, 2, 3]


def bo_moli(search_iterations, run_test):
    data_path = Path('../../..', 'data')
    egfr_path = Path(data_path, 'EGFR_experiments_data')
    GDSCE, GDSCM, GDSCC, Y = egfr_data.load_train_data(egfr_path)

    best_parameters, values, experiment, model = optimize(
        parameters=[{"name": "mini_batch", "type": "choice", "values": mini_batch_list},
                    {"name": "h_dim1", "type": "choice", "values": dim_list},
                    {"name": "h_dim2", "type": "choice", "values": dim_list},
                    {"name": "h_dim3", "type": "choice", "values": dim_list},
                    {"name": "h_dim4", "type": "choice", "values": dim_list},
                    {"name": "h_dim5", "type": "choice", "values": dim_list},
                    {"name": "depth_1", "type": "choice", "values": depth_list},
                    {"name": "depth_2", "type": "choice", "values": depth_list},
                    {"name": "depth_3", "type": "choice", "values": depth_list},
                    {"name": "depth_4", "type": "choice", "values": depth_list},
                    {"name": "depth_5", "type": "choice", "values": depth_list},
                    {"name": "lr_e", "type": "choice", "values": learning_rate_list},
                    {"name": "lr_m", "type": "choice", "values": learning_rate_list},
                    {"name": "lr_c", "type": "choice", "values": learning_rate_list},
                    {"name": "lr_cl", "type": "choice", "values": learning_rate_list},
                    {"name": "dropout_rate_e", "type": "choice", "values": drop_rate_list},
                    {"name": "dropout_rate_m", "type": "choice", "values": drop_rate_list},
                    {"name": "dropout_rate_c", "type": "choice", "values": drop_rate_list},
                    {"name": "dropout_rate_clf", "type": "choice", "values": drop_rate_list},
                    {"name": "dropout_rate_middle", "type": "choice", "values": drop_rate_list},
                    {"name": "weight_decay", "type": "choice", "values": weight_decay_list},
                    {"name": "gamma", "type": "choice", "values": gamma_list},
                    {"name": "epochs", "type": "choice", "values": epoch_list},
                    {"name": "margin", "type": "choice", "values": margin_list, "value_type": "float"},
                    {"name": "combination", "type": "choice", "values": combination_list, "value_type": "int"},
                    ],
        evaluation_function=lambda parameterization: moli_egfr_bo.train_evaluate(parameterization,
                                                                                 GDSCE, GDSCM, GDSCC, Y),
        objective_name='accuracy',
        total_trials=search_iterations,
    )
    means, covariances = values
    print(best_parameters)
    print(means, covariances)

    result_path = Path('..', '..', '..', 'results', 'egfr')
    best_objectives = np.array([[trial.objective_mean * 100 for trial in experiment.trials.values()]])
    save_auroc_plots(best_objectives, result_path, 'bo')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--search_iterations', default=1, type=int)
    args = parser.parse_args()
    bo_moli(args.search_iterations, args.run_test)
