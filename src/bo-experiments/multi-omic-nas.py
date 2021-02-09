import numpy as np
import torch
from ax import ChoiceParameter, ParameterType, optimize
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render

import moli_egfr_bo

#mini_batch_list = [8, 16, 32, 64]
mini_batch_list = [32, 64]
dim_list = [1024, 512, 256, 128, 64, 32, 16]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
# epoch_list = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]

epoch_list = [20, 50, 10, 15, 5]
drop_rate_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.01, 0.001, 0.1, 0.0001]
gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def bo_moli():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    best_parameters, values, experiment, model = optimize(
        parameters=[{"name": "mini_batch", "type": "choice", "values": mini_batch_list},
                    {"name": "h_dim1", "type": "choice", "values": dim_list},
                    {"name": "h_dim2", "type": "choice", "values": dim_list},
                    {"name": "h_dim3", "type": "choice", "values": dim_list},
                    {"name": "lr_e", "type": "choice", "values": learning_rate_list},
                    {"name": "lr_m", "type": "choice", "values": learning_rate_list},
                    {"name": "lr_c", "type": "choice", "values": learning_rate_list},
                    {"name": "lr_cl", "type": "choice", "values": learning_rate_list},
                    {"name": "dropout_rate_e", "type": "choice", "values": drop_rate_list},
                    {"name": "dropout_rate_m", "type": "choice", "values": drop_rate_list},
                    {"name": "dropout_rate_c", "type": "choice", "values": drop_rate_list},
                    {"name": "weight_decay", "type": "choice", "values": weight_decay_list},
                    {"name": "gamma", "type": "choice", "values": gamma_list},
                    {"name": "epochs", "type": "choice", "values": epoch_list},
                    {"name": "margin", "type": "choice", "values": margin_list, "value_type": "float"}
                    ],
        evaluation_function=moli_egfr_bo.train_evaluate,
        objective_name='accuracy',
        total_trials=5,
    )
    means, covariances = values
    print(best_parameters)
    print(means, covariances)

    # `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple
    # optimization runs, so we wrap out best objectives array in another array.
    best_objectives = np.array([[trial.objective_mean * 100 for trial in experiment.trials.values()]])
    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )
    render(best_objective_plot)
    data = best_objective_plot[0]['data']
    lay = best_objective_plot[0]['layout']

    import plotly.graph_objects as go
    fig = {
        "data": data,
        "layout": lay,
    }
    go.Figure(fig).write_image("test.pdf")


if __name__ == '__main__':
    bo_moli()
