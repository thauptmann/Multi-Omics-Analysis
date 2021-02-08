import torch
from torch import nn
from ax import ChoiceParameter, ParameterType, SearchSpace, Experiment

mini_batch_list = [8, 16, 32, 64]
dim_list = [1024, 512, 256, 128, 64, 32, 16]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
epoch_list = [20, 50, 10, 15, 30, 40, 60, 70, 80, 90, 100]
drop_rate_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.01, 0.001, 0.1, 0.0001]
gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

mini_batch = ChoiceParameter(name="mini_batch", parameter_type=ParameterType.INT, values=mini_batch_list)
h_dim1 = ChoiceParameter(name="h_dim1", parameter_type=ParameterType.INT, values=dim_list)
h_dim2 = ChoiceParameter(name="h_dim2", parameter_type=ParameterType.INT, values=dim_list)
h_dim3 = ChoiceParameter(name="h_dim3", parameter_type=ParameterType.INT, values=dim_list)
lr_e = ChoiceParameter(name="lr_e", parameter_type=ParameterType.FLOAT, values=learning_rate_list)
lr_m = ChoiceParameter(name="lr_m", parameter_type=ParameterType.FLOAT, values=learning_rate_list)
lr_c = ChoiceParameter(name="lr_c", parameter_type=ParameterType.FLOAT, values=learning_rate_list)
lr_cl = ChoiceParameter(name="lr_cl", parameter_type=ParameterType.FLOAT, values=learning_rate_list)
dropout_rate_e = ChoiceParameter(name="dropout_rate_e", parameter_type=ParameterType.FLOAT, values=drop_rate_list)
dropout_rate_m = ChoiceParameter(name="dropout_rate_m", parameter_type=ParameterType.FLOAT, values=drop_rate_list)
dropout_rate_c = ChoiceParameter(name="dropout_rate_c", parameter_type=ParameterType.FLOAT, values=drop_rate_list)
dropout_rate_clf = ChoiceParameter(name="dropout_rate_clf", parameter_type=ParameterType.FLOAT, values=drop_rate_list)
weight_decay = ChoiceParameter(name="weight_decay", parameter_type=ParameterType.FLOAT, values=weight_decay_list)
gamma = ChoiceParameter(name="gamma", parameter_type=ParameterType.FLOAT, values=gamma_list)
epochs = ChoiceParameter(name="epochs", parameter_type=ParameterType.FLOAT, values=epoch_list)
margin = ChoiceParameter(name="margin", parameter_type=ParameterType.FLOAT, values=margin_list)

# reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def bo_moli():
    search_space = SearchSpace(
        parameters=[mini_batch, h_dim1, h_dim2, h_dim3, lr_e, lr_m, lr_c, lr_cl, dropout_rate_e, dropout_rate_m,
                    dropout_rate_c, dropout_rate_clf, weight_decay, gamma, epochs, margin],
    )

    experiment = Experiment(
        name="auto-moli",
        search_space=search_space,
    )


if __name__ == '__main__':
    bo_moli()
