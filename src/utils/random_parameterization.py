import random

batch_size_list = [8, 16, 32, 64]
dim_list = [16, 32, 64, 128, 256, 512, 1024]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
epoch_list = [10, 20, 50, 30, 40, 60, 70, 80]
drop_rate_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.1, 0.01, 0.001, 0.0001]
gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def create_random_parameterization():
    random_parameterization = {
        'batch_size': random.choice(batch_size_list),
        'h_dim1': random.choice(dim_list),
        'h_dim2': random.choice(dim_list),
        'h_dim3': random.choice(dim_list),
        'lr_e': random.choice(learning_rate_list),
        'lr_m': random.choice(learning_rate_list),
        'lr_c': random.choice(learning_rate_list),
        'lr_cl': random.choice(learning_rate_list),
        'dropout_rate_e': random.choice(drop_rate_list),
        'dropout_rate_m': random.choice(drop_rate_list),
        'dropout_rate_c': random.choice(drop_rate_list),
        'weight_decay': random.choice(weight_decay_list),
        'dropout_rate_clf': random.choice(drop_rate_list),
        'gamma': random.choice(gamma_list),
        'epochs': random.choice(epoch_list),
        'margin': random.choice(margin_list)
    }
    return random_parameterization
