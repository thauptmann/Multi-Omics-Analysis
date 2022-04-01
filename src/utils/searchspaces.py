from pathlib import Path
import yaml

with open(Path('../../config/hyperparameter.yaml'), 'r') as stream:
    parameter = yaml.safe_load(stream)


def create_holi_moli_search_space(combination):
    if combination is None:
        combination_parameter = {'name': 'combination', "values": parameter['combination_choices'],
                                 "value_type": "int", 'type': 'choice'}
    else:
        combination_parameter = {'name': 'combination', 'value': combination, 'type': 'fixed', "value_type": "int"}

    gamma = {'name': 'gamma', "values": parameter['gamma_choices'], "value_type": "float", 'type': 'choice'}
    margin = {'name': 'margin', "values": parameter['margin_choices'], "value_type": "float", 'type': 'choice'}

    if combination is None:
        search_space = [
            {'name': 'mini_batch', 'values': parameter['all_triplet_batch_size_choices'], 'value_type': 'int',
             'type': 'choice'},
            {'name': 'h_dim1', "values": parameter['dim_choice'], "value_type": "int", 'type': 'choice'},
            {'name': "h_dim2", "values": parameter['dim_choice'], "value_type": "int", 'type': 'choice'},
            {'name': "h_dim3", "values": parameter['dim_choice'], "value_type": "int", 'type': 'choice'},
            {'name': "h_dim4", "values": parameter['dim_choice'], "value_type": "int", 'type': 'choice'},
            {'name': "lr_e", "values": parameter['learning_rate_choices'], "value_type": "float", 'log_scale': True,
             'type': 'choice'},
            {'name': "lr_m", "values": parameter['learning_rate_choices'], "value_type": "float",
             'log_scale': True, 'type': 'choice'},
            {'name': "lr_c", "values": parameter['learning_rate_choices'], "value_type": "float",
             'log_scale': True, 'type': 'choice'},
            {'name': "lr_cl", "values": parameter['learning_rate_choices'], "value_type": "float", 'log_scale': True,
             'type': 'choice'},
            {'name': "lr_middle", "values": parameter['learning_rate_choices'], "value_type": "float",
             'log_scale': True, 'type': 'choice'},
            {'name': "dropout_rate_e", "values": parameter['drop_rate_choices'], "value_type": "float",
             'type': 'choice'},
            {'name': "dropout_rate_m", "values": parameter['drop_rate_choices'], "value_type": "float",
             'type': 'choice'},
            {'name': "dropout_rate_c", "values": parameter['drop_rate_choices'], "value_type": "float",
             'type': 'choice'},
            {'name': "dropout_rate_clf", "values": parameter['drop_rate_choices'], "value_type": "float",
             'type': 'choice'},
            {'name': "dropout_rate_middle", "values": parameter['drop_rate_choices'], "value_type": "float",
             'type': 'choice'},
            {'name': 'weight_decay', "values": parameter['weight_decay_choices'], 'log_scale': True,
             "value_type": "float", 'type': 'choice'},
            gamma,
            margin,
            {'name': 'epochs', "bounds": [parameter['epoch_lower'], parameter['epoch_upper']],
             "value_type": "int", 'type': 'range'},
            combination_parameter
        ]

    # moli
    else:
        search_space = [{'name': 'mini_batch', 'values': parameter['all_triplet_batch_size_choices'], 'type': 'choice', 'value_type': 'int'},
                        {'name': "h_dim1", 'values': parameter['dim_choice'], "value_type": "int", 'type': 'choice'},
                        {'name': "h_dim2", 'values': parameter['dim_choice'], "value_type": "int", 'type': 'choice'},
                        {'name': "h_dim3", 'values': parameter['dim_choice'], "value_type": "int", 'type': 'choice'},
                        {'name': "lr_e", 'values': parameter['learning_rate_choices'],
                         "value_type": "float", 'log_scale': True, 'type': 'choice'},
                        {'name': "lr_m", 'values': parameter['learning_rate_choices'],
                         "value_type": "float", 'log_scale': True,
                         'type': 'choice'},
                        {'name': "lr_c", 'values': parameter['learning_rate_choices'], "value_type": "float",
                         'type': 'choice'},
                        {'name': "lr_cl", 'values': parameter['learning_rate_choices'], "value_type": "float",
                         'type': 'choice'},
                        {'name': "dropout_rate_e", 'values': parameter['drop_rate_choices'],
                         "value_type": "float", 'type': 'choice'},
                        {'name': "dropout_rate_m", 'values': parameter['drop_rate_choices'],
                         "value_type": "float", 'type': 'choice'},
                        {'name': "dropout_rate_c", 'values': parameter['drop_rate_choices'],
                         "value_type": "float", 'type': 'choice'},
                        {'name': "dropout_rate_clf", 'values': parameter['drop_rate_choices'], "value_type": "float",
                         'type': 'choice'},
                        {'name': 'weight_decay', 'values': parameter['weight_decay_choices'], 'log_scale': True,
                         "value_type": "float", 'type': 'choice'},
                        gamma,
                        margin,
                        {'name': 'epochs', 'bounds': [parameter['epoch_lower'], parameter['epoch_upper']],
                         "value_type": "int", 'type': 'range'},
                        combination_parameter
                        ]
    return search_space


def create_super_felt_search_space():
    search_space = [{'name': 'encoder_dropout', 'values': parameter['drop_rate_choices'], 'type': 'choice',
                     'value_type': 'float'},
                    {'name': 'classifier_dropout', 'values': parameter['drop_rate_choices'], 'type': 'choice',
                     'value_type': 'float'},
                    {'name': 'classifier_weight_decay', 'values': parameter['weight_decay_choices'], 'type': 'choice',
                     'value_type': 'float'},
                    {'name': 'encoder_weight_decay', 'values': parameter['weight_decay_choices'], 'type': 'choice',
                     'value_type': 'float'},
                    {'name': 'learning_rate_e', 'values': parameter['learning_rate_choices'], 'type': 'choice',
                     'value_type': 'float'},
                    {'name': 'learning_rate_m', 'values': parameter['learning_rate_choices'], 'type': 'choice',
                     'value_type': 'float'},
                    {'name': 'learning_rate_c', 'values': parameter['learning_rate_choices'], 'type': 'choice',
                     'value_type': 'float'},
                    {'name': 'learning_rate_classifier', 'values': parameter['learning_rate_choices'], 'type': 'choice',
                     'value_type': 'float'},
                    {'name': 'e_epochs', 'bounds': [parameter['epoch_lower'], parameter['epoch_upper']],
                     'type': 'range',
                     'value_type': 'int'},
                    {'name': 'm_epochs', 'bounds': [parameter['epoch_lower'], parameter['epoch_upper']],
                     'type': 'range',
                     'value_type': 'int'},
                    {'name': 'c_epochs', 'bounds': [parameter['epoch_lower'], parameter['epoch_upper']],
                     'type': 'range',
                     'value_type': 'int'},
                    {'name': 'classifier_epochs', 'bounds': [parameter['epoch_lower'], parameter['epoch_upper']],
                     'type': 'range', 'value_type': 'int'},
                    {'name': 'mini_batch', 'values': parameter['all_triplet_batch_size_choices'],
                     'value_type': 'int', 'type': 'choice'},
                    {'name': 'margin', 'values': parameter['margin_choices'], 'type': 'choice', 'value_type': 'float'},
                    {'name': 'e_dimension', 'values': parameter['dim_choice'], 'type': 'choice', 'value_type': 'int'},
                    {'name': 'm_dimension', 'values': parameter['dim_choice'], 'type': 'choice', 'value_type': 'int'},
                    {'name': 'c_dimension', 'values': parameter['dim_choice'], 'type': 'choice', 'value_type': 'int'}
                    ]

    return search_space


def create_random_forest_search_space():
    search_space = {
        {'name': 'max_depth', 'values': parameter['max_depth'], 'type': 'choice', 'value_type': 'int'},
        {'name': 'n_estimators', 'values': parameter['n_estimators'], 'type': 'choice', 'value_type': 'int'},
        {'name': 'min_samples_split', 'values': parameter['min_samples_split'], 'type': 'choice', 'value_type': 'int'},
        {'name': 'min_samples_leaf', 'values': parameter['min_samples_leaf'], 'type': 'choice', 'value_type': 'int'}
    }
    return search_space
