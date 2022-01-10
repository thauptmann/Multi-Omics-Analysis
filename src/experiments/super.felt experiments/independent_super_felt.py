import numpy as np
import torch
from ax import optimize
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.experiment_utils import create_generation_strategy
from utils.network_training_util import get_triplet_selector, create_sampler, train_encoder, train_classifier
from utils.searchspaces import get_encoder_search_space, get_classifier_search_space, parameter
from super_felt_model import SupervisedEncoder, Classifier


def train_validate_encoder(hyperparameters, x_train_val, y_train_val, semi_hard_triplet, device, omic_number):
    margin = hyperparameters['margin']
    output_dimension = hyperparameters['dimension']
    dropout = hyperparameters['dropout']
    epochs = hyperparameters['epochs']
    weight_decay = hyperparameters['weight_decay']
    mini_batch_size = hyperparameters['mini_batch']
    learning_rate = hyperparameters['learning_rate']
    triplet_selector = get_triplet_selector(margin, semi_hard_triplet)
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)
    skf = StratifiedKFold(n_splits=parameter['cv_splits'])
    loss_list = list()
    for train_index, validate_index in tqdm(skf.split(x_train_val, y_train_val), total=skf.get_n_splits(),
                                            desc="k-fold"):
        X_train = x_train_val[train_index]
        X_val = x_train_val[validate_index]
        y_train = y_train_val[train_index]
        y_val = y_train_val[validate_index]
        sampler = create_sampler(y_train)

        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train),
                                                      torch.FloatTensor(y_train.astype(int)))
        trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                                  num_workers=8, sampler=sampler, drop_last=True)
        input_dimension = X_train.shape[-1]
        supervised_encoder = SupervisedEncoder(input_dimension, output_dimension, dropout)
        supervised_encoder.to(device)
        optimizer = optim.Adagrad(supervised_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_encoder(epochs, optimizer, triplet_selector, device, supervised_encoder,
                      trainLoader, trip_loss_fun, semi_hard_triplet, omic_number, True)
        with torch.no_grad():
            supervised_encoder.eval()
            encoded_val = supervised_encoder(torch.FloatTensor(X_val).to(device))
            triplets_list = triplet_selector.get_triplets(encoded_val, torch.FloatTensor(y_val))
            val_loss = trip_loss_fun(encoded_val[triplets_list[:, 0], :],
                                     encoded_val[triplets_list[:, 1], :],
                                     encoded_val[triplets_list[:, 2], :])
        loss_list.append(val_loss.cpu().detach().numpy())

    return np.mean(loss_list)


def optimise_independent_super_felt_parameter(combine_latent_features, random_seed, same_dimension_latent_features,
                                              sampling_method, search_iterations, semi_hard_triplet, sobol_iterations,
                                              x_train_val_e, x_train_val_m, x_train_val_c, y_train_val, device,
                                              deactivate_skip_bad_iterations):
    best_parameters_list = list()
    generation_strategy = create_generation_strategy(sampling_method, sobol_iterations, random_seed)
    encoder_search_space = get_encoder_search_space(semi_hard_triplet)
    evaluation_function_e = lambda parameterization: train_validate_encoder(parameterization, x_train_val_e,
                                                                            y_train_val, semi_hard_triplet, device, 0)
    evaluation_function_m = lambda parameterization: train_validate_encoder(parameterization, x_train_val_m,
                                                                            y_train_val, semi_hard_triplet, device, 1)
    evaluation_function_c = lambda parameterization: train_validate_encoder(parameterization, x_train_val_c,
                                                                            y_train_val, semi_hard_triplet, device, 2)

    for evaluation_function in (evaluation_function_e, evaluation_function_m, evaluation_function_c):
        best_parameters, _, _, _ = optimize(
            total_trials=search_iterations,
            experiment_name='Encoder',
            objective_name='triplet_loss',
            parameters=encoder_search_space,
            evaluation_function=evaluation_function,
            minimize=False,
            generation_strategy=generation_strategy,
        )
        best_parameters_list.append(best_parameters)

    # retrain best encoder
    best_encoder_e, scaler_e = final_training_encoder(best_parameters_e, x_train_val_e, Y_train_val, device)
    best_encoder_m, _ = final_training_encoder(best_parameters_m, x_train_val_m, Y_train_val, device)
    best_encoder_c, _ = final_training_encoder(best_parameters_c, x_train_val_c, Y_train_val, device)

    input_dimension = best_parameters_list[0]['dimension'] + best_parameters_list[1]['dimension'] \
                      + best_parameters_list[2]['dimension']

    classifier_search_space = get_classifier_search_space(semi_hard_triplet)
    evaluation_function_classifier = lambda parameterization: train_validate_classifier(parameterization,
                                                                                        x_train_val_e, x_train_val_m,
                                                                                        x_train_val_c,
                                                                                        y_train_val, input_dimension,
                                                                                        best_encoder_e,
                                                                                        best_encoder_m, best_encoder_c,
                                                                                        device)
    best_parameters_classifier, _, _, _ = optimize(
        total_trials=search_iterations,
        experiment_name='classifier',
        objective_name='auroc',
        parameters=classifier_search_space,
        evaluation_function=evaluation_function_classifier,
        minimize=False,
        generation_strategy=generation_strategy,
    )

    best_parameters_list.append(best_parameters_classifier)
    return best_parameters_list


def compute_independent_super_felt_metrics(x_test_e, x_test_m, x_test_c, x_train_val_e, x_train_val_m, x_train_val_c,
                                           best_parameters, device, extern_e, extern_m,
                                           extern_c, extern_r, same_dimension_latent_features, semi_hard_triplet,
                                           y_test, y_train_val):
    pass


def train_validate_classifier(hyperparameters, x_train_val_e, x_train_val_m, x_train_val_c,
                              y_train_val, input_dimension,
                              final_f_supervised_encoder,
                              final_m_supervised_encoder, final_c_supervised_encoder, device):
    dropout = hyperparameters['dropout']
    epochs = hyperparameters['epochs']
    weight_decay = hyperparameters['weight_decay']
    mini_batch_size = hyperparameters['mini_batch']
    learning_rate = hyperparameters['learning_rate']
    skf = StratifiedKFold(n_splits=parameter['cv_splits'])

    loss_list = list()
    for train_index, validate_index in tqdm(skf.split(x_train_val_e, y_train_val), total=skf.get_n_splits(),
                                            desc="k-fold"):
        X_train_e = x_train_val_e[train_index]
        X_val_e = x_train_val_e[validate_index]
        X_train_m = x_train_val_m[train_index]
        X_val_m = x_train_val_m[validate_index]
        X_train_c = x_train_val_c[train_index]
        X_val_c = x_train_val_c[validate_index]
        y_train = y_train_val[train_index]
        y_val = y_train_val[validate_index]
        sampler = create_sampler(y_train)
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_e),
                                                      torch.FloatTensor(X_train_m),
                                                      torch.FloatTensor(X_train_c),
                                                      torch.FloatTensor(y_train.astype(int)))
        train_loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                                   num_workers=8, sampler=sampler)

        classifier = Classifier(input_dimension, dropout)
        classifier.to(device)
        optimizer = optim.Adagrad(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_classifier(classifier, epochs, train_loader, optimizer,
                         final_f_supervised_encoder,
                         final_m_supervised_encoder, final_c_supervised_encoder,
                         device)
        with torch.no_grad():
            classifier.eval()
            encoded_e = final_f_supervised_encoder(torch.FloatTensor(X_val_e).to(device))
            encoded_m = final_m_supervised_encoder(torch.FloatTensor(X_val_m).to(device))
            encoded_c = final_c_supervised_encoder(torch.FloatTensor(X_val_c).to(device))
            classified = classifier(encoded_e, encoded_m, encoded_c)
            test_y_pred = classified.cpu()
            val_auroc = roc_auc_score(y_val, test_y_pred.detach().numpy())

        loss_list.append(val_auroc)

    return np.mean(loss_list)
