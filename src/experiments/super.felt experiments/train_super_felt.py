from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import yaml
from ax import optimize
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from torch import optim
from tqdm import tqdm

from models.super_felt_model import Encoder, Classifier
from siamese_triplet.utils import AllTripletSelector
from utils.experiment_utils import create_generation_strategy
from utils.network_training_util import train_encoder, train_classifier, create_sampler, \
    super_felt_test, train_validate_classifier
from utils.searchspaces import create_super_felt_search_space


best_auroc = -1
with open(Path('../../config/hyperparameter.yaml'), 'r') as stream:
    parameter = yaml.safe_load(stream)


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def optimise_super_felt_parameter(search_iterations, x_train_val_e,
                                  x_train_val_m, x_train_val_c, y_train_val, device):
    evaluation_function = lambda parameterization: train_validate_hyperparameter_set(x_train_val_e,
                                                                                     x_train_val_m, x_train_val_c,
                                                                                     y_train_val, device,
                                                                                     parameterization)
    generation_strategy = create_generation_strategy()
    search_space = create_super_felt_search_space()
    best_parameters, values, experiment, model = optimize(
        total_trials=search_iterations,
        experiment_name='Super.FELT',
        objective_name='auroc',
        parameters=search_space,
        evaluation_function=evaluation_function,
        minimize=False,
        generation_strategy=generation_strategy,
    )
    return best_parameters, experiment


def compute_super_felt_metrics(x_test_e, x_test_m, x_test_c, x_train_val_e, x_train_val_m, x_train_val_c,
                               best_parameters, device, extern_e, extern_m, extern_c, extern_r,
                               y_test, y_train_val):
    # retrain best
    final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, final_Classifier, \
    final_scaler_gdsc = train_final(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val, best_parameters,
                                    device)
    # Test
    test_AUC, test_AUCPR = super_felt_test(x_test_e, x_test_m, x_test_c, y_test, device, final_C_Supervised_Encoder,
                                           final_Classifier, final_E_Supervised_Encoder, final_M_Supervised_Encoder,
                                           final_scaler_gdsc)
    # Extern
    external_AUC, external_AUCPR = super_felt_test(extern_e, extern_m, extern_c, extern_r, device,
                                                   final_C_Supervised_Encoder,
                                                   final_Classifier, final_E_Supervised_Encoder,
                                                   final_M_Supervised_Encoder,
                                                   final_scaler_gdsc)
    return external_AUC, external_AUCPR, test_AUC, test_AUCPR


def train_validate_hyperparameter_set(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val,
                                      device, hyperparameters):
    skf = StratifiedKFold(n_splits=parameter['cv_splits'])
    all_validation_aurocs = []
    encoder_dropout = hyperparameters['encoder_dropout']
    encoder_weight_decay = hyperparameters['encoder_weight_decay']
    classifier_dropout = hyperparameters['classifier_dropout']
    classifier_weight_decay = hyperparameters['classifier_weight_decay']

    lrE = hyperparameters['learning_rate_e']
    lrM = hyperparameters['learning_rate_m']
    lrC = hyperparameters['learning_rate_c']
    lrCL = hyperparameters['learning_rate_classifier']

    OE_dim = hyperparameters['e_dimension']
    OM_dim = hyperparameters['m_dimension']
    OC_dim = hyperparameters['c_dimension']

    e_Encoder_epoch = hyperparameters['e_epochs']
    c_Encoder_epoch = hyperparameters['m_epochs']
    m_Encoder_epoch = hyperparameters['c_epochs']
    classifier_epoch = hyperparameters['classifier_epochs']
    mini_batch_size = hyperparameters['mini_batch']
    margin = hyperparameters['margin']
    triplet_selector = AllTripletSelector()
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)
    iteration = 1

    for train_index, validate_index in tqdm(skf.split(x_train_val_e, y_train_val), total=skf.get_n_splits(),
                                            desc="k-fold"):
        X_trainE = x_train_val_e[train_index]
        x_val_e = x_train_val_e[validate_index]
        X_trainM = x_train_val_m[train_index]
        x_val_m = x_train_val_m[validate_index]
        X_trainC = x_train_val_c[train_index]
        x_val_c = x_train_val_c[validate_index]
        Y_train = y_train_val[train_index]
        y_val = y_train_val[validate_index]
        sampler = create_sampler(Y_train)
        scalerGDSC = StandardScaler()
        X_trainE = scalerGDSC.fit_transform(X_trainE)
        x_val_e = torch.FloatTensor(scalerGDSC.transform(x_val_e)).to(device)
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                      torch.FloatTensor(X_trainC),
                                                      torch.FloatTensor(Y_train.astype(int)))

        train_loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                                   num_workers=8, sampler=sampler, drop_last=True)

        IE_dim = X_trainE.shape[-1]
        IM_dim = X_trainM.shape[-1]
        IC_dim = X_trainC.shape[-1]

        e_encoder = Encoder(IE_dim, OE_dim, encoder_dropout).to(device)
        m_encoder = Encoder(IM_dim, OM_dim, encoder_dropout).to(device)
        c_encoder = Encoder(IC_dim, OC_dim, encoder_dropout).to(device)

        E_optimizer = optim.Adagrad(e_encoder.parameters(), lr=lrE, weight_decay=encoder_weight_decay)
        M_optimizer = optim.Adagrad(m_encoder.parameters(), lr=lrM, weight_decay=encoder_weight_decay)
        C_optimizer = optim.Adagrad(c_encoder.parameters(), lr=lrC, weight_decay=encoder_weight_decay)

        # train each Supervised_Encoder with triplet loss
        train_encoder(e_Encoder_epoch, E_optimizer, triplet_selector, device, e_encoder, train_loader, trip_loss_fun, 0)
        train_encoder(m_Encoder_epoch, M_optimizer, triplet_selector, device, m_encoder, train_loader, trip_loss_fun, 1)
        train_encoder(c_Encoder_epoch, C_optimizer, triplet_selector, device, c_encoder, train_loader, trip_loss_fun, 2)

        # train classifier
        classifier_input_dimension = OE_dim + OM_dim + OC_dim
        classifier = Classifier(classifier_input_dimension, classifier_dropout).to(device)
        classifier_optimizer = optim.Adagrad(classifier.parameters(), lr=lrCL,
                                             weight_decay=classifier_weight_decay)
        val_auroc = train_validate_classifier(classifier_epoch, device, e_encoder,
                                              m_encoder, c_encoder, train_loader,
                                              classifier_optimizer,
                                              x_val_e, x_val_m, x_val_c, y_val, classifier)
        all_validation_aurocs.append(val_auroc)

        if iteration < parameter['cv_splits']:
            open_folds = parameter['cv_splits'] - iteration
            remaining_best_results = np.ones(open_folds)
            best_possible_mean = np.mean(np.concatenate([all_validation_aurocs, remaining_best_results]))
            if check_best_auroc(best_possible_mean):
                print('Skip remaining folds.')
                break
        iteration += 1

    val_auroc = np.mean(all_validation_aurocs)
    standard_error_of_mean = sem(all_validation_aurocs)

    return {'auroc': (val_auroc, standard_error_of_mean)}


def check_best_auroc(best_reachable_auroc):
    global best_auroc
    return best_reachable_auroc < best_auroc


def train_final(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val, best_hyperparameter, device):
    E_dr = best_hyperparameter['encoder_dropout']
    C_dr = best_hyperparameter['classifier_dropout']
    Cwd = best_hyperparameter['classifier_weight_decay']
    Ewd = best_hyperparameter['encoder_weight_decay']
    lrE = best_hyperparameter['learning_rate_e']
    lrM = best_hyperparameter['learning_rate_m']
    lrC = best_hyperparameter['learning_rate_c']
    lrCL = best_hyperparameter['learning_rate_classifier']
    OE_dim = best_hyperparameter['e_dimension']
    OM_dim = best_hyperparameter['m_dimension']
    OC_dim = best_hyperparameter['c_dimension']
    E_Supervised_Encoder_epoch = best_hyperparameter['e_epochs']
    C_Supervised_Encoder_epoch = best_hyperparameter['m_epochs']
    M_Supervised_Encoder_epoch = best_hyperparameter['c_epochs']
    classifier_epoch = best_hyperparameter['classifier_epochs']
    mb_size = best_hyperparameter['mini_batch']
    margin = best_hyperparameter['margin']

    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)
    sampler = create_sampler(y_train_val)
    final_scaler = StandardScaler()
    x_train_val_e = final_scaler.fit_transform(x_train_val_e)
    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_val_e), torch.FloatTensor(x_train_val_m),
                                                  torch.FloatTensor(x_train_val_c),
                                                  torch.FloatTensor(y_train_val.astype(int)))
    train_loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                               num_workers=8, sampler=sampler, drop_last=True)
    IE_dim = x_train_val_e.shape[-1]
    IM_dim = x_train_val_m.shape[-1]
    IC_dim = x_train_val_c.shape[-1]
    final_E_encoder = Encoder(IE_dim, OE_dim, E_dr).to(device)
    final_M_encoder = Encoder(IM_dim, OM_dim, E_dr).to(device)
    final_C_encoder = Encoder(IC_dim, OC_dim, E_dr).to(device)
    E_optimizer = optim.Adagrad(final_E_encoder.parameters(), lr=lrE, weight_decay=Ewd)
    M_optimizer = optim.Adagrad(final_M_encoder.parameters(), lr=lrM, weight_decay=Ewd)
    C_optimizer = optim.Adagrad(final_C_encoder.parameters(), lr=lrC, weight_decay=Ewd)
    triplet_selector = AllTripletSelector()
    OCP_dim = OE_dim + OM_dim + OC_dim
    final_classifier = Classifier(OCP_dim, C_dr).to(device)
    classifier_optimizer = optim.Adagrad(final_classifier.parameters(), lr=lrCL, weight_decay=Cwd)

    # train each Supervised_Encoder with triplet loss
    train_encoder(E_Supervised_Encoder_epoch, E_optimizer, triplet_selector, device, final_E_encoder,
                  train_loader, trip_loss_fun, 0)
    train_encoder(M_Supervised_Encoder_epoch, M_optimizer, triplet_selector, device, final_M_encoder,
                  train_loader, trip_loss_fun, 1)
    train_encoder(C_Supervised_Encoder_epoch, C_optimizer, triplet_selector, device, final_C_encoder,
                  train_loader, trip_loss_fun, 2)

    # train classifier
    train_classifier(final_classifier, classifier_epoch, train_loader, classifier_optimizer, final_E_encoder,
                     final_M_encoder, final_C_encoder, device)
    return final_E_encoder, final_M_encoder, final_C_encoder, final_classifier, final_scaler
