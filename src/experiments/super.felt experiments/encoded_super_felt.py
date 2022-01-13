import sys

import numpy as np
import torch
from pathlib import Path
import yaml
from ax import optimize
import sklearn.preprocessing as sk
from scipy.stats import sem
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch import optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.experiment_utils import create_generation_strategy
from models.super_felt_model import SupervisedEncoder, Classifier
from utils.network_training_util import get_triplet_selector, train_encoder, create_sampler
from utils.searchspaces import get_super_felt_search_space

with open(Path('../../config/hyperparameter.yaml'), 'r') as stream:
    parameter = yaml.safe_load(stream)
best_auroc = -1


def optimise_encoded_super_felt_parameter(combine_latent_features, random_seed,
                                          same_dimension_latent_features,
                                          sampling_method, search_iterations,
                                          semi_hard_triplet,
                                          sobol_iterations, x_train_val_e,
                                          x_train_val_m, x_train_val_c, y_train_val, device,
                                          deactivate_skip_bad_iterations):
    evaluation_function = lambda parameterization: train_validate_hyperparameter_set(x_train_val_e,
                                                                                     x_train_val_m, x_train_val_c,
                                                                                     y_train_val, device,
                                                                                     parameterization,
                                                                                     semi_hard_triplet,
                                                                                     deactivate_skip_bad_iterations,
                                                                                     same_dimension_latent_features,
                                                                                     combine_latent_features)
    generation_strategy = create_generation_strategy(sampling_method, sobol_iterations, random_seed)
    search_space = get_super_felt_search_space(semi_hard_triplet, same_dimension_latent_features, True)
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


def compute_encoded_super_felt_metrics(x_test_e, x_test_m, x_test_c, x_train_val_e, x_train_val_m, x_train_val_c,
                                       best_parameters, combine_latent_features, device, extern_e, extern_m, extern_c,
                                       extern_r, same_dimension_latent_features, semi_hard_triplet, y_test,
                                       y_train_val):
    # retrain best
    final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, final_Classifier,\
    final_combiner, final_scaler_gdsc = encoded_train_final(x_train_val_e, x_train_val_m, x_train_val_c,
                                                            y_train_val, best_parameters,
                                                            device, semi_hard_triplet, same_dimension_latent_features)
    # Test
    test_AUC, test_AUCPR = encoded_test(x_test_e, x_test_m, x_test_c, y_test, device, final_C_Supervised_Encoder,
                                        final_Classifier, final_E_Supervised_Encoder, final_M_Supervised_Encoder,
                                        final_scaler_gdsc, final_combiner)
    # Extern
    external_AUC, external_AUCPR = encoded_test(extern_e, extern_m, extern_c, extern_r, device,
                                                final_C_Supervised_Encoder,
                                                final_Classifier, final_E_Supervised_Encoder,
                                                final_M_Supervised_Encoder,
                                                final_scaler_gdsc, final_combiner)

    return external_AUC, external_AUCPR, test_AUC, test_AUCPR


def train_combiner(integration_epoch, combiner_optimiser, combiner, train_loader,
                   e_supervised_encoder, m_supervised_encoder, c_supervised_encoder, triplet_selector,
                   triplet_loss_function, semi_hard_triplet, device):
    combiner.train().to(device)
    for epoch in range(integration_epoch):
        last_epochs = False if epoch < integration_epoch - 2 else True
        for dataE, dataM, dataC, target in train_loader:
            dataE = dataE.to(device)
            dataM = dataM.to(device)
            dataC = dataC.to(device)
            encoded_E = e_supervised_encoder(dataE)
            encoded_M = m_supervised_encoder(dataM)
            encoded_C = c_supervised_encoder(dataC)
            concat_encoded = torch.cat((encoded_E, encoded_M, encoded_C), 1)
            concat_encoded = combiner(concat_encoded)
            if not last_epochs and semi_hard_triplet:
                encoded_Triplets_list = triplet_selector[0].get_triplets(concat_encoded, target)
            elif last_epochs and semi_hard_triplet:
                encoded_Triplets_list = triplet_selector[1].get_triplets(concat_encoded, target)
            else:
                encoded_Triplets_list = triplet_selector.get_triplets(concat_encoded, target)
            integrated_loss = triplet_loss_function(concat_encoded[encoded_Triplets_list[:, 0], :],
                                                    concat_encoded[encoded_Triplets_list[:, 1], :],
                                                    concat_encoded[encoded_Triplets_list[:, 2], :])

            combiner_optimiser.zero_grad()
            integrated_loss.backward()
            combiner_optimiser.step()
    combiner.eval()


def train_validate_hyperparameter_set(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val, device,
                                      hyperparameters, semi_hard_triplet, deactivate_skip_bad_iterations,
                                      same_dimension_latent_features, combine_latent_features):
    bce_loss_function = torch.nn.BCELoss()
    skf = StratifiedKFold(n_splits=parameter['cv_splits'])
    encoder_dropout = hyperparameters['encoder_dropout']
    encoder_weight_decay = hyperparameters['encoder_weight_decay']
    classifier_dropout = hyperparameters['classifier_dropout']
    classifier_weight_decay = hyperparameters['classifier_weight_decay']

    lrE = hyperparameters['learning_rate_e']
    lrM = hyperparameters['learning_rate_m']
    lrC = hyperparameters['learning_rate_c']
    lrCL = hyperparameters['learning_rate_classifier']
    combiner_dropout = hyperparameters['combiner_dropout']
    combiner_weight_decay = hyperparameters['combiner_weight_decay']
    learning_rate_combiner = hyperparameters['learning_rate_combiner']
    combiner_epoch = hyperparameters['combiner_epoch']
    combiner_dimension = hyperparameters['combiner_dimension']
    if same_dimension_latent_features:
        OE_dim = hyperparameters['encoder_dimension']
        OM_dim = hyperparameters['encoder_dimension']
        OC_dim = hyperparameters['encoder_dimension']
    else:
        OE_dim = hyperparameters['e_dimension']
        OM_dim = hyperparameters['m_dimension']
        OC_dim = hyperparameters['c_dimension']

    E_Supervised_Encoder_epoch = hyperparameters['e_epochs']
    C_Supervised_Encoder_epoch = hyperparameters['m_epochs']
    M_Supervised_Encoder_epoch = hyperparameters['c_epochs']
    Classifier_epoch = hyperparameters['classifier_epochs']
    mini_batch_size = hyperparameters['mini_batch']
    margin = hyperparameters['margin']
    triplet_selector = get_triplet_selector(margin, semi_hard_triplet)
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)
    iteration = 1
    val_auroc_list = list()
    for train_index, validate_index in tqdm(skf.split(x_train_val_e, y_train_val), total=skf.get_n_splits(),
                                            desc="k-fold"):
        X_trainE = x_train_val_e[train_index]
        X_valE = x_train_val_e[validate_index]
        X_trainM = x_train_val_m[train_index]
        X_valM = x_train_val_m[validate_index]
        X_trainC = x_train_val_c[train_index]
        X_valC = x_train_val_c[validate_index]
        Y_train = y_train_val[train_index]
        Y_val = y_train_val[validate_index]
        sampler = create_sampler(Y_train)
        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(X_trainE)
        X_trainE = scalerGDSC.transform(X_trainE)
        X_valE = torch.FloatTensor(scalerGDSC.transform(X_valE)).to(device)
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                      torch.FloatTensor(X_trainC),
                                                      torch.FloatTensor(Y_train.astype(int)))

        trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                                  num_workers=1, sampler=sampler, drop_last=True)

        IE_dim = X_trainE.shape[-1]
        IM_dim = X_trainM.shape[-1]
        IC_dim = X_trainC.shape[-1]

        E_Supervised_Encoder = SupervisedEncoder(IE_dim, OE_dim, encoder_dropout).to(device)
        M_Supervised_Encoder = SupervisedEncoder(IM_dim, OM_dim, encoder_dropout).to(device)
        C_Supervised_Encoder = SupervisedEncoder(IC_dim, OC_dim, encoder_dropout).to(device)
        OCP_dim = OE_dim + OM_dim + OC_dim
        combiner = SupervisedEncoder(OCP_dim, combiner_dimension, combiner_dropout).to(device)

        E_optimizer = optim.Adagrad(E_Supervised_Encoder.parameters(), lr=lrE, weight_decay=encoder_weight_decay)
        M_optimizer = optim.Adagrad(M_Supervised_Encoder.parameters(), lr=lrM, weight_decay=encoder_weight_decay)
        C_optimizer = optim.Adagrad(C_Supervised_Encoder.parameters(), lr=lrC, weight_decay=encoder_weight_decay)
        combiner_optimizer = optim.Adagrad(combiner.parameters(), lr=learning_rate_combiner,
                                           weight_decay=combiner_weight_decay)

        classifier = Classifier(combiner_dimension, classifier_dropout).to(device)
        classifier_optimizer = optim.Adagrad(classifier.parameters(), lr=lrCL, weight_decay=classifier_weight_decay)

        # train each Supervised_Encoder with triplet loss
        train_encoder(E_Supervised_Encoder_epoch, E_optimizer, triplet_selector, device, E_Supervised_Encoder,
                      trainLoader, trip_loss_fun, semi_hard_triplet, 0)
        train_encoder(M_Supervised_Encoder_epoch, M_optimizer, triplet_selector, device, M_Supervised_Encoder,
                      trainLoader, trip_loss_fun, semi_hard_triplet, 1)
        train_encoder(C_Supervised_Encoder_epoch, C_optimizer, triplet_selector, device, C_Supervised_Encoder,
                      trainLoader, trip_loss_fun, semi_hard_triplet, 2)

        train_combiner(combiner_epoch, combiner_optimizer, combiner, trainLoader,
                       E_Supervised_Encoder, M_Supervised_Encoder, C_Supervised_Encoder, triplet_selector,
                       trip_loss_fun, semi_hard_triplet, device)

        # train classifier
        for cl_epoch in range(Classifier_epoch):
            classifier.train()
            for i, (dataE, dataM, dataC, target) in enumerate(trainLoader):
                classifier_optimizer.zero_grad()
                dataE = dataE.to(device)
                dataM = dataM.to(device)
                dataC = dataC.to(device)
                target = target.to(device)
                encoded_E = E_Supervised_Encoder(dataE)
                encoded_M = M_Supervised_Encoder(dataM)
                encoded_C = C_Supervised_Encoder(dataC)
                concat_encoded = torch.cat((encoded_E, encoded_M, encoded_C), 1)
                concat_encoded = combiner(concat_encoded)
                Pred = classifier(concat_encoded, torch.FloatTensor().to(device),
                                  torch.FloatTensor().to(device))
                cl_loss = bce_loss_function(Pred, target.view(-1, 1))
                classifier_optimizer.zero_grad()
                cl_loss.backward()
                classifier_optimizer.step()

        with torch.no_grad():
            classifier.eval()
            """
                inner validation
            """
            encoded_val_E = E_Supervised_Encoder(X_valE)
            encoded_val_M = M_Supervised_Encoder(torch.FloatTensor(X_valM).to(device))
            encoded_val_C = C_Supervised_Encoder(torch.FloatTensor(X_valC).to(device))

            concat_encoded = torch.cat((encoded_val_E, encoded_val_M, encoded_val_C), 1)
            concat_encoded = combiner(concat_encoded)
            test_pred = classifier(concat_encoded, torch.FloatTensor().to(device),
                                   torch.FloatTensor().to(device))

            test_y_true = Y_val
            test_y_pred = test_pred.cpu()

            val_auroc = roc_auc_score(test_y_true, test_y_pred.detach().numpy())
            val_auroc_list.append(val_auroc)

            if not deactivate_skip_bad_iterations:
                open_folds = parameter['cv_splits'] - iteration
                remaining_best_results = np.ones(open_folds)
                best_possible_mean = np.mean(np.concatenate([val_auroc_list, remaining_best_results]))
                if check_best_auroc(best_possible_mean):
                    print('Skip remaining folds.')
                    break
            iteration += 1

    val_auroc = np.mean(val_auroc_list)
    standard_error_of_mean = sem(val_auroc_list)

    return {'auroc': (val_auroc, standard_error_of_mean)}


def check_best_auroc(best_reachable_auroc):
    global best_auroc
    return best_reachable_auroc < best_auroc


def encoded_test(x_test_e, x_test_m, x_test_c, y_test, device, final_c_supervised_encoder, final_classifier,
                 final_e_supervised_encoder, final_m_supervised_encoder, final_scaler_gdsc, combiner):
    X_testE = torch.FloatTensor(final_scaler_gdsc.transform(x_test_e))
    encoded_test_E = final_e_supervised_encoder(torch.FloatTensor(X_testE).to(device))
    encoded_test_M = final_m_supervised_encoder(torch.FloatTensor(x_test_m).to(device))
    encoded_test_C = final_c_supervised_encoder(torch.FloatTensor(x_test_c).to(device))
    concat_encoded = torch.cat((encoded_test_E, encoded_test_M, encoded_test_C), 1)
    concat_encoded = combiner(concat_encoded)
    test_Pred = final_classifier(concat_encoded, torch.FloatTensor().to(device), torch.FloatTensor().to(device))
    test_y_pred = test_Pred.cpu().detach().numpy()
    test_AUC = roc_auc_score(y_test, test_y_pred)
    test_AUCPR = average_precision_score(y_test, test_y_pred)
    return test_AUC, test_AUCPR


def encoded_train_final(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val, best_hyperparameter,
                        device, semi_hard_triplet, same_dimension_latent_features):
    bce_loss_function = torch.nn.BCELoss()
    E_dr = best_hyperparameter['encoder_dropout']
    C_dr = best_hyperparameter['classifier_dropout']
    Cwd = best_hyperparameter['classifier_weight_decay']
    Ewd = best_hyperparameter['encoder_weight_decay']
    lrE = best_hyperparameter['learning_rate_e']
    lrM = best_hyperparameter['learning_rate_m']
    lrC = best_hyperparameter['learning_rate_c']
    lrCL = best_hyperparameter['learning_rate_classifier']
    combiner_dropout = best_hyperparameter['combiner_dropout']
    combiner_weight_decay = best_hyperparameter['combiner_weight_decay']
    learning_rate_combiner = best_hyperparameter['learning_rate_combiner']
    combiner_epoch = best_hyperparameter['combiner_epoch']
    combiner_dimension = best_hyperparameter['combiner_dimension']
    if same_dimension_latent_features:
        OE_dim = best_hyperparameter['encoder_dimension']
        OM_dim = best_hyperparameter['encoder_dimension']
        OC_dim = best_hyperparameter['encoder_dimension']
    else:
        OE_dim = best_hyperparameter['e_dimension']
        OM_dim = best_hyperparameter['m_dimension']
        OC_dim = best_hyperparameter['c_dimension']

    margin = best_hyperparameter['margin']
    E_Supervised_Encoder_epoch = best_hyperparameter['e_epochs']
    C_Supervised_Encoder_epoch = best_hyperparameter['m_epochs']
    M_Supervised_Encoder_epoch = best_hyperparameter['c_epochs']
    classifier_epoch = best_hyperparameter['classifier_epochs']
    mb_size = best_hyperparameter['mini_batch']

    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)
    sampler = create_sampler(y_train_val)
    final_scaler_gdsc = sk.StandardScaler()
    final_scaler_gdsc.fit(x_train_val_e)
    x_train_val_e = final_scaler_gdsc.transform(x_train_val_e)
    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_val_e), torch.FloatTensor(x_train_val_m),
                                                  torch.FloatTensor(x_train_val_c),
                                                  torch.FloatTensor(y_train_val.astype(int)))
    train_loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                               num_workers=1, sampler=sampler, drop_last=True)
    IE_dim = x_train_val_e.shape[-1]
    IM_dim = x_train_val_m.shape[-1]
    IC_dim = x_train_val_c.shape[-1]
    final_E_Supervised_Encoder = SupervisedEncoder(IE_dim, OE_dim, E_dr).to(device)
    final_M_Supervised_Encoder = SupervisedEncoder(IM_dim, OM_dim, E_dr).to(device)
    final_C_Supervised_Encoder = SupervisedEncoder(IC_dim, OC_dim, E_dr).to(device)
    OCP_dim = OE_dim + OM_dim + OC_dim

    final_combiner = SupervisedEncoder(OCP_dim, combiner_dimension, combiner_dropout).to(device)

    E_optimizer = optim.Adagrad(final_E_Supervised_Encoder.parameters(), lr=lrE, weight_decay=Ewd)
    M_optimizer = optim.Adagrad(final_M_Supervised_Encoder.parameters(), lr=lrM, weight_decay=Ewd)
    C_optimizer = optim.Adagrad(final_C_Supervised_Encoder.parameters(), lr=lrC, weight_decay=Ewd)
    combiner_optimizer = optim.Adagrad(final_combiner.parameters(), lr=learning_rate_combiner,
                                       weight_decay=combiner_weight_decay)
    triplet_selector = get_triplet_selector(margin, semi_hard_triplet)
    final_classifier = Classifier(combiner_dimension, C_dr).to(device)
    classifier_optimizer = optim.Adagrad(final_classifier.parameters(), lr=lrCL, weight_decay=Cwd)

    # train each Supervised_Encoder with triplet loss
    train_encoder(E_Supervised_Encoder_epoch, E_optimizer, triplet_selector, device, final_E_Supervised_Encoder,
                  train_loader,
                  trip_loss_fun, semi_hard_triplet, 0)
    train_encoder(M_Supervised_Encoder_epoch, M_optimizer, triplet_selector, device, final_M_Supervised_Encoder,
                  train_loader,
                  trip_loss_fun, semi_hard_triplet, 1)
    train_encoder(C_Supervised_Encoder_epoch, C_optimizer, triplet_selector, device, final_C_Supervised_Encoder,
                  train_loader,
                  trip_loss_fun, semi_hard_triplet, 2)

    train_combiner(combiner_epoch, combiner_optimizer, final_combiner, train_loader,
                   final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, triplet_selector,
                   trip_loss_fun, semi_hard_triplet, device)

    # train classifier
    for cl_epoch in range(classifier_epoch):
        final_classifier.train()
        for dataE, dataM, dataC, target in train_loader:
            classifier_optimizer.zero_grad()
            dataE = dataE.to(device)
            dataM = dataM.to(device)
            dataC = dataC.to(device)
            target = target.to(device)
            encoded_E = final_E_Supervised_Encoder(dataE)
            encoded_M = final_M_Supervised_Encoder(dataM)
            encoded_C = final_C_Supervised_Encoder(dataC)
            concat_encoded = torch.cat((encoded_E, encoded_M, encoded_C), 1)
            concat_encoded = final_combiner(concat_encoded)
            Pred = final_classifier(concat_encoded, torch.FloatTensor().to(device),
                                    torch.FloatTensor().to(device))
            cl_loss = bce_loss_function(Pred, target.view(-1, 1))
            classifier_optimizer.zero_grad()
            cl_loss.backward()
            classifier_optimizer.step()
    final_classifier.eval()
    return final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, final_classifier, \
           final_combiner, final_scaler_gdsc
