import argparse
import sys
import time

import yaml
import torch
from pathlib import Path
import sklearn.preprocessing as sk
from ax import optimize
from scipy.stats import sem
from torch import optim
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.visualisation import save_auroc_plots, save_auroc_with_variance_plots
from utils.experiment_utils import create_generation_strategy
from utils.searchspaces import get_super_felt_search_space
from models.super_felt_model import SupervisedEncoder, Classifier
from utils.network_training_util import calculate_mean_and_std_auc, get_triplet_selector, create_sampler, train_encoder, \
    train_classifier, train_validate_classifier, super_felt_test
from utils import multi_omics_data
from utils.choose_gpu import get_free_gpu
from independent_super_felt import optimise_independent_super_felt_parameter, compute_independent_super_felt_metrics
from encoded_super_felt import optimise_encoded_super_felt_parameter, compute_encoded_super_felt_metrics

with open(Path('../../config/hyperparameter.yaml'), 'r') as stream:
    parameter = yaml.safe_load(stream)
best_auroc = 0


def super_felt(experiment_name, drug_name, extern_dataset_name, gpu_number, search_iterations, sobol_iterations,
               sampling_method, deactivate_elbow_method, deactivate_skip_bad_iterations, semi_hard_triplet,
               combine_latent_features, optimise_independent, same_dimension_latent_features):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")
    random_seed = parameter['random_seed']
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    sobol_iterations = search_iterations if sampling_method == 'sobol' else sobol_iterations

    data_path = Path('..', '..', '..', 'data')
    result_path = Path('..', '..', '..', 'results', 'super.felt', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = open(result_path / 'results.txt', 'w')
    if deactivate_elbow_method:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name)
    else:
        gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
            = multi_omics_data.load_drug_data_with_elbow(data_path, drug_name, extern_dataset_name)

    test_auc_list = []
    extern_auc_list = []
    objectives_list = []
    test_auprc_list = []
    test_validation_list = []
    extern_auprc_list = []
    start_time = time.time()

    skf_outer = StratifiedKFold(n_splits=parameter['cv_splits'], random_state=random_seed, shuffle=True)
    iteration = 0
    for train_index_outer, test_index in tqdm(skf_outer.split(gdsc_e, gdsc_r), total=skf_outer.get_n_splits(),
                                              desc=" Outer k-fold"):
        global best_auroc
        best_auroc = 0
        x_train_val_e = gdsc_e[train_index_outer]
        x_test_e = gdsc_e[test_index]
        x_train_val_m = gdsc_m[train_index_outer]
        x_test_m = gdsc_m[test_index]
        x_train_val_c = gdsc_c[train_index_outer]
        x_test_c = gdsc_c[test_index]
        y_train_val = gdsc_r[train_index_outer]
        y_test = gdsc_r[test_index]
        if combine_latent_features:
            best_parameters, experiment = optimise_encoded_super_felt_parameter(combine_latent_features, random_seed,
                                                                                same_dimension_latent_features,
                                                                                sampling_method, search_iterations,
                                                                                semi_hard_triplet,
                                                                                sobol_iterations, x_train_val_e,
                                                                                x_train_val_m, x_train_val_c,
                                                                                y_train_val, device,
                                                                                deactivate_skip_bad_iterations)
            external_AUC, external_AUCPR, test_AUC, test_AUCPR = compute_encoded_super_felt_metrics(x_test_e, x_test_m,
                                                                                                    x_test_c,
                                                                                                    x_train_val_e,
                                                                                                    x_train_val_m,
                                                                                                    x_train_val_c,
                                                                                                    best_parameters,
                                                                                                    combine_latent_features,
                                                                                                    device,
                                                                                                    extern_e, extern_m,
                                                                                                    extern_c,
                                                                                                    extern_r,
                                                                                                    same_dimension_latent_features,
                                                                                                    semi_hard_triplet,
                                                                                                    y_test,
                                                                                                    y_train_val)
        elif optimise_independent:
            best_parameters, experiment, final_encoder_e, final_encoder_m, final_encoder_c, scaler_e = \
                optimise_independent_super_felt_parameter(combine_latent_features,
                                                          random_seed,
                                                          same_dimension_latent_features,
                                                          sampling_method, search_iterations,
                                                          semi_hard_triplet,
                                                          sobol_iterations, x_train_val_e,
                                                          x_train_val_m, x_train_val_c,
                                                          y_train_val,
                                                          device,
                                                          deactivate_skip_bad_iterations)
            external_AUC, external_AUCPR, test_AUC, test_AUCPR = compute_independent_super_felt_metrics(x_test_e,
                                                                                                        x_test_m,
                                                                                                        x_test_c,
                                                                                                        x_train_val_e,
                                                                                                        x_train_val_m,
                                                                                                        x_train_val_c,
                                                                                                        best_parameters,
                                                                                                        device,
                                                                                                        extern_e,
                                                                                                        extern_m,
                                                                                                        extern_c,
                                                                                                        extern_r,
                                                                                                        y_test,
                                                                                                        y_train_val,
                                                                                                        final_encoder_e,
                                                                                                        final_encoder_m,
                                                                                                        final_encoder_c,
                                                                                                        scaler_e)
        else:
            best_parameters, experiment = optimise_super_felt_parameter(combine_latent_features, random_seed,
                                                                        same_dimension_latent_features,
                                                                        sampling_method, search_iterations,
                                                                        semi_hard_triplet,
                                                                        sobol_iterations, x_train_val_e,
                                                                        x_train_val_m, x_train_val_c, y_train_val,
                                                                        device,
                                                                        deactivate_skip_bad_iterations)
            external_AUC, external_AUCPR, test_AUC, test_AUCPR = compute_super_felt_metrics(x_test_e, x_test_m,
                                                                                            x_test_c,
                                                                                            x_train_val_e,
                                                                                            x_train_val_m,
                                                                                            x_train_val_c,
                                                                                            best_parameters,
                                                                                            device,
                                                                                            extern_e, extern_m,
                                                                                            extern_c,
                                                                                            extern_r,
                                                                                            same_dimension_latent_features,
                                                                                            semi_hard_triplet, y_test,
                                                                                            y_train_val)

        test_auc_list.append(test_AUC)
        extern_auc_list.append(external_AUC)
        test_auprc_list.append(test_AUCPR)
        extern_auprc_list.append(external_AUCPR)
        objectives = np.array([trial.objective_mean for trial in experiment.trials.values()])
        save_auroc_plots(objectives, result_path, iteration, sobol_iterations)

        max_objective = max(np.array([trial.objective_mean for trial in experiment.trials.values()]))
        objectives_list.append(objectives)

        test_validation_list.append(max_objective)
        result_file.write(f'\t\tBest {drug_name} validation Auroc = {max_objective}\n')
        iteration += 1

    end_time = time.time()
    result_file.write(f'\tMinutes needed: {round((end_time - start_time) / 60)}')
    write_results_to_file(drug_name, extern_auc_list, extern_auprc_list, result_file, test_auc_list, test_auprc_list)
    save_auroc_with_variance_plots(objectives_list, result_path, 'final', sobol_iterations)
    positive_extern = np.count_nonzero(extern_r == 1)
    negative_extern = np.count_nonzero(extern_r == 0)
    no_skill_prediction_auprc = positive_extern / (positive_extern + negative_extern)
    result_file.write(f'\n No skill predictor extern AUPRC: {no_skill_prediction_auprc} \n')
    result_file.close()
    print("Done!")


def optimise_super_felt_parameter(combine_latent_features, random_seed, same_dimension_latent_features, sampling_method,
                                  search_iterations, semi_hard_triplet, sobol_iterations, x_train_val_e,
                                  x_train_val_m, x_train_val_c, y_train_val, device, deactivate_skip_bad_iterations):
    evaluation_function = lambda parameterization: train_validate_hyperparameter_set(x_train_val_e,
                                                                                     x_train_val_m, x_train_val_c,
                                                                                     y_train_val, device,
                                                                                     parameterization,
                                                                                     semi_hard_triplet,
                                                                                     deactivate_skip_bad_iterations,
                                                                                     same_dimension_latent_features)
    generation_strategy = create_generation_strategy(sampling_method, sobol_iterations, random_seed)
    search_space = get_super_felt_search_space(semi_hard_triplet, same_dimension_latent_features,
                                               combine_latent_features)
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
                               same_dimension_latent_features, semi_hard_triplet, y_test, y_train_val):
    # retrain best
    final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, final_Classifier, \
    final_scaler_gdsc = train_final(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val, best_parameters,
                                    device, semi_hard_triplet, same_dimension_latent_features)
    # Test
    test_AUC, test_AUCPR = super_felt_test(x_test_e, x_test_m, x_test_c, y_test, device, final_C_Supervised_Encoder,
                                           final_Classifier, final_E_Supervised_Encoder, final_M_Supervised_Encoder,
                                           final_scaler_gdsc)
    # Extern
    external_AUC, external_AUCPR = super_felt_test(extern_e, extern_m, extern_c, extern_r, device,
                                                   final_C_Supervised_Encoder,
                                                   final_Classifier, final_E_Supervised_Encoder, final_M_Supervised_Encoder,
                                                   final_scaler_gdsc)
    return external_AUC, external_AUCPR, test_AUC, test_AUCPR


def train_validate_hyperparameter_set(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val,
                                      device, hyperparameters, semi_hard_triplet, deactivate_skip_bad_iterations,
                                      same_dimension_latent_features):
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
    classifier_epoch = hyperparameters['classifier_epochs']
    mini_batch_size = hyperparameters['mini_batch']
    margin = hyperparameters['margin']
    triplet_selector = get_triplet_selector(margin, semi_hard_triplet)
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
        scalerGDSC = sk.StandardScaler()
        scalerGDSC.fit(X_trainE)
        X_trainE = scalerGDSC.transform(X_trainE)
        x_val_e = torch.FloatTensor(scalerGDSC.transform(x_val_e)).to(device)
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_trainE), torch.FloatTensor(X_trainM),
                                                      torch.FloatTensor(X_trainC),
                                                      torch.FloatTensor(Y_train.astype(int)))

        train_loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                                   num_workers=8, sampler=sampler, drop_last=True)

        IE_dim = X_trainE.shape[-1]
        IM_dim = X_trainM.shape[-1]
        IC_dim = X_trainC.shape[-1]

        e_supervised_encoder = SupervisedEncoder(IE_dim, OE_dim, encoder_dropout).to(device)
        m_supervised_encoder = SupervisedEncoder(IM_dim, OM_dim, encoder_dropout).to(device)
        c_supervised_encoder = SupervisedEncoder(IC_dim, OC_dim, encoder_dropout).to(device)

        E_optimizer = optim.Adagrad(e_supervised_encoder.parameters(), lr=lrE, weight_decay=encoder_weight_decay)
        M_optimizer = optim.Adagrad(m_supervised_encoder.parameters(), lr=lrM, weight_decay=encoder_weight_decay)
        C_optimizer = optim.Adagrad(c_supervised_encoder.parameters(), lr=lrC, weight_decay=encoder_weight_decay)

        # train each Supervised_Encoder with triplet loss
        train_encoder(E_Supervised_Encoder_epoch, E_optimizer, triplet_selector, device, e_supervised_encoder,
                      train_loader, trip_loss_fun, semi_hard_triplet, 0)
        train_encoder(M_Supervised_Encoder_epoch, M_optimizer, triplet_selector, device, m_supervised_encoder,
                      train_loader, trip_loss_fun, semi_hard_triplet, 1)
        train_encoder(C_Supervised_Encoder_epoch, C_optimizer, triplet_selector, device, c_supervised_encoder,
                      train_loader, trip_loss_fun, semi_hard_triplet, 2)

        # train classifier
        classifier_input_dimension = OE_dim + OM_dim + OC_dim
        val_auroc = train_validate_classifier(classifier_epoch, device, e_supervised_encoder,
                                              m_supervised_encoder, c_supervised_encoder, train_loader,
                                              classifier_input_dimension,
                                              classifier_dropout, lrCL, classifier_weight_decay,
                                              x_val_e, x_val_m, x_val_c, y_val)
        all_validation_aurocs.append(val_auroc)

        if not deactivate_skip_bad_iterations and iteration < parameter['cv_splits']:
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


def write_results_to_file(drug_name, extern_auc_list, extern_auprc_list, result_file, test_auc_list, test_auprc_list):
    result_dict = {
        'test auroc': test_auc_list,
        'test auprc': test_auprc_list,
        'extern auroc': extern_auc_list,
        'extern auprc': extern_auprc_list
    }
    result_file.write(f'\n test auroc list: {test_auc_list} \n')
    result_file.write(f'\n test auprc list: {test_auprc_list} \n')
    result_file.write(f'\n extern auroc list: {extern_auc_list} \n')
    result_file.write(f'\n extern auprc list: {extern_auprc_list} \n')
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)


def train_final(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val, best_hyperparameter,
                device, semi_hard_triplet, same_dimension_latent_features):
    E_dr = best_hyperparameter['encoder_dropout']
    C_dr = best_hyperparameter['classifier_dropout']
    Cwd = best_hyperparameter['classifier_weight_decay']
    Ewd = best_hyperparameter['encoder_weight_decay']
    lrE = best_hyperparameter['learning_rate_e']
    lrM = best_hyperparameter['learning_rate_m']
    lrC = best_hyperparameter['learning_rate_c']
    lrCL = best_hyperparameter['learning_rate_classifier']
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
                                               num_workers=8, sampler=sampler, drop_last=True)
    IE_dim = x_train_val_e.shape[-1]
    IM_dim = x_train_val_m.shape[-1]
    IC_dim = x_train_val_c.shape[-1]
    final_E_Supervised_Encoder = SupervisedEncoder(IE_dim, OE_dim, E_dr).to(device)
    final_M_Supervised_Encoder = SupervisedEncoder(IM_dim, OM_dim, E_dr).to(device)
    final_C_Supervised_Encoder = SupervisedEncoder(IC_dim, OC_dim, E_dr).to(device)
    E_optimizer = optim.Adagrad(final_E_Supervised_Encoder.parameters(), lr=lrE, weight_decay=Ewd)
    M_optimizer = optim.Adagrad(final_M_Supervised_Encoder.parameters(), lr=lrM, weight_decay=Ewd)
    C_optimizer = optim.Adagrad(final_C_Supervised_Encoder.parameters(), lr=lrC, weight_decay=Ewd)
    triplet_selector = get_triplet_selector(margin, semi_hard_triplet)
    OCP_dim = OE_dim + OM_dim + OC_dim
    final_classifier = Classifier(OCP_dim, C_dr).to(device)
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

    # train classifier
    train_classifier(final_classifier, classifier_epoch, train_loader, classifier_optimizer, final_E_Supervised_Encoder,
                     final_M_Supervised_Encoder, final_C_Supervised_Encoder,
                     device)
    return final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, final_classifier, \
           final_scaler_gdsc


def check_best_auroc(best_reachable_auroc):
    global best_auroc
    return best_reachable_auroc < best_auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--gpu_number', type=int)
    parser.add_argument('--drug', default='all', choices=['Gemcitabine_tcga', 'Gemcitabine_pdx', 'Cisplatin',
                                                          'Docetaxel', 'Erlotinib', 'Cetuximab', 'Paclitaxel'])
    parser.add_argument('--semi_hard_triplet', default=False, action='store_true')
    parser.add_argument('--deactivate_elbow_method', default=False, action='store_true')
    parser.add_argument('--combine_latent_features', default=False, action='store_true')
    parser.add_argument('--search_iterations', default=200, type=int)
    parser.add_argument('--sobol_iterations', default=50, type=int)
    parser.add_argument('--deactivate_skip_bad_iterations', default=False, action='store_true')
    parser.add_argument('--same_dimension_latent_features', default=False, action='store_true')
    parser.add_argument('--optimise_independent', default=False, action='store_true')
    parser.add_argument('--sampling_method', default='gp', choices=['gp', 'sobol', 'saasbo'])
    args = parser.parse_args()

    if args.drug == 'all':
        for drug, extern_dataset in parameter['drugs'].items():
            super_felt(args.experiment_name, drug, extern_dataset, args.gpu_number, args.search_iterations,
                       args.sobol_iterations, args.sampling_method, args.deactivate_elbow_method,
                       args.deactivate_skip_bad_iterations, args.semi_hard_triplet, args.combine_latent_features,
                       args.optimise_independent, args.same_dimension_latent_features)
    else:
        extern_dataset = parameter['drugs'][args.drug]
        super_felt(args.experiment_name, args.drug, extern_dataset, args.gpu_number, args.search_iterations,
                   args.sobol_iterations, args.sampling_method, args.deactivate_elbow_method,
                   args.deactivate_skip_bad_iterations, args.semi_hard_triplet, args.combine_latent_features,
                   args.optimise_independent, args.same_dimension_latent_features)
