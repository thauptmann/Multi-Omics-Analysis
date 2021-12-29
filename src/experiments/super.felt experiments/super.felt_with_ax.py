import argparse
import sys
from pathlib import Path

import sklearn.preprocessing as sk
import yaml
from ax import optimize
from scipy.stats import sem
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import optim
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from utils.experiment_utils import create_generation_strategy
from utils.searchspaces import get_super_felt_search_space
from super_felt_model import SupervisedEncoder, Classifier
from utils.network_training_util import calculate_mean_and_std_auc, get_triplet_selector
from utils import multi_omics_data
from utils.choose_gpu import get_free_gpu

with open(Path('../../config/hyperparameter.yaml'), 'r') as stream:
    parameter = yaml.safe_load(stream)
best_auroc = 0


def super_felt(experiment_name, drug_name, extern_dataset_name, gpu_number, search_iterations, sobol_iterations,
               sampling_method, deactivate_elbow_method, deactivate_skip_bad_iterations, semi_hard_triplet,
               combine_latent_features, optimise_independent):
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
    test_auprc_list = []
    extern_auprc_list = []

    skf_outer = StratifiedKFold(n_splits=parameter['cv_splits'], random_state=random_seed, shuffle=True)
    for train_index_outer, test_index in tqdm(skf_outer.split(gdsc_e, gdsc_r), total=skf_outer.get_n_splits(),
                                              desc=" Outer k-fold"):
        global best_auroc
        best_auroc = 0
        X_train_valE = gdsc_e[train_index_outer]
        X_testE = gdsc_e[test_index]
        X_train_valM = gdsc_m[train_index_outer]
        X_testM = gdsc_m[test_index]
        X_train_valC = gdsc_c[train_index_outer]
        X_testC = gdsc_c[test_index]
        Y_train_val = gdsc_r[train_index_outer]
        Y_test = gdsc_r[test_index]
        evaluation_function = lambda parameterization: train_validate_hyperparameter_set(X_train_valE,
                                                                                         X_train_valM, X_train_valC,
                                                                                         Y_train_val, device,
                                                                                         parameterization,
                                                                                         semi_hard_triplet,
                                                                                         deactivate_skip_bad_iterations)

        generation_strategy = create_generation_strategy(sampling_method, sobol_iterations, random_seed)
        search_space = get_super_felt_search_space(semi_hard_triplet)
        best_parameters, values, experiment, model = optimize(
            total_trials=search_iterations,
            experiment_name='Super.FELT',
            objective_name='auroc',
            parameters=search_space,
            evaluation_function=evaluation_function,
            minimize=False,
            generation_strategy=generation_strategy,
        )

        # retrain best
        final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, final_Classifier, \
        final_scaler_gdsc = train_final(X_train_valE, X_train_valM, X_train_valC, Y_train_val, best_parameters, device,
                                        semi_hard_triplet)

        # Test
        test_AUC, test_AUCPR = test(X_testE, X_testM, X_testC, Y_test, device, final_C_Supervised_Encoder,
                                    final_Classifier, final_E_Supervised_Encoder, final_M_Supervised_Encoder,
                                    final_scaler_gdsc)

        # Extern
        external_AUC, external_AUCPR = test(extern_e, extern_m, extern_c, extern_r, device,
                                            final_C_Supervised_Encoder,
                                            final_Classifier, final_E_Supervised_Encoder, final_M_Supervised_Encoder,
                                            final_scaler_gdsc)

        test_auc_list.append(test_AUC)
        extern_auc_list.append(external_AUC)
        test_auprc_list.append(test_AUCPR)
        extern_auprc_list.append(external_AUCPR)

    write_results_to_file(drug_name, extern_auc_list, extern_auprc_list, result_file, test_auc_list, test_auprc_list)
    result_file.close()
    print("Done!")


def train_validate_hyperparameter_set(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val,
                                      device, hyperparameters, semi_hard_triplet, deactivate_skip_bad_iterations):
    bce_loss_function = torch.nn.BCELoss()
    skf = StratifiedKFold(n_splits=parameter['cv_splits'])
    all_validation_aurocs = []
    encoder_dropout = hyperparameters['encoder_dropout']
    classifier_dropout = hyperparameters['classifier_dropout']
    classifier_weight_decay = hyperparameters['classifier_weight_decay']
    encoder_weight_decay = hyperparameters['encoder_weight_decay']
    lrE = hyperparameters['learning_rate']
    lrM = hyperparameters['learning_rate']
    lrC = hyperparameters['learning_rate']
    lrCL = hyperparameters['learning_rate']
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
        class_sample_count = np.array([len(np.where(Y_train == t)[0]) for t in np.unique(Y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in Y_train])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)
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

        E_optimizer = optim.Adagrad(E_Supervised_Encoder.parameters(), lr=lrE, weight_decay=encoder_weight_decay)
        M_optimizer = optim.Adagrad(M_Supervised_Encoder.parameters(), lr=lrM, weight_decay=encoder_weight_decay)
        C_optimizer = optim.Adagrad(C_Supervised_Encoder.parameters(), lr=lrC, weight_decay=encoder_weight_decay)

        OCP_dim = OE_dim + OM_dim + OC_dim
        classifier = Classifier(OCP_dim, classifier_dropout).to(device)
        classifier_optimizer = optim.Adagrad(classifier.parameters(), lr=lrCL, weight_decay=classifier_weight_decay)

        # train each Supervised_Encoder with triplet loss
        train_encoder(E_Supervised_Encoder_epoch, E_optimizer, triplet_selector, device, E_Supervised_Encoder,
                      trainLoader, trip_loss_fun, 0, semi_hard_triplet)
        train_encoder(M_Supervised_Encoder_epoch, M_optimizer, triplet_selector, device, M_Supervised_Encoder,
                      trainLoader, trip_loss_fun, 1, semi_hard_triplet)
        train_encoder(C_Supervised_Encoder_epoch, C_optimizer, triplet_selector, device, C_Supervised_Encoder,
                      trainLoader, trip_loss_fun, 2, semi_hard_triplet)

        # train classifier
        val_auroc = 0
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
                Pred = classifier(encoded_E, encoded_M, encoded_C)
                cl_loss = bce_loss_function(Pred, target.view(-1, 1))
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
                test_Pred = classifier(encoded_val_E, encoded_val_M, encoded_val_C)
                test_y_true = Y_val
                test_y_pred = test_Pred.cpu()
                val_auroc = roc_auc_score(test_y_true, test_y_pred.detach().numpy())

                if not deactivate_skip_bad_iterations:
                    open_folds = parameter['cv_splits'] - iteration
                    remaining_best_results = np.ones(open_folds)
                    best_possible_mean = np.mean(np.concatenate([val_auroc, remaining_best_results]))
                    if check_best_auroc(best_possible_mean):
                        print('Skip remaining folds.')
                        break
                iteration += 1

        all_validation_aurocs.append(val_auroc)
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


def test(x_test_e, x_test_m, x_test_c, y_test, device, final_c_supervised_encoder, final_classifier,
         final_e_supervised_encoder, final_m_supervised_encoder, final_scaler_gdsc):
    x_test_e = torch.FloatTensor(final_scaler_gdsc.transform(x_test_e))
    encoded_test_E = final_e_supervised_encoder(torch.FloatTensor(x_test_e).to(device))
    encoded_test_M = final_m_supervised_encoder(torch.FloatTensor(x_test_m).to(device))
    encoded_test_C = final_c_supervised_encoder(torch.FloatTensor(x_test_c).to(device))
    test_Pred = final_classifier(encoded_test_E, encoded_test_M, encoded_test_C)
    test_y_true = y_test
    test_y_pred = test_Pred.cpu().detach().numpy()
    test_AUC = roc_auc_score(test_y_true, test_y_pred)
    test_AUCPR = average_precision_score(test_y_true, test_y_pred)
    return test_AUC, test_AUCPR


def train_final(x_train_val_e, x_train_val_m, x_train_val_c, y_train_val, best_hyperparameter,
                device, semi_hard_triplet):
    bce_loss_function = torch.nn.BCELoss()
    E_dr = best_hyperparameter['encoder_dropout']
    C_dr = best_hyperparameter['classifier_dropout']
    Cwd = best_hyperparameter['classifier_weight_decay']
    Ewd = best_hyperparameter['encoder_weight_decay']
    lrE = best_hyperparameter['learning_rate']
    lrM = best_hyperparameter['learning_rate']
    lrC = best_hyperparameter['learning_rate']
    lrCL = best_hyperparameter['learning_rate']
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
    class_sample_count = np.array([len(np.where(y_train_val == t)[0]) for t in np.unique(y_train_val)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train_val])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)
    final_scaler_gdsc = sk.StandardScaler()
    final_scaler_gdsc.fit(x_train_val_e)
    x_train_val_e = final_scaler_gdsc.transform(x_train_val_e)
    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_val_e), torch.FloatTensor(x_train_val_m),
                                                  torch.FloatTensor(x_train_val_c),
                                                  torch.FloatTensor(y_train_val.astype(int)))
    train_Loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mb_size, shuffle=False,
                                               num_workers=1, sampler=sampler, drop_last=True)
    n_sample_E, IE_dim = x_train_val_e.shape
    n_sample_M, IM_dim = x_train_val_m.shape
    n_sample_C, IC_dim = x_train_val_c.shape
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
                  train_Loader,
                  trip_loss_fun, 0, semi_hard_triplet)
    train_encoder(M_Supervised_Encoder_epoch, M_optimizer, triplet_selector, device, final_M_Supervised_Encoder,
                  train_Loader,
                  trip_loss_fun, 1, semi_hard_triplet)
    train_encoder(C_Supervised_Encoder_epoch, C_optimizer, triplet_selector, device, final_C_Supervised_Encoder,
                  train_Loader,
                  trip_loss_fun, 2, semi_hard_triplet)

    # train classifier
    train_classifier(bce_loss_function, classifier_epoch, classifier_optimizer, device, final_C_Supervised_Encoder,
                     final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_classifier, train_Loader)
    return final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, final_classifier, \
           final_scaler_gdsc


def train_classifier(bce_loss_function, classifier_epoch, classifier_optimizer, device, final_c_supervised_encoder,
                     final_e_supervised_encoder, final_m_supervised_encoder, classifier, train_loader):
    classifier.train()
    for epoch in range(classifier_epoch):
        for dataE, dataM, dataC, target in train_loader:
            classifier_optimizer.zero_grad()
            dataE = dataE.to(device)
            dataM = dataM.to(device)
            dataC = dataC.to(device)
            target = target.to(device)
            encoded_E = final_e_supervised_encoder(dataE)
            encoded_M = final_m_supervised_encoder(dataM)
            encoded_C = final_c_supervised_encoder(dataC)

            predictions = classifier(encoded_E, encoded_M, encoded_C)
            cl_loss = bce_loss_function(predictions, target.view(-1, 1))
            cl_loss.backward()
            classifier_optimizer.step()
    classifier.eval()


def train_encoder(supervised_encoder_epoch, optimizer, triplet_selector, device, supervised_encoder, train_loader,
                  trip_loss_fun, omic_number, semi_hard_triplet):
    supervised_encoder.train()
    for epoch in range(supervised_encoder_epoch):
        last_epochs = False if epoch < supervised_encoder_epoch - 2 else True
        for all_data in train_loader:
            target = all_data[-1]
            data = all_data[omic_number]
            optimizer.zero_grad()
            data = data.to(device)
            encoded_data = supervised_encoder(data)
            if not last_epochs and semi_hard_triplet:
                triplets = triplet_selector[0].get_triplets(encoded_data, target)
            elif last_epochs and semi_hard_triplet:
                triplets = triplet_selector[1].get_triplets(encoded_data, target)
            else:
                triplets = triplet_selector.get_triplets(encoded_data, target)
            loss = trip_loss_fun(encoded_data[triplets[:, 0], :],
                                 encoded_data[triplets[:, 1], :],
                                 encoded_data[triplets[:, 2], :])
            loss.backward()
            optimizer.step()
    supervised_encoder.eval()


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

    parser.add_argument('--optimise_independent', default=False, action='store_true')
    parser.add_argument('--sampling_method', default='gp', choices=['gp', 'sobol', 'saasbo'])
    args = parser.parse_args()

    for drug, extern_dataset in parameter['drugs'].items():
        super_felt(args.experiment_name, drug, extern_dataset, args.gpu_number, args.search_iterations,
                   args.sobol_iterations, args.sampling_method, args.deactivate_elbow_method,
                   args.deactivate_skip_bad_iterations, args.semi_hard_triplet, args.combine_latent_features,
                   args.optimise_independent)
