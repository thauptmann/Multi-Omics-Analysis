import argparse
import sys
from pathlib import Path

import sklearn.preprocessing as sk
from ax import Models, optimize
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import optim
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.network_training_util import calculate_mean_and_std_auc, get_triplet_selector, feature_selection
from utils import multi_omics_data
from super_felt_model import SupervisedEncoder, Classifier

from utils.choose_gpu import get_free_gpu

drugs = {
    'Gemcitabine_tcga': 'TCGA',
    'Gemcitabine_pdx': 'PDX',
    'Cisplatin': 'TCGA',
    'Docetaxel': 'TCGA',
    'Erlotinib': 'PDX',
    'Cetuximab': 'PDX',
    'Paclitaxel': 'PDX'
}

learning_rate = 0.01
weight_decays = [0.0, 0.01, 0.05, 0.15, 0.1]
dropouts = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
epoch_lower = 2
epoch_upper = 10
dimension_choice = [32, 64, 128, 256, 512, 1024]
margin_choice = [0.2, 0.5, 1.0]
mini_batch_choice_encoder = [16, 32, 64]
mini_batch_choice_classifier = [16, 32, 64]
BCE_loss_fun = torch.nn.BCELoss()

random_seed = 42


def super_felt_optimise_independently(experiment_name, drug_name, extern_dataset_name, gpu_number,
                                      iterations):
    device = determine_device(gpu_number)
    set_random_seeds()

    result_file = create_result_file(drug_name, experiment_name)
    ExternalC, ExternalE, ExternalM, ExternalY, GDSCC, GDSCE, GDSCM, GDSCR = load_data(drug_name, extern_dataset_name)

    test_auc_list = []
    extern_auc_list = []
    test_auprc_list = []
    extern_auprc_list = []
    cv_splits = 5
    skf_outer = StratifiedKFold(n_splits=cv_splits, random_state=random_seed, shuffle=True)
    for train_index_outer, test_index in tqdm(skf_outer.split(GDSCE, GDSCR), total=skf_outer.get_n_splits(),
                                              desc=" Outer k-fold"):
        X_train_valE = torch.FloatTensor(GDSCE.to_numpy()[train_index_outer])
        X_testE = torch.FloatTensor(GDSCE.to_numpy()[test_index])
        X_train_valM = torch.FloatTensor(GDSCM.to_numpy()[train_index_outer])
        X_testM = torch.FloatTensor(GDSCM.to_numpy()[test_index])
        X_train_valC = torch.FloatTensor(GDSCC.to_numpy()[train_index_outer])
        X_testC = torch.FloatTensor(GDSCC.to_numpy()[test_index])
        Y_train_val = GDSCR[train_index_outer]
        Y_test = GDSCR[test_index]
        skf = StratifiedKFold(n_splits=cv_splits)

        splits = list(skf.split(np.zeros(len(Y_train_val)), Y_train_val))
        evaluation_function_e = lambda parameterization: train_validate_encoder(
            parameterization, X_train_valE, Y_train_val, splits, device)
        evaluation_function_m = lambda parameterization: train_validate_encoder(parameterization, X_train_valM,
                                                                                Y_train_val, splits, device)
        evaluation_function_c = lambda parameterization: train_validate_encoder(parameterization, X_train_valC,
                                                                                Y_train_val, splits, device)
        generation_strategy_e = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=iterations),
            ],
            name="Sobol"
        )

        generation_strategy_m = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=iterations),
            ],
            name="Sobol"
        )

        generation_strategy_c = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=iterations),
            ],
            name="Sobol"
        )

        generation_strategy_classifier = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=iterations),
            ],
            name="Sobol"
        )

        encoder_search_space = get_encoder_search_space()

        best_parameters_e, values, experiment, model = optimize(
            total_trials=iterations,
            experiment_name='encoder_e',
            objective_name='triplet',
            parameters=encoder_search_space,
            evaluation_function=evaluation_function_e,
            minimize=False,
            generation_strategy=generation_strategy_e,
        )

        best_parameters_m, values, experiment, model = optimize(
            total_trials=iterations,
            experiment_name='encoder_m',
            objective_name='triplet',
            parameters=encoder_search_space,
            evaluation_function=evaluation_function_m,
            minimize=False,
            generation_strategy=generation_strategy_m,
        )

        best_parameters_c, values, experiment, model = optimize(
            total_trials=iterations,
            experiment_name='encoder_c',
            objective_name='triplet',
            parameters=encoder_search_space,
            evaluation_function=evaluation_function_c,
            minimize=False,
            generation_strategy=generation_strategy_c,
        )

        # retrain best encoder
        best_encoder_e, scaler_e = final_training_encoder(best_parameters_e, X_train_valE, Y_train_val, device)
        best_encoder_m, scaler_m = final_training_encoder(best_parameters_m, X_train_valM, Y_train_val, device)
        best_encoder_c, scaler_c = final_training_encoder(best_parameters_c, X_train_valC, Y_train_val, device)

        best_encoder_e.eval()
        best_encoder_m.eval()
        best_encoder_c.eval()

        input_dimension = best_parameters_e['dimension'] + best_parameters_m['dimension'] \
                          + best_parameters_c['dimension']

        evaluation_function_classifier = lambda parameterization: train_validate_classifier_hyperparameter_set(
            parameterization, X_train_valE, X_train_valM, X_train_valC, Y_train_val, best_encoder_e, best_encoder_m,
            best_encoder_c, scaler_e, scaler_m, scaler_c, splits, input_dimension, device
        )

        classifier_search_space = get_classifier_search_space()
        best_parameters_classifier, values, experiment, model = optimize(
            total_trials=iterations,
            experiment_name='classifier',
            objective_name='auroc',
            parameters=classifier_search_space,
            evaluation_function=evaluation_function_classifier,
            minimize=False,
            generation_strategy=generation_strategy_classifier,
        )

        # retrain best classifier
        best_classifier = final_training_classifier(best_parameters_classifier, X_train_valE, X_train_valM,
                                                    X_train_valC,
                                                    Y_train_val, best_encoder_e, best_encoder_m, best_encoder_c,
                                                    scaler_e, scaler_m, scaler_c, input_dimension, device)

        best_classifier.eval()

        # Test
        X_testE = torch.FloatTensor(scaler_e.transform(X_testE))
        X_testM = torch.FloatTensor(scaler_m.transform(X_testM))
        X_testC = torch.FloatTensor(scaler_c.transform(X_testC))
        encoded_test_E = best_encoder_e(torch.FloatTensor(X_testE).to(device))
        encoded_test_M = best_encoder_m(torch.FloatTensor(X_testM).to(device))
        encoded_test_C = best_encoder_c(torch.FloatTensor(X_testC).to(device))
        test_Pred = best_classifier(encoded_test_E, encoded_test_M, encoded_test_C)
        test_y_pred = test_Pred.cpu().detach().numpy()
        test_AUC = roc_auc_score(Y_test, test_y_pred)
        test_AUCPR = average_precision_score(Y_test, test_y_pred)

        # Extern
        ExternalE = torch.FloatTensor(scaler_e.transform(ExternalE))
        ExternalM = torch.FloatTensor(scaler_m.transform(ExternalM))
        ExternalC = torch.FloatTensor(scaler_c.transform(ExternalC))

        encoded_external_E = best_encoder_e(torch.FloatTensor(ExternalE).to(device))
        encoded_external_M = best_encoder_m(torch.FloatTensor(ExternalM).to(device))
        encoded_external_C = best_encoder_c(torch.FloatTensor(ExternalC).to(device))
        external_Pred = best_classifier(encoded_external_E, encoded_external_M, encoded_external_C)
        external_y_true = ExternalY
        external_y_pred = external_Pred.cpu().detach().numpy()
        external_AUC = roc_auc_score(external_y_true, external_y_pred)
        external_AUCPR = average_precision_score(external_y_true, external_y_pred)

        test_auc_list.append(test_AUC)
        extern_auc_list.append(external_AUC)
        test_auprc_list.append(test_AUCPR)
        extern_auprc_list.append(external_AUCPR)

    print("Done!")

    result_dict = {
        'test auroc': test_auc_list,
        'test auprc': test_auprc_list,
        'extern auroc': extern_auc_list,
        'extern auprc': extern_auprc_list
    }
    calculate_mean_and_std_auc(result_dict, result_file, drug_name)
    result_file.write(f'\n test auroc list: {test_auc_list} \n')
    result_file.write(f'\n test auprc list: {test_auprc_list} \n')
    result_file.write(f'\n extern auroc list: {extern_auc_list} \n')
    result_file.write(f'\n extern auprc list: {extern_auprc_list} \n')
    result_file.close()


def train_validate_encoder(hyperparameters, x_train_validation, y_train_validation, splits, device):
    margin = hyperparameters['margin']
    output_dimension = hyperparameters['dimension']
    dropout = hyperparameters['dropout']
    epochs = hyperparameters['epochs']
    weight_decay = hyperparameters['weight_decay']
    mini_batch_size = hyperparameters['mini_batch_size']
    # triplet_selector = get_triplet_selector(margin, 'semi_hard')
    triplet_selector = get_triplet_selector(margin, 'all')
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)
    loss_list = list()
    input_dimension = x_train_validation.shape[-1]
    supervised_encoder = SupervisedEncoder(input_dimension, output_dimension, dropout)
    supervised_encoder.to(device)
    optimizer = optim.Adagrad(supervised_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for train_index, validation_index in tqdm(splits, desc="k-fold"):
        X_train = x_train_validation[train_index]
        X_validation = x_train_validation[validation_index]
        y_train = y_train_validation[train_index]
        y_validation = y_train_validation[validation_index]
        scaler = sk.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_validation = torch.FloatTensor(scaler.transform(X_validation))
        sampler = create_sampler(y_train)
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train),
                                                      torch.FloatTensor(y_train.astype(int)))
        trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                                  num_workers=1, sampler=sampler)
        for _ in range(epochs):
            supervised_encoder.train()
            for i, (data, target) in enumerate(trainLoader):
                if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                    data = data.to(device)
                    encoded_data = supervised_encoder(data)

                    triplets_list = triplet_selector.get_triplets(encoded_data, target)
                    loss = trip_loss_fun(encoded_data[triplets_list[:, 0], :],
                                         encoded_data[triplets_list[:, 1], :],
                                         encoded_data[triplets_list[:, 2], :])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            supervised_encoder.eval()
            encoded_val = supervised_encoder(X_validation.to(device))
            triplets_list = triplet_selector.get_triplets(encoded_val, torch.FloatTensor(y_validation))
            val_loss = trip_loss_fun(encoded_val[triplets_list[:, 0], :],
                                     encoded_val[triplets_list[:, 1], :],
                                     encoded_val[triplets_list[:, 2], :])
        loss_list.append(val_loss.cpu().detach().numpy())

    return np.mean(loss_list)


def create_sampler(y_train):
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight),
                                    replacement=True)
    return sampler


def train_validate_classifier_hyperparameter_set(hyperparameters, x_train_validation_e, x_train_validation_m,
                                                 x_train_validation_c, y_train_validation,
                                                 best_encoder_e, best_encoder_m, best_encoder_c,
                                                 scaler_e, scaler_m, scaler_c,
                                                 splits, input_dimension,  device):
    dropout = hyperparameters['dropout']
    epochs = hyperparameters['epochs']
    weight_decay = hyperparameters['weight_decay']
    mini_batch_size = hyperparameters['mini_batch_size']
    auroc_list = list()
    classifier = Classifier(input_dimension, dropout)
    classifier.to(device)
    optimizer = optim.Adagrad(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for train_index, validation_index in tqdm(splits, desc="k-fold"):
        e_train = x_train_validation_e[train_index]
        e_validation = x_train_validation_e[validation_index]
        e_train = scaler_e.transform(e_train)
        e_validation = scaler_e.transform(e_validation)

        m_train = x_train_validation_m[train_index]
        m_validation = x_train_validation_m[validation_index]
        m_train = scaler_m.transform(m_train)
        m_validation = scaler_m.transform(m_validation)

        c_train = x_train_validation_c[train_index]
        c_validation = x_train_validation_c[validation_index]
        c_train = scaler_c.transform(c_train)
        c_validation = scaler_c.transform(c_validation)

        y_train = y_train_validation[train_index]
        y_validation = y_train_validation[validation_index]

        sampler = create_sampler(y_train)
        trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(e_train), torch.FloatTensor(m_train),
                                                      torch.FloatTensor(c_train),
                                                      torch.FloatTensor(y_train.astype(int)))
        trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                                  sampler=sampler)
        for _ in range(epochs):
            classifier.train()
            for i, (e, m, c, target) in enumerate(trainLoader):
                e = e.to(device)
                m = m.to(device)
                c = c.to(device)
                target = target.to(device)
                encoded_e = best_encoder_e(e)
                encoded_m = best_encoder_m(m)
                encoded_c = best_encoder_c(c)
                predictions = classifier(encoded_e, encoded_m, encoded_c)
                loss = BCE_loss_fun(torch.squeeze(predictions), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            classifier.eval()
            encoded_e = best_encoder_e(e_validation.to(device))
            encoded_m = best_encoder_m(m_validation.to(device))
            encoded_c = best_encoder_c(c_validation.to(device))
            test_prediction = classifier(encoded_e, encoded_m, encoded_c)
            val_AUC = roc_auc_score(y_validation, test_prediction.cpu().detach().numpy())
        auroc_list.append(val_AUC)
    return np.mean(auroc_list)


def final_training_encoder(best_parameter, data, y, device):
    sampler = create_sampler(y)
    margin = best_parameter['margin']
    output_dimension = best_parameter['dimension']
    dropout = best_parameter['dropout']
    epochs = best_parameter['epochs']
    weight_decay = best_parameter['weight_decay']
    mini_batch_size = best_parameter['mini_batch_size']
    scaler = sk.StandardScaler()
    scaled_data = scaler.fit_transform(data)
    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(scaled_data), torch.FloatTensor(y.astype(int)))
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                              sampler=sampler)
    input_dimension = scaled_data.shape[-1]

    supervised_encoder = SupervisedEncoder(input_dimension, output_dimension, dropout)
    supervised_encoder.to(device)
    optimizer = optim.Adagrad(supervised_encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # triplet_selector = get_triplet_selector(margin, 'semi_hard')
    triplet_selector = get_triplet_selector(margin, 'all')
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)

    for _ in range(epochs):
        supervised_encoder.train()
        for i, (scaled_data, target) in enumerate(trainLoader):
            if torch.mean(target) != 0. and torch.mean(target) != 1. and len(target) > 2:
                scaled_data = scaled_data.to(device)
                encoded_data = supervised_encoder(scaled_data)

                triplets_list = triplet_selector.get_triplets(encoded_data, target)
                loss = trip_loss_fun(encoded_data[triplets_list[:, 0], :],
                                     encoded_data[triplets_list[:, 1], :],
                                     encoded_data[triplets_list[:, 2], :])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return supervised_encoder, scaler


def final_training_classifier(best_hyperparameters, x_train_validation_e, x_train_validation_m,
                              x_train_validation_c, y_train_validation, best_encoder_e, best_encoder_m, best_encoder_c,
                              scaler_e, scaler_m, scaler_c, input_dimension, device):
    dropout = best_hyperparameters['dropout']
    epochs = best_hyperparameters['epochs']
    weight_decay = best_hyperparameters['weight_decay']
    mini_batch_size = best_hyperparameters['mini_batch_size']
    classifier = Classifier(input_dimension, dropout)
    optimizer = optim.Adagrad(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    classifier.to(device)

    x_train_validation_e = scaler_e.transform(x_train_validation_e)
    x_train_validation_m = scaler_m.transform(x_train_validation_m)
    x_train_validation_c = scaler_c.transform(x_train_validation_c)

    sampler = create_sampler(y_train_validation)
    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_validation_e),
                                                  torch.FloatTensor(x_train_validation_m),
                                                  torch.FloatTensor(x_train_validation_c),
                                                  torch.FloatTensor(y_train_validation.astype(int)))
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch_size, shuffle=False,
                                              sampler=sampler)
    for _ in range(epochs):
        classifier.train()
        for i, (e, m, c, target) in enumerate(trainLoader):
            e = e.to(device)
            m = m.to(device)
            c = c.to(device)
            target = target.to(device)

            encoded_e = best_encoder_e(e)
            encoded_m = best_encoder_m(m)
            encoded_c = best_encoder_c(c)
            predictions = classifier(encoded_e, encoded_m, encoded_c)
            loss = BCE_loss_fun(torch.squeeze(predictions), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return classifier


def set_random_seeds():
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def determine_device(gpu_number):
    if torch.cuda.is_available():
        if gpu_number is None:
            free_gpu_id = get_free_gpu()
        else:
            free_gpu_id = gpu_number
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")
    return device


def load_data(drug_name, extern_dataset_name):
    data_path = Path('..', '..', '..', 'data')
    gdsc_e, gdsc_m, gdsc_c, gdsc_r, extern_e, extern_m, extern_c, extern_r \
        = multi_omics_data.load_drug_data(data_path, drug_name, extern_dataset_name, return_data_frames=True)
    GDSCE, GDSCM, GDSCC = feature_selection(gdsc_e, gdsc_m, gdsc_c)
    expression_intersection_genes_index = GDSCE.columns.intersection(extern_e.columns)
    mutation_intersection_genes_index = GDSCM.columns.intersection(extern_m.columns)
    cna_intersection_genes_index = GDSCC.columns.intersection(extern_c.columns)
    GDSCR = gdsc_r
    ExternalE = extern_e.loc[:, expression_intersection_genes_index]
    ExternalM = extern_m.loc[:, mutation_intersection_genes_index]
    ExternalC = extern_c.loc[:, cna_intersection_genes_index]
    ExternalY = extern_r
    return ExternalC, ExternalE, ExternalM, ExternalY, GDSCC, GDSCE, GDSCM, GDSCR


def create_result_file(drug_name, experiment_name):
    result_path = Path('..', '..', '..', 'results', 'super.felt', drug_name, experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)
    result_file = open(result_path / 'results.txt', 'w')
    return result_file


def get_encoder_search_space():
    return [
        {'name': 'dropout', 'values': [0.1, 0.3, 0.4, 0.5], 'type': 'choice'},
        {'name': 'weight_decay', 'values': [0.0, 0.01, 0.1, 0.15], 'type': 'choice'},
        {'name': 'margin', 'values': margin_choice, 'type': 'choice'},
        {'name': 'dimension', 'values': dimension_choice, 'type': 'choice'},
        {'name': 'mini_batch_size', 'values': mini_batch_choice_encoder, 'type': 'choice'},
        {'name': 'epochs', 'bounds': [epoch_lower, epoch_upper], 'type': 'range'}
    ]


def get_classifier_search_space():
    return [
        {'name': 'dropout', 'values': [0.1, 0.3, 0.4, 0.5], 'type': 'choice'},
        {'name': 'weight_decay', 'values': [0.0, 0.01, 0.1, 0.15], 'type': 'choice'},
        {'name': 'epochs', 'bounds': [epoch_lower, epoch_upper], 'type': 'range'},
        {'name': 'mini_batch_size', 'values': mini_batch_choice_classifier, 'type': 'choice'},
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--gpu_number', type=int)
    parser.add_argument('--iterations', default=2, type=int)
    parser.add_argument('--drug', default='all', choices=['Gemcitabine_tcga', 'Gemcitabine_pdx', 'Cisplatin',
                                                          'Docetaxel', 'Erlotinib', 'Cetuximab', 'Paclitaxel'])
    parser.add_argument('--triplet_selector_type', default='all', choices=['all', 'hardest', 'random', 'semi_hard',
                                                                           'none'])
    args = parser.parse_args()

    for drug, extern_dataset in drugs.items():
        super_felt_optimise_independently(args.experiment_name, drug, extern_dataset, args.gpu_number,
                                          args.iterations)
