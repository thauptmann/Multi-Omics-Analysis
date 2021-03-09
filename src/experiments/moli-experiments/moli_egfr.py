import sys
import argparse
import random
import numpy as np
import sklearn
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import WeightedRandomSampler
import tqdm
import tqdm.contrib
from pathlib import Path
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
from utils import network_training_util
from models.moli_model import Moli
from utils import egfr_data
from utils.visualisation import save_auroc_plots
from siamese_triplet.utils import AllTripletSelector
from utils import random_parameterization


def cv_and_train(run_test, random_search_iterations, load_checkpoint, experiment_name):
    # reproducibility
    random_seed = 42
    cv_splits = 5
    test_batch_size = 256
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        pin_memory = True
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")
        pin_memory = False

    result_path = Path('..', '..', '..', 'results', 'egfr', 'random_search', experiment_name)
    result_file = open(result_path / 'logs.txt', "a")
    result_path.mkdir(parents=True, exist_ok=True)

    data_path = Path('..', '..', '..', 'data')
    GDSCE, GDSCM, GDSCC, GDSCR, PDXEerlo, PDXMerlo, PDXCerlo, PDXRerlo, PDXEcet, PDXMcet, PDXCcet, PDXRcet = \
        egfr_data.load_data(data_path)

    skf = StratifiedKFold(n_splits=cv_splits)

    if load_checkpoint and Path(result_path / 'all_aucs').exists() \
            and Path(result_path / 'best_parameterization').exists():
        print("Load checkpoint...")
        all_aucs = np.load(result_path / 'all_aucs', allow_pickle=True)
        best_parameterization = pickle.load(open(result_path / 'best_parameterization', 'rb'))
        best_auc = max(all_aucs)

    else:
        print("Beginning from scratch...")
        best_auc = 0.5
        all_aucs = []

    for _ in tqdm.trange(len(all_aucs), random_search_iterations, desc='Random Search Iteration'):
        aucs_validate = []
        fold_number = 1
        parameterization = random_parameterization.create_random_parameterization()
        for train_index, test_index in skf.split(GDSCE, GDSCR):
            x_train_e = GDSCE[train_index]
            x_train_m = GDSCM[train_index]
            x_train_c = GDSCC[train_index]

            x_test_e = GDSCE[test_index]
            x_test_m = GDSCM[test_index]
            x_test_c = GDSCC[test_index]

            y_train = GDSCR[train_index]
            y_test = GDSCR[test_index]

            scaler_gdsc = sklearn.preprocessing.StandardScaler()
            x_train_e = scaler_gdsc.fit_transform(x_train_e)
            x_test_e = scaler_gdsc.transform(x_test_e)
            x_train_m = np.nan_to_num(x_train_m)
            x_train_c = np.nan_to_num(x_train_c)

            # Initialisation
            class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                            replacement=True)

            train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                                           torch.FloatTensor(x_train_c),
                                                           torch.FloatTensor(y_train))
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=parameterization['batch_size'],
                                                       shuffle=False,
                                                       num_workers=8, sampler=sampler, pin_memory=pin_memory)

            test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e), torch.FloatTensor(x_test_m),
                                                          torch.FloatTensor(x_test_c),
                                                          torch.FloatTensor(y_test))
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size,
                                                      shuffle=False, num_workers=8, pin_memory=pin_memory)

            n_sample_e, ie_dim = x_train_e.shape
            _, im_dim = x_train_m.shape
            _, ic_dim = x_train_c.shape

            all_triplet_selector = AllTripletSelector()

            moli_model = Moli([ie_dim, im_dim, ic_dim],
                              [parameterization['h_dim1'], parameterization['h_dim2'], parameterization['h_dim3']],
                              [parameterization['dropout_rate_e'], parameterization['dropout_rate_m'],
                               parameterization['dropout_rate_c'], parameterization['dropout_rate_clf']]).to(device)

            moli_optimiser = torch.optim.Adagrad([
                {'params': moli_model.expression_encoder.parameters(), 'lr': parameterization['lr_e']},
                {'params': moli_model.mutation_encoder.parameters(), 'lr': parameterization['lr_m']},
                {'params': moli_model.cna_encoder.parameters(), 'lr': parameterization['lr_c']},
                {'params': moli_model.classifier.parameters(), 'lr': parameterization['lr_cl'],
                 'weight_decay': parameterization['weight_decay']},
            ])

            trip_criterion = torch.nn.TripletMarginLoss(margin=parameterization['margin'], p=2)

            bce_loss = torch.nn.BCEWithLogitsLoss()
            for _ in tqdm.trange(parameterization['epochs'], desc='Epoch'):
                network_training_util.train(train_loader, moli_model, moli_optimiser,
                                            all_triplet_selector, trip_criterion, bce_loss,
                                            device, parameterization['gamma'])

            # validate
            auc_validate = network_training_util.validate(test_loader, moli_model, device)
            aucs_validate.append(auc_validate)

            # check for break
            if fold_number != cv_splits:
                splits_left = np.ones(cv_splits - fold_number)
                best_possible_result = np.mean(np.append(aucs_validate, splits_left))
                if best_possible_result < best_auc:
                    print("Experiment can't get better than the baseline. Skip next folds...")
                    break
            fold_number += 1

        auc_cv = np.mean(aucs_validate)
        all_aucs.append(auc_cv)
        pickle.dump(all_aucs, open(result_path / 'all_aucs', "wb"))

        if auc_cv > best_auc:
            best_auc = auc_cv
            best_parameterization = parameterization
            result_file.write(f'New best validation AUROC: {best_auc}\n')
            pickle.dump(best_parameterization, open(result_path / 'best_parameterization', "wb"))

    result_file.write(f'Best validation AUROC: {best_auc}\n')
    result_file.write(f'{best_parameterization=}\n')

    save_auroc_plots(all_aucs, result_path)
    pickle.dump(all_aucs, open(result_path / 'all_aucs', "wb"))
    pickle.dump(best_parameterization, open(result_path / 'best_parameterization', "wb"))

    # Test
    if run_test:
        x_train_e = GDSCE
        x_train_m = GDSCM
        x_train_c = GDSCC
        y_train = GDSCR

        x_test_e_erlo = PDXEerlo
        x_test_m_erlo = torch.FloatTensor(PDXMerlo)
        x_test_c_erlo = torch.FloatTensor(PDXCerlo)
        y_ts_erlo = PDXRerlo

        x_test_e_cet = PDXEcet
        x_test_m_cet = torch.FloatTensor(PDXMcet)
        x_test_c_cet = torch.FloatTensor(PDXCcet)
        y_ts_cet = PDXRcet

        train_scaler_gdsc = sklearn.preprocessing.StandardScaler()
        x_train_e = train_scaler_gdsc.fit_transform(x_train_e)
        x_test_e_cet = torch.FloatTensor(train_scaler_gdsc.transform(x_test_e_cet))
        x_test_e_erlo = torch.FloatTensor(train_scaler_gdsc.transform(x_test_e_erlo))

        y_ts_cet = torch.FloatTensor(y_ts_cet.astype(int))
        y_ts_erlo = torch.FloatTensor(y_ts_erlo.astype(int))

        _, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_c.shape

        all_triplet_selector = AllTripletSelector()
        moli_model = Moli([ie_dim, im_dim, ic_dim],
                          [best_parameterization['h_dim1'], best_parameterization['h_dim2'],
                           best_parameterization['h_dim3']],
                          [best_parameterization['dropout_rate_e'], best_parameterization['dropout_rate_m'],
                           best_parameterization['dropout_rate_c'], best_parameterization['dropout_rate_clf']])
        moli_model = moli_model.to(device)

        moli_optimiser = torch.optim.Adagrad([
            {'params': moli_model.expression_encoder.parameters(), 'lr': best_parameterization['lr_e']},
            {'params': moli_model.mutation_encoder.parameters(), 'lr': best_parameterization['lr_m']},
            {'params': moli_model.cna_encoder.parameters(), 'lr': best_parameterization['lr_c']},
            {'params': moli_model.classifier.parameters(), 'lr': best_parameterization['lr_cl'],
             'weight_decay': best_parameterization['weight_decay']},
        ])

        trip_criterion = torch.nn.TripletMarginLoss(margin=best_parameterization['margin'], p=2)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                                       torch.FloatTensor(x_train_c),
                                                       torch.FloatTensor(y_train))
        test_dataset_erlo = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e_erlo),
                                                           torch.FloatTensor(x_test_m_erlo),
                                                           torch.FloatTensor(x_test_c_erlo),
                                                           torch.FloatTensor(y_ts_erlo))
        test_dataset_cet = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e_cet),
                                                          torch.FloatTensor(x_test_m_cet),
                                                          torch.FloatTensor(x_test_c_cet), torch.FloatTensor(y_ts_cet))
        test_dataset_both = torch.utils.data.ConcatDataset((test_dataset_cet, test_dataset_erlo))

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=best_parameterization['batch_size'],
                                                   shuffle=False,
                                                   num_workers=8, sampler=sampler, pin_memory=pin_memory,
                                                   drop_last=True)
        test_loader_erlo = torch.utils.data.DataLoader(dataset=test_dataset_erlo,
                                                       batch_size=test_batch_size,
                                                       shuffle=False, num_workers=8, pin_memory=pin_memory)
        test_loader_cet = torch.utils.data.DataLoader(dataset=test_dataset_cet,
                                                      batch_size=test_batch_size,
                                                      shuffle=False, num_workers=8, pin_memory=pin_memory)
        test_loader_both = torch.utils.data.DataLoader(dataset=test_dataset_both,
                                                       batch_size=test_batch_size,
                                                       shuffle=False, num_workers=8, pin_memory=pin_memory)
        auc_train = 0
        for _ in range(best_parameterization['epochs']):
            auc_train = network_training_util.train(train_loader, moli_model, moli_optimiser,
                                                    all_triplet_selector, trip_criterion,
                                                    bce_loss, device, best_parameterization['gamma'])

        auc_test_erlo = network_training_util.validate(test_loader_erlo, moli_model, device)
        auc_test_cet = network_training_util.validate(test_loader_cet, moli_model, device)
        auc_test_both = network_training_util.validate(test_loader_both, moli_model, device)

        result_file.write(f'EGFR: AUROC Train = {auc_train}\n')
        result_file.write(f'EGFR Cetuximab: AUROC = {auc_test_cet}\n')
        result_file.write(f'EGFR Erlotinib: AUROC = {auc_test_erlo}\n')
        result_file.write(f'EGFR Erlotinib and Cetuximab: AUROC = {auc_test_both}\n')
        result_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_test', default=False, action='store_true')
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--random_search_iteration', default=1, type=int)
    parser.add_argument('--experiment_name', required=True)
    args = parser.parse_args()
    cv_and_train(args.run_test, args.random_search_iteration, args.load_checkpoint, args.experiment_name)
