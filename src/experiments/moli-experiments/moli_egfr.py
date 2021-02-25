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

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.choose_gpu import get_free_gpu
from utils import network_training_util
from models.moli_model import Moli
from utils import egfr_data
from utils.visualisation import save_auroc_plots
from siamese_triplet.utils import AllTripletSelector

mini_batch_list = [8, 16, 32, 64]
dim_list = [16, 32, 64, 128, 256, 512, 1024]
margin_list = [0.5, 1, 1.5, 2, 2.5]
learning_rate_list = [0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
epoch_list = [10, 20, 50, 30, 40, 60, 70, 80]
drop_rate_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
weight_decay_list = [0.1, 0.01, 0.001, 0.0001]
gamma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
cv_splits = 5


def cv_and_train(run_test, random_search_iterations, load_checkpoint, experiment_name):
    # reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    if torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")

    result_path = Path('..', '..', '..', 'results', 'egfr', 'random_search', experiment_name)
    result_path.mkdir(parents=True, exist_ok=True)

    data_path = Path('..', '..', '..', 'data')
    egfr_path = Path(data_path, 'EGFR_experiments_data')
    GDSCE, GDSCM, GDSCC, GDSCR, PDXEerlo, PDXMerlo, PDXCerlo, PDXRerlo, PDXEcet, PDXMcet, PDXCcet, PDXRcet = \
        egfr_data.load_data(egfr_path)

    skf = StratifiedKFold(n_splits=cv_splits)

    if load_checkpoint:
        first_iteration = False
        all_aucs = np.load(result_path / 'all_aucs')
        best_auc = max(all_aucs)
    else:
        first_iteration = True
        best_auc = 0
        all_aucs = []

    for iteration in tqdm.trange(len(all_aucs), random_search_iterations, desc='Random Search Iteration'):
        mini_batch = random.choice(mini_batch_list)
        h_dim1 = random.choice(dim_list)
        h_dim2 = random.choice(dim_list)
        h_dim3 = random.choice(dim_list)
        lr_e = random.choice(learning_rate_list)
        lr_m = random.choice(learning_rate_list)
        lr_c = random.choice(learning_rate_list)
        lr_cl = random.choice(learning_rate_list)
        dropout_rate_e = random.choice(drop_rate_list)
        dropout_rate_m = random.choice(drop_rate_list)
        dropout_rate_c = random.choice(drop_rate_list)
        weight_decay = random.choice(weight_decay_list)
        dropout_rate_clf = random.choice(drop_rate_list)
        gamma = random.choice(gamma_list)
        epochs = random.choice(epoch_list)
        margin = random.choice(margin_list)

        aucs_validate = []
        fold_number = 1
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
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=mini_batch,
                                                       shuffle=False,
                                                       num_workers=8, sampler=sampler, pin_memory=True)

            test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e), torch.FloatTensor(x_test_m),
                                                          torch.FloatTensor(x_test_c),
                                                          torch.FloatTensor(y_test))
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=mini_batch,
                                                      shuffle=False, num_workers=8, pin_memory=True)

            n_samp_e, ie_dim = x_train_e.shape
            _, im_dim = x_train_m.shape
            _, ic_dim = x_train_c.shape

            all_triplet_selector = AllTripletSelector()

            moli_model = Moli([ie_dim, im_dim, ic_dim],
                              [h_dim1, h_dim2, h_dim3],
                              [dropout_rate_e, dropout_rate_m, dropout_rate_c, dropout_rate_clf]).to(device)

            moli_optimiser = torch.optim.Adagrad([
                {'params': moli_model.expression_encoder.parameters(), 'lr': lr_e},
                {'params': moli_model.mutation_encoder.parameters(), 'lr': lr_m},
                {'params': moli_model.cna_encoder.parameters(), 'lr': lr_c},
                {'params': moli_model.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
            ])

            trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

            bce_loss = torch.nn.BCEWithLogitsLoss()
            for _ in tqdm.trange(epochs, desc='Epoch'):
                auc_train, cost_train = network_training_util.train(train_loader, moli_model, moli_optimiser,
                                                                    all_triplet_selector, trip_criterion, bce_loss,
                                                                    device, gamma)

            # validate
            auc_validate = network_training_util.validate(test_loader, moli_model, device)
            aucs_validate.append(auc_validate)

            # check for break
            if fold_number != cv_splits and not first_iteration:
                splits_left = np.ones(cv_splits - fold_number)
                best_possible_result = np.mean(np.append(aucs_validate, splits_left))
                if best_possible_result < best_auc:
                    break
            fold_number += 1

        auc_cv = np.mean(aucs_validate)

        if iteration % 10 == 0:
            save_auroc_plots(np.array([all_aucs]), result_path)

        all_aucs.append(auc_cv)
        first_iteration = False
        if auc_cv > best_auc:
            best_auc = auc_cv
            best_mini_batch = mini_batch
            best_h_dim1 = h_dim1
            best_h_dim2 = h_dim2
            best_h_dim3 = h_dim3
            best_lr_e = lr_e
            best_lr_m = lr_m
            best_lr_c = lr_c
            best_lr_cl = lr_cl
            best_dropout_rate_e = dropout_rate_e
            best_dropout_rate_m = dropout_rate_m
            best_dropout_rate_c = dropout_rate_c
            best_dropout_rate_clf = dropout_rate_clf
            best_weight_decay = weight_decay
            best_gamma = gamma
            best_epochs = epochs
            best_margin = margin
            print(f'New best validation AUROC: {best_auc}')
            np.save(all_aucs, result_path / 'all_aucs')

    print(f'Best validation AUROC: {best_auc}')
    print(f'{best_mini_batch=}, {best_h_dim1=}, {best_h_dim2=}, {best_h_dim3=}, {best_lr_e=}, {best_lr_m=}, '
          f'{best_lr_c=}, {best_lr_cl=}, {best_dropout_rate_e=}, {best_dropout_rate_m=}, {best_dropout_rate_c=}, '
          f'{best_dropout_rate_clf=}, {best_weight_decay=}, {best_gamma=}, {best_epochs=}, {best_margin=}')

    all_aucs = np.array([all_aucs])
    save_auroc_plots(all_aucs, result_path)
    np.save(all_aucs, result_path / 'all_aucs')

    # Test
    if run_test:
        x_train_e = GDSCE
        x_train_m = GDSCM
        x_train_c = GDSCC
        y_train = GDSCR

        x_test_eerlo = PDXEerlo
        x_test_merlo = torch.FloatTensor(PDXMerlo)
        x_test_cerlo = torch.FloatTensor(PDXCerlo)
        ytserlo = PDXRerlo

        x_test_ecet = PDXEcet
        x_test_mcet = torch.FloatTensor(PDXMcet)
        x_test_ccet = torch.FloatTensor(PDXCcet)
        ytscet = PDXRcet

        train_scaler_gdsc = sklearn.preprocessing.StandardScaler()
        x_train_e = train_scaler_gdsc.fit_transform(x_train_e)

        x_test_ecet = train_scaler_gdsc.transform(x_test_ecet)
        x_test_ecet = torch.FloatTensor(x_test_ecet)
        x_test_eerlo = train_scaler_gdsc.transform(x_test_eerlo)
        x_test_eerlo = torch.FloatTensor(x_test_eerlo)

        ytscet = torch.FloatTensor(ytscet.astype(int))
        ytserlo = torch.FloatTensor(ytserlo.astype(int))

        _, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_m.shape

        all_triplet_selector = AllTripletSelector()

        moli_model = Moli([ie_dim, im_dim, ic_dim],
                          [best_h_dim1, best_h_dim2, best_h_dim3],
                          [best_dropout_rate_e, best_dropout_rate_m, best_dropout_rate_c, best_dropout_rate_clf])
        moli_model = moli_model.to(device)

        moli_optimiser = torch.optim.Adagrad([
            {'params': moli_model.expression_encoder.parameters(), 'lr': best_lr_e},
            {'params': moli_model.mutation_encoder.parameters(), 'lr': best_lr_m},
            {'params': moli_model.cna_encoder.parameters(), 'lr': best_lr_c},
            {'params': moli_model.classifier.parameters(), 'lr': best_lr_cl, 'weight_decay': best_weight_decay},
        ])

        trip_criterion = torch.nn.TripletMarginLoss(margin=best_margin, p=2)
        bce_loss = torch.nn.BCEWithLogitsLoss()

        scaler_gdsc = sklearn.preprocessing.StandardScaler()
        scaler_gdsc.fit(x_train_e)
        x_train_e = scaler_gdsc.transform(x_train_e)
        x_train_m = np.nan_to_num(x_train_m)
        x_train_c = np.nan_to_num(x_train_c)
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)
        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                                       torch.FloatTensor(x_train_c),
                                                       torch.FloatTensor(y_train))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=best_mini_batch,
                                                   shuffle=False,
                                                   num_workers=8, sampler=sampler, pin_memory=True, drop_last=True)

        test_dataset_erlo = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_eerlo),
                                                           torch.FloatTensor(x_test_merlo),
                                                           torch.FloatTensor(x_test_cerlo), torch.FloatTensor(ytserlo))
        test_loader_erlo = torch.utils.data.DataLoader(dataset=test_dataset_erlo, batch_size=best_mini_batch,
                                                       shuffle=False, num_workers=8, pin_memory=True)

        test_dataset_cet = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_ecet),
                                                          torch.FloatTensor(x_test_mcet),
                                                          torch.FloatTensor(x_test_ccet), torch.FloatTensor(ytscet))
        test_loader_cet = torch.utils.data.DataLoader(dataset=test_dataset_cet, batch_size=best_mini_batch,
                                                      shuffle=False, num_workers=8, pin_memory=True)
        auc_train = 0
        for _ in range(best_epochs):
            auc_train, cost_train = network_training_util.train(train_loader, moli_model, moli_optimiser,
                                                                all_triplet_selector, trip_criterion,
                                                                bce_loss, device, best_gamma)

        auc_test_erlo = network_training_util.validate(test_loader_erlo, moli_model, device)
        auc_test_cet = network_training_util.validate(test_loader_cet, moli_model, device)

        print(f'EGFR: AUROC Train = {auc_train}')
        print(f'EGFR Cetuximab: AUROC = {auc_test_cet}')
        print(f'EGFR Erlotinib: AUROC = {auc_test_erlo}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_test', default=False, action='store_true')
    parser.add_argument('--load_checkpoint', default=False, action='store_true')
    parser.add_argument('--random_search_iteration', default=1, type=int)
    parser.add_argument('--experiment_name', required=True)
    args = parser.parse_args()
    cv_and_train(args.run_test, args.random_search_iteration, args.load_checkpoint, args.experiment_name)
