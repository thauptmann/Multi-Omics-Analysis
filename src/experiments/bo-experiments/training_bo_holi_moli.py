import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm

from models.bo_holi_moli_model import AdaptiveMoli
from siamese_triplet.utils import AllTripletSelector
from utils import network_training_util
from utils.network_training_util import BceWithTripletsToss

torch.manual_seed(42)
np.random.seed(42)


def train_evaluate(parameterization, GDSCE, GDSCM, GDSCC, Y, best_auc, device):
    combination = parameterization['combination']
    mini_batch = parameterization['mini_batch']
    h_dim1 = parameterization['h_dim1']
    h_dim2 = parameterization['h_dim2']
    h_dim3 = parameterization['h_dim3']
    h_dim4 = parameterization['h_dim4']
    h_dim5 = parameterization['h_dim5']
    depth_1 = parameterization['depth_1']
    depth_2 = parameterization['depth_2']
    depth_3 = parameterization['depth_3']
    depth_4 = parameterization['depth_4']
    depth_5 = parameterization['depth_5']
    lr_e = parameterization['lr_e']
    lr_m = parameterization['lr_m']
    lr_c = parameterization['lr_c']
    lr_middle = parameterization['lr_middle']
    lr_cl = parameterization['lr_cl']
    dropout_rate_e = parameterization['dropout_rate_e']
    dropout_rate_m = parameterization['dropout_rate_m']
    dropout_rate_c = parameterization['dropout_rate_c']
    dropout_rate_clf = parameterization['dropout_rate_clf']
    dropout_rate_middle = parameterization['dropout_rate_middle']
    weight_decay = parameterization['weight_decay']
    gamma = parameterization['gamma']
    epochs = parameterization['epochs']
    margin = parameterization['margin']

    aucs_validate = []
    cv_splits = 5
    skf = StratifiedKFold(n_splits=cv_splits)
    fold_number = 1
    for train_index, test_index in tqdm(skf.split(GDSCE, Y), total=skf.get_n_splits(), desc="k-fold"):
        x_train_e = GDSCE[train_index]
        x_train_m = GDSCM[train_index]
        x_train_c = GDSCC[train_index]

        x_test_e = GDSCE[test_index]
        x_test_m = GDSCM[test_index]
        x_test_c = GDSCC[test_index]

        y_train = Y[train_index]
        y_test = Y[test_index]

        scaler_gdsc = StandardScaler()
        x_train_e = scaler_gdsc.fit_transform(x_train_e)
        x_test_e = scaler_gdsc.transform(x_test_e)
        x_train_m = np.nan_to_num(x_train_m)
        x_train_c = np.nan_to_num(x_train_c)

        # Initialisation
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e),
                                                       torch.FloatTensor(x_train_m),
                                                       torch.FloatTensor(x_train_c),
                                                       torch.FloatTensor(y_train))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=mini_batch,
                                                   shuffle=False, num_workers=8, sampler=sampler, pin_memory=True,
                                                   drop_last=True)

        test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e), torch.FloatTensor(x_test_m),
                                                      torch.FloatTensor(x_test_c),
                                                      torch.FloatTensor(y_test))
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=mini_batch,
                                                  shuffle=False, num_workers=8, pin_memory=True)

        n_sample_e, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_c.shape

        all_triplet_selector = AllTripletSelector()

        depths = [depth_1, depth_2, depth_3, depth_4, depth_5]
        input_sizes = [ie_dim, im_dim, ic_dim]
        dropout_rates = [dropout_rate_e, dropout_rate_m, dropout_rate_c, dropout_rate_middle, dropout_rate_clf]
        output_sizes = [h_dim1, h_dim2, h_dim3, h_dim4, h_dim5]
        moli_model = AdaptiveMoli(input_sizes, output_sizes, dropout_rates, combination, depths).to(device)

        moli_optimiser = torch.optim.Adagrad([
            {'params': moli_model.left_encoder.parameters(), 'lr': lr_middle},
            {'params': moli_model.expression_encoder.parameters(), 'lr': lr_e},
            {'params': moli_model.mutation_encoder.parameters(), 'lr': lr_m},
            {'params': moli_model.cna_encoder.parameters(), 'lr': lr_c},
            {'params': moli_model.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
        ])

        trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

        bce_with_triplet_loss = BceWithTripletsToss(parameterization['gamma'], all_triplet_selector, trip_criterion)
        for _ in trange(epochs, desc='Epoch'):
            network_training_util.train(train_loader, moli_model, moli_optimiser,
                                        bce_with_triplet_loss,
                                        device, gamma)

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

    mean = np.mean(aucs_validate)
    sem = np.std(aucs_validate)
    return (mean, sem)


def train_final(parameterization, x_train_e, x_train_m, x_train_c, y_train, device):
    combination = parameterization['combination']
    mini_batch = parameterization['mini_batch']
    h_dim1 = parameterization['h_dim1']
    h_dim2 = parameterization['h_dim2']
    h_dim3 = parameterization['h_dim3']
    h_dim4 = parameterization['h_dim4']
    h_dim5 = parameterization['h_dim5']
    depth_1 = parameterization['depth_1']
    depth_2 = parameterization['depth_2']
    depth_3 = parameterization['depth_3']
    depth_4 = parameterization['depth_4']
    depth_5 = parameterization['depth_5']
    lr_e = parameterization['lr_e']
    lr_m = parameterization['lr_m']
    lr_c = parameterization['lr_c']
    lr_cl = parameterization['lr_cl']
    lr_middle = parameterization['lr_middle']
    dropout_rate_e = parameterization['dropout_rate_e']
    dropout_rate_m = parameterization['dropout_rate_m']
    dropout_rate_c = parameterization['dropout_rate_c']
    dropout_rate_clf = parameterization['dropout_rate_clf']
    dropout_rate_middle = parameterization['dropout_rate_middle']
    weight_decay = parameterization['weight_decay']
    gamma = parameterization['gamma']
    epochs = parameterization['epochs']
    margin = parameterization['margin']

    train_scaler_gdsc = StandardScaler()
    x_train_e = train_scaler_gdsc.fit_transform(x_train_e)

    _, ie_dim = x_train_e.shape
    _, im_dim = x_train_m.shape
    _, ic_dim = x_train_m.shape

    all_triplet_selector = AllTripletSelector()

    depths = [depth_1, depth_2, depth_3, depth_4, depth_5]
    input_sizes = [ie_dim, im_dim, ic_dim]
    dropout_rates = [dropout_rate_e, dropout_rate_m, dropout_rate_c, dropout_rate_clf, dropout_rate_middle]
    output_sizes = [h_dim1, h_dim2, h_dim3, h_dim4, h_dim5]
    moli_model = AdaptiveMoli(input_sizes, output_sizes, dropout_rates, combination, depths).to(device)

    moli_optimiser = torch.optim.Adagrad([
        {'params': moli_model.left_encoder.parameters(), 'lr': lr_middle},
        {'params': moli_model.expression_encoder.parameters(), 'lr': lr_e},
        {'params': moli_model.mutation_encoder.parameters(), 'lr': lr_m},
        {'params': moli_model.cna_encoder.parameters(), 'lr': lr_c},
        {'params': moli_model.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
    ])

    trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
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
                                               num_workers=8, sampler=sampler, pin_memory=True, drop_last=True)
    bce_with_triplet_loss = BceWithTripletsToss(parameterization['gamma'], all_triplet_selector, trip_criterion)
    for _ in range(epochs):
        network_training_util.train(train_loader, moli_model, moli_optimiser,
                                    bce_with_triplet_loss, device, gamma)
    return moli_model, train_scaler_gdsc


def test(moli_model, scaler, x_test_e, x_test_m, x_test_c, test_y, device):
    x_test_e = torch.FloatTensor(scaler.transform(x_test_e))
    test_y = torch.FloatTensor(test_y.astype(int))
    train_batch_size = 256
    x_test_m = torch.FloatTensor(x_test_m)
    x_test_c = torch.FloatTensor(x_test_c)

    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e),
                                                  torch.FloatTensor(x_test_m),
                                                  torch.FloatTensor(x_test_c), torch.FloatTensor(test_y))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=train_batch_size, shuffle=False,
                                              num_workers=8, pin_memory=True)
    auc_test = network_training_util.validate(test_loader, moli_model, device)

    return auc_test
