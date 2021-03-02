import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm

from models.bo_moli_model import AdaptiveMoli
from siamese_triplet.utils import AllTripletSelector
from utils import network_training_util


def train_evaluate(parameterization, GDSCE, GDSCM, GDSCC, Y, best_auc, device):
    # reproducibility

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

        bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
        for _ in trange(epochs, desc='Epoch'):
            _ = network_training_util.train(train_loader, moli_model, moli_optimiser,
                                            all_triplet_selector, trip_criterion, bce_with_logits_loss,
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


def train_and_test(parameterization, GDSCE, GDSCM, GDSCC, GDSCR, PDXEerlo, PDXMerlo, PDXCerlo, PDXRerlo,
                   PDXEcet, PDXMcet, PDXCcet, PDXRcet, device):
    train_batch_size = 256
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
    torch.manual_seed(42)
    np.random.seed(42)

    x_train_e = GDSCE
    x_train_m = GDSCM
    x_train_c = GDSCC
    y_train = GDSCR

    x_test_e_erlo = PDXEerlo
    x_test_m_erlo = torch.FloatTensor(PDXMerlo)
    x_test_c_erlo = torch.FloatTensor(PDXCerlo)
    y_test_erlo = PDXRerlo

    x_test_e_cet = PDXEcet
    x_test_m_cet = torch.FloatTensor(PDXMcet)
    x_test_c_cet = torch.FloatTensor(PDXCcet)
    y_test_cet = PDXRcet

    train_scaler_gdsc = StandardScaler()
    x_train_e = train_scaler_gdsc.fit_transform(x_train_e)
    x_test_e_cet = torch.FloatTensor(train_scaler_gdsc.transform(x_test_e_cet))
    x_test_e_erlo = torch.FloatTensor(train_scaler_gdsc.transform(x_test_e_erlo))

    y_test_cet = torch.FloatTensor(y_test_cet.astype(int))
    y_test_erlo = torch.FloatTensor(y_test_erlo.astype(int))

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
    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

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

    test_dataset_erlo = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e_erlo),
                                                       torch.FloatTensor(x_test_m_erlo),
                                                       torch.FloatTensor(x_test_c_erlo), torch.FloatTensor(y_test_erlo))
    test_loader_erlo = torch.utils.data.DataLoader(dataset=test_dataset_erlo, batch_size=train_batch_size,
                                                   shuffle=False, num_workers=8, pin_memory=True)

    test_dataset_cet = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e_cet),
                                                      torch.FloatTensor(x_test_m_cet),
                                                      torch.FloatTensor(x_test_c_cet), torch.FloatTensor(y_test_cet))
    test_loader_cet = torch.utils.data.DataLoader(dataset=test_dataset_cet, batch_size=train_batch_size, shuffle=False,
                                                  num_workers=8, pin_memory=True)

    test_dataset_both = torch.utils.data.ConcatDataset((test_dataset_cet, test_dataset_erlo))
    test_loader_both = torch.utils.data.DataLoader(dataset=test_dataset_both, batch_size=train_batch_size,
                                                   shuffle=False, num_workers=8, pin_memory=True)

    auc_train = 0
    for _ in range(epochs):
        auc_train = network_training_util.train(train_loader, moli_model, moli_optimiser,
                                                all_triplet_selector, trip_criterion,
                                                bce_with_logits_loss, device, gamma)

    auc_test_erlo = network_training_util.validate(test_loader_erlo, moli_model, device)
    auc_test_cet = network_training_util.validate(test_loader_cet, moli_model, device)
    auc_test_both = network_training_util.validate(test_loader_both, moli_model, device)
    return auc_train, auc_test_erlo, auc_test_cet, auc_test_both
