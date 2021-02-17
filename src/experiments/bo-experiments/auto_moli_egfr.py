import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm

from models.bo_moli_model import AdaptiveMoli
from siamese_triplet.utils import AllTripletSelector
from utils import network_training_util
from utils.choose_gpu import get_free_gpu


def train_evaluate(parameterization, GDSCE, GDSCM, GDSCC, Y):
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        free_gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{free_gpu_id}")
    else:
        device = torch.device("cpu")

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
    dropout_rate_e = parameterization['dropout_rate_e']
    dropout_rate_m = parameterization['dropout_rate_m']
    dropout_rate_c = parameterization['dropout_rate_c']
    dropout_rate_clf = parameterization['dropout_rate_clf']
    dropout_rate_middle = parameterization['dropout_rate_middle']
    weight_decay = parameterization['weight_decay']
    gamma = parameterization['gamma']
    epochs = parameterization['epochs']
    margin = parameterization['margin']

    skf = StratifiedKFold(n_splits=5)
    aucs_validate = []
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
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                        replacement=True)

        train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e),
                                                       torch.FloatTensor(x_train_m),
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

        depths = [depth_1, depth_2, depth_3, depth_4, depth_5]
        input_sizes = [ie_dim, im_dim, ic_dim]
        dropout_rates = [dropout_rate_e, dropout_rate_m, dropout_rate_c, dropout_rate_clf, dropout_rate_middle]
        output_sizes = [h_dim1, h_dim2, h_dim3, h_dim4, h_dim5]
        moli_model = AdaptiveMoli(input_sizes, output_sizes, dropout_rates, combination, depths).to(device)

        moli_optimiser = torch.optim.Adagrad([
            {'params': moli_model.left.parameters(), 'lr': lr_e},
            {'params': moli_model.middle.parameters(), 'lr': lr_m},
            {'params': moli_model.right.parameters(), 'lr': lr_c},
            {'params': moli_model.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
        ])

        trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

        bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
        auc_validate = 0
        for _ in trange(epochs, desc='Epoch'):
            _, _ = network_training_util.train(train_loader, moli_model, moli_optimiser,
                                               all_triplet_selector, trip_criterion, bce_with_logits_loss,
                                               device, gamma)

            # validate
            auc_validate = network_training_util.validate(test_loader, moli_model, device)
        aucs_validate.append(auc_validate)
    return np.mean(aucs_validate)


def train_and_test(parameterization, GDSCE, GDSCM, GDSCC, GDSCR, PDXEerlo, PDXMerlo, PDXCerlo, PDXRerlo,
                                    PDXEcet, PDXMcet, PDXCcet, PDXRcet):
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

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    x_train_e = GDSCE.values
    x_train_m = GDSCM.values
    x_train_c = GDSCC.values
    y_train = GDSCR.values

    x_test_eerlo = PDXEerlo.values
    x_test_merlo = torch.FloatTensor(PDXMerlo.values)
    x_test_cerlo = torch.FloatTensor(PDXCerlo.values)
    ytserlo = PDXRerlo['response'].values

    x_test_ecet = PDXEcet.values
    x_test_mcet = torch.FloatTensor(PDXMcet.values)
    x_test_ccet = torch.FloatTensor(PDXCcet.values)
    ytscet = PDXRcet['response'].values

    train_scaler_gdsc = StandardScaler()
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

    depths = [depth_1, depth_2, depth_3, depth_4, depth_5]
    input_sizes = [ie_dim, im_dim, ic_dim]
    dropout_rates = [dropout_rate_e, dropout_rate_m, dropout_rate_c, dropout_rate_clf, dropout_rate_middle]
    output_sizes = [h_dim1, h_dim2, h_dim3, h_dim4, h_dim5]
    moli_model = AdaptiveMoli(input_sizes, output_sizes, dropout_rates, combination, depths).to(device)

    moli_optimiser = torch.optim.Adagrad([
        {'params': moli_model.left.parameters(), 'lr': lr_e},
        {'params': moli_model.middle.parameters(), 'lr': lr_m},
        {'params': moli_model.right.parameters(), 'lr': lr_c},
        {'params': moli_model.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
    ])

    trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

    bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()

    scaler_gdsc = StandardScaler()
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
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=mini_batch,
                                               shuffle=False,
                                               num_workers=8, sampler=sampler, pin_memory=True, drop_last=True)

    test_dataset_erlo = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_eerlo),
                                                       torch.FloatTensor(x_test_merlo),
                                                       torch.FloatTensor(x_test_cerlo), torch.FloatTensor(ytserlo))
    test_loader_erlo = torch.utils.data.DataLoader(dataset=test_dataset_erlo, batch_size=mini_batch,
                                                   shuffle=False, num_workers=8, pin_memory=True)

    test_dataset_cet = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_ecet),
                                                      torch.FloatTensor(x_test_mcet),
                                                      torch.FloatTensor(x_test_ccet), torch.FloatTensor(ytscet))
    test_loader_cet = torch.utils.data.DataLoader(dataset=test_dataset_cet, batch_size=mini_batch, shuffle=False,
                                                  num_workers=8, pin_memory=True)
    auc_train = 0
    for _ in range(epochs):
        auc_train, cost_train = network_training_util.train(train_loader, moli_model, moli_optimiser,
                                                            all_triplet_selector, trip_criterion,
                                                            bce_with_logits_loss, device, gamma)

    auc_test_erlo = network_training_util.validate(test_loader_erlo, moli_model, device)
    auc_test_cet = network_training_util.validate(test_loader_cet, moli_model, device)

    print(f'EGFR: AUROC Train = {auc_train}')
    print(f'EGFR Cetuximab: AUROC = {auc_test_cet}')
    print(f'EGFR Erlotinib: AUROC = {auc_test_erlo}')
