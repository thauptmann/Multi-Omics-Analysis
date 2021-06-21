from pathlib import Path

import numpy as np
import sklearn.preprocessing as sk
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange

from models.moli_model import Moli
from siamese_triplet.utils import AllTripletSelector
from utils import network_training_util, egfr_data
from utils.network_training_util import create_dataloader


def main():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        pin_memory = True
    else:
        device = torch.device("cpu")
        pin_memory = False

    data_path = Path('..', '..', '..', 'data')
    GDSCE, GDSCM, GDSCC, GDSCR, PDXEerlo, PDXMerlo, PDXCerlo, PDXRerlo, PDXEcet, PDXMcet, PDXCcet, PDXRcet = \
        egfr_data.load_egfr_data(data_path)

    mini_batch = 16
    h_dim1 = 32
    h_dim2 = 16
    h_dim3 = 256
    lr_e = 0.001
    lr_m = 0.0001
    lr_c = 5.00E-05
    lr_cl = 0.005
    dropout_rate_e = 0.5
    dropout_rate_m = 0.8
    dropout_rate_c = 0.5
    weight_decay = 0.0001
    dropout_rate_clf = 0.3
    gamma = 0.5
    epochs = 20
    margin = 1.5

    y_train = GDSCR
    x_train_e = GDSCE
    x_train_m = GDSCM
    x_train_c = GDSCC

    # Train
    x_test_e_erlo = PDXEerlo
    x_test_m_erlo = torch.FloatTensor(PDXMerlo)
    x_test_c_erlo = torch.FloatTensor(PDXCerlo)
    y_test_erlo = PDXRerlo

    x_test_e_cet = PDXEcet
    x_test_m_cet = torch.FloatTensor(PDXMcet)
    x_test_c_cet = torch.FloatTensor(PDXCcet)
    y_test_cet = PDXRcet

    y_test_cet = torch.FloatTensor(y_test_cet.astype(int))
    y_test_erlo = torch.FloatTensor(y_test_erlo.astype(int))

    scaler_gdsc = sk.StandardScaler()
    scaler_gdsc.fit(x_train_e)
    x_train_e = torch.FloatTensor(scaler_gdsc.transform(x_train_e))
    x_test_e_cet = torch.FloatTensor(scaler_gdsc.transform(x_test_e_cet))
    x_test_e_erlo = torch.FloatTensor(scaler_gdsc.transform(x_test_e_erlo))

    x_train_m = torch.FloatTensor(np.nan_to_num(x_train_m))
    x_train_c = torch.FloatTensor(np.nan_to_num(x_train_c))

    n_sample_e, ie_dim = x_train_e.shape
    _, im_dim = x_train_m.shape
    _, ic_dim = x_train_m.shape

    triplet_selector2 = AllTripletSelector()
    moli = Moli([ie_dim, im_dim, ic_dim], [h_dim1, h_dim2, h_dim3],
                           [dropout_rate_e, dropout_rate_m, dropout_rate_c,
                            dropout_rate_clf]).to(device)

    moli_optimiser = torch.optim.Adagrad([
        {'params': moli.expression_encoder.parameters(), 'lr': lr_e},
        {'params': moli.mutation_encoder.parameters(), 'lr': lr_m},
        {'params': moli.cna_encoder.parameters(), 'lr': lr_c},
        {'params': moli.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
    ])

    trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)

    train_loader = create_dataloader(x_train_e, x_train_m, x_train_c, y_train, mini_batch, pin_memory, sampler, True)
    test_loader_erlo = create_dataloader(x_test_e_erlo, x_test_m_erlo, x_test_c_erlo, y_test_erlo, mini_batch,
                                         pin_memory)
    test_loader_cet = create_dataloader(x_test_e_cet, x_test_m_cet, x_test_c_cet, y_test_cet, mini_batch, pin_memory)

    auc_train = 0
    for _ in trange(epochs):
        auc_train, _ = network_training_util.train(train_loader, moli, moli_optimiser, triplet_selector2,
                                                   trip_criterion, bce_loss, device, gamma)

    auc_test_erlo = network_training_util.validate(test_loader_erlo, moli, device)
    auc_test_cet = network_training_util.validate(test_loader_cet, moli, device)

    print(f'EGFR: AUROC Train = {auc_train}')
    print(f'EGFR Cetuximab: AUROC Test = {auc_test_cet}')
    print(f'EGFR Erlotinib: AUROC Test = {auc_test_erlo}')


if __name__ == "__main__":
    main()
