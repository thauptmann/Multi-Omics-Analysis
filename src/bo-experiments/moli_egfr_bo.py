from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange

from models.auto_moli_model import Moli
from siamese_triplet.utils import AllTripletSelector
from utils import egfr_data
from utils import network_training_util


def train_evaluate(parameterization):
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    mini_batch = parameterization['mini_batch']
    h_dim1 = parameterization['h_dim1']
    h_dim2 = parameterization['h_dim2']
    h_dim3 = parameterization['h_dim3']
    lr_e = parameterization['lr_e']
    lr_m = parameterization['lr_m']
    lr_c = parameterization['lr_c']
    lr_cl = parameterization['lr_cl']
    dropout_rate_e = parameterization['dropout_rate_e']
    dropout_rate_m = parameterization['dropout_rate_m']
    dropout_rate_c = parameterization['dropout_rate_c']
    weight_decay = parameterization['weight_decay']
    gamma = parameterization['gamma']
    epochs = parameterization['epochs']
    margin = parameterization['margin']

    data_path = Path('../..', 'data')
    egfr_path = Path(data_path, 'EGFR_experiments_data')

    GDSCEv2, GDSCMv2, GDSCCv2, Y = egfr_data.load_train_data(egfr_path)

    skf = StratifiedKFold(n_splits=5)
    aucs_validate = []
    for train_index, test_index in skf.split(GDSCEv2.to_numpy(), Y):
        x_train_e = GDSCEv2.values[train_index]
        x_train_m = GDSCMv2.values[train_index]
        x_train_c = GDSCCv2.values[train_index]

        x_test_e = GDSCEv2.values[test_index]
        x_test_m = GDSCMv2.values[test_index]
        x_test_c = GDSCCv2.values[test_index]

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

        moli_model = Moli([ie_dim, im_dim, ic_dim],
                          [h_dim1, h_dim2, h_dim3],
                          [dropout_rate_e, dropout_rate_m, dropout_rate_c]).to(device)

        moli_optimiser = torch.optim.Adagrad([
            {'params': moli_model.expression_encoder.parameters(), 'lr': lr_e},
            {'params': moli_model.mutation_encoder.parameters(), 'lr': lr_m},
            {'params': moli_model.cna_encoder.parameters(), 'lr': lr_c},
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
