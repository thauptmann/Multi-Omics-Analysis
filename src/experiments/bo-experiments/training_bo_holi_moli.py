import numpy as np
import scipy.signal
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm
from scipy.stats import sem
from models.bo_holi_moli_model import AdaptiveMoli
from siamese_triplet.utils import AllTripletSelector
from utils import network_training_util
from utils.network_training_util import BceWithTripletsToss
from ax import Data, Metric
import pandas as pd


class MoliMetric(Metric):
    def __init__(self, name, x_e, x_m, x_c, y,  device, pin_memory):
        super(MoliMetric, self).__init__(name)
        self.x_e = x_e
        self.x_m = x_m
        self.x_c = x_c
        self.y = y
        self.device = device
        self.pin_memory = pin_memory

    def fetch_trial_data(self, trial, **kwargs):
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            parameterization = arm.parameters
            combination = parameterization['combination']
            mini_batch = parameterization['mini_batch']
            h_dim1 = parameterization['h_dim1']
            h_dim2 = parameterization['h_dim2'] if 'h_dim2' in parameterization else parameterization['h_dim1']
            h_dim3 = parameterization['h_dim3'] if 'h_dim3' in parameterization else parameterization['h_dim1']
            h_dim4 = parameterization['h_dim4'] if 'h_dim4' in parameterization else parameterization['h_dim1']
            h_dim5 = parameterization['h_dim5'] if 'h_dim5' in parameterization else parameterization['h_dim1']
            depth_1 = parameterization['depth_1']
            depth_2 = parameterization['depth_2'] if 'depth_2' in parameterization else parameterization['depth_1']
            depth_3 = parameterization['depth_3'] if 'depth_3' in parameterization else parameterization['depth_1']
            depth_4 = parameterization['depth_4'] if 'depth_4' in parameterization else parameterization['depth_1']
            depth_5 = parameterization['depth_5'] if 'depth_5' in parameterization else parameterization['depth_1']
            lr_e = parameterization['lr_e']
            lr_m = parameterization['lr_m'] if 'lr_m' in parameterization else parameterization['lr_e']
            lr_c = parameterization['lr_c'] if 'lr_c' in parameterization else parameterization['lr_e']
            lr_middle = parameterization['lr_middle'] if 'lr_middle' in parameterization else parameterization['lr_e']
            lr_cl = parameterization['lr_cl'] if 'lr_cl' in parameterization else parameterization['lr_e']
            dropout_rate_e = parameterization['dropout_rate_e']
            dropout_rate_m = parameterization['dropout_rate_m'] if 'dropout_rate_m' in parameterization \
                else parameterization['dropout_rate_e']
            dropout_rate_c = parameterization['dropout_rate_c'] if 'dropout_rate_c' in parameterization \
                else parameterization['dropout_rate_e']
            dropout_rate_clf = parameterization['dropout_rate_clf'] if 'dropout_rate_clf' in parameterization \
                else parameterization['dropout_rate_e']
            dropout_rate_middle = parameterization['dropout_rate_middle'] if 'dropout_rate_middle' in parameterization \
                else parameterization['dropout_rate_e']
            weight_decay = parameterization['weight_decay']
            gamma = parameterization['gamma']
            epochs = parameterization['epochs']
            margin = parameterization['margin']

            aucs_validate = []
            cv_splits = 5
            skf = StratifiedKFold(n_splits=cv_splits)
            for train_index, validate_index in tqdm(skf.split(self.x_e, self.y), total=skf.get_n_splits(),
                                                    desc="k-fold"):
                x_train_e = self.x_e[train_index]
                x_train_m = self.x_m[train_index]
                x_train_c = self.x_c[train_index]
                y_train = self.y[train_index]

                x_validate_e = self.x_e[validate_index]
                x_validate_m = self.x_m[validate_index]
                x_validate_c = self.x_c[validate_index]
                y_validate = self.y[validate_index]

                scaler_gdsc = StandardScaler()
                x_train_e = scaler_gdsc.fit_transform(x_train_e)
                x_validate_e = scaler_gdsc.transform(x_validate_e)

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
                                                           shuffle=False, num_workers=8, sampler=sampler,
                                                           pin_memory=self.pin_memory,
                                                           drop_last=True)

                validation_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_validate_e),
                                                                    torch.FloatTensor(x_validate_m),
                                                                    torch.FloatTensor(x_validate_c),
                                                                    torch.FloatTensor(y_validate))
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=mini_batch,
                                                                shuffle=False, num_workers=8,
                                                                pin_memory=self.pin_memory)

                n_sample_e, ie_dim = x_train_e.shape
                _, im_dim = x_train_m.shape
                _, ic_dim = x_train_c.shape

                all_triplet_selector = AllTripletSelector()

                depths = [depth_1, depth_2, depth_3, depth_4, depth_5]
                input_sizes = [ie_dim, im_dim, ic_dim]
                dropout_rates = [dropout_rate_e, dropout_rate_m, dropout_rate_c, dropout_rate_middle, dropout_rate_clf]
                output_sizes = [h_dim1, h_dim2, h_dim3, h_dim4, h_dim5]
                moli_model = AdaptiveMoli(input_sizes, output_sizes, dropout_rates, combination, depths).to(self.device)

                moli_optimiser = torch.optim.Adagrad([
                    {'params': moli_model.left_encoder.parameters(), 'lr': lr_middle},
                    {'params': moli_model.expression_encoder.parameters(), 'lr': lr_e},
                    {'params': moli_model.mutation_encoder.parameters(), 'lr': lr_m},
                    {'params': moli_model.cna_encoder.parameters(), 'lr': lr_c},
                    {'params': moli_model.classifier.parameters(), 'lr': lr_cl, 'weight_decay': weight_decay},
                ])

                trip_criterion = torch.nn.TripletMarginLoss( margin=margin, p=2)

                bce_with_triplet_loss = BceWithTripletsToss(parameterization['gamma'], all_triplet_selector,
                                                            trip_criterion)
                for _ in trange(epochs, desc='Epoch'):
                    network_training_util.train(train_loader, moli_model, moli_optimiser,
                                                bce_with_triplet_loss,
                                                self.device, gamma)

                # validate
                auc_validate = network_training_util.validate(validation_loader, moli_model, self.device)
                aucs_validate.append(auc_validate)

            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                "mean": np.mean(aucs_validate),
                "sem": scipy.stats.sem(aucs_validate)
            })

        return Data(df=pd.DataFrame.from_records(records))

    def is_available_while_running(self) -> bool:
        return False


def train_final(parameterization, x_train_e, x_train_m, x_train_c, y_train, device, pin_memory):
    combination = parameterization['combination']
    mini_batch = parameterization['mini_batch']
    h_dim1 = parameterization['h_dim1']
    h_dim2 = parameterization['h_dim2'] if 'h_dim2' in parameterization else parameterization['h_dim1']
    h_dim3 = parameterization['h_dim3'] if 'h_dim3' in parameterization else parameterization['h_dim1']
    h_dim4 = parameterization['h_dim4'] if 'h_dim4' in parameterization else parameterization['h_dim1']
    h_dim5 = parameterization['h_dim5'] if 'h_dim5' in parameterization else parameterization['h_dim1']
    depth_1 = parameterization['depth_1']
    depth_2 = parameterization['depth_2'] if 'depth_2' in parameterization else parameterization['depth_1']
    depth_3 = parameterization['depth_3'] if 'depth_3' in parameterization else parameterization['depth_1']
    depth_4 = parameterization['depth_4'] if 'depth_4' in parameterization else parameterization['depth_1']
    depth_5 = parameterization['depth_5'] if 'depth_5' in parameterization else parameterization['depth_1']
    lr_e = parameterization['lr_e']
    lr_m = parameterization['lr_m'] if 'lr_m' in parameterization else parameterization['lr_e']
    lr_c = parameterization['lr_c'] if 'lr_c' in parameterization else parameterization['lr_e']
    lr_middle = parameterization['lr_middle'] if 'lr_middle' in parameterization else parameterization['lr_e']
    lr_cl = parameterization['lr_cl'] if 'lr_cl' in parameterization else parameterization['lr_e']
    dropout_rate_e = parameterization['dropout_rate_e']
    dropout_rate_m = parameterization['dropout_rate_m'] if 'dropout_rate_m' in parameterization \
        else parameterization['dropout_rate_e']
    dropout_rate_c = parameterization['dropout_rate_c'] if 'dropout_rate_c' in parameterization \
        else parameterization['dropout_rate_e']
    dropout_rate_clf = parameterization['dropout_rate_clf'] if 'dropout_rate_clf' in parameterization \
        else parameterization['dropout_rate_e']
    dropout_rate_middle = parameterization['dropout_rate_middle'] if 'dropout_rate_middle' in parameterization \
        else parameterization['dropout_rate_e']
    weight_decay = parameterization['weight_decay']
    gamma = parameterization['gamma']
    epochs = parameterization['epochs']
    margin = parameterization['margin']

    train_scaler_gdsc = StandardScaler()
    x_train_e = train_scaler_gdsc.fit_transform(x_train_e)

    _, ie_dim = x_train_e.shape
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
                                               num_workers=8, sampler=sampler, pin_memory=pin_memory, drop_last=True)
    bce_with_triplet_loss = BceWithTripletsToss(parameterization['gamma'], all_triplet_selector, trip_criterion)
    for _ in range(epochs):
        network_training_util.train(train_loader, moli_model, moli_optimiser,
                                    bce_with_triplet_loss, device, gamma)
    return moli_model, train_scaler_gdsc
