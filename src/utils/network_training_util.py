import numpy as np
import torch
import torch.utils.data
import torch.nn
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from torch.utils.data import WeightedRandomSampler
from tqdm import trange

from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector

sigmoid = torch.nn.Sigmoid()


def create_dataloader(x_expression, x_mutation, x_cna, y_response, mini_batch, pin_memory, sampler=None,
                      drop_last=False):
    dataset = torch.utils.data.TensorDataset(torch.Tensor(x_expression), torch.Tensor(x_mutation),
                                             torch.Tensor(x_cna), torch.Tensor(y_response))
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=mini_batch, shuffle=False,
                                       num_workers=8, sampler=sampler, pin_memory=pin_memory, drop_last=drop_last)


def train(train_loader, moli_model, moli_optimiser, loss_fn, device, gamma, last_epochs):
    y_true = []
    predictions = []
    moli_model.train()
    for (data_e, data_m, data_c, target) in train_loader:
        moli_optimiser.zero_grad()
        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            y_true.extend(target)
            data_e = data_e.to(device)
            data_m = data_m.to(device)
            data_c = data_c.to(device)
            target = target.to(device)
            prediction = moli_model.forward(data_e, data_m, data_c)
            if gamma > 0:
                loss = loss_fn(prediction, target, last_epochs)
                prediction = sigmoid(prediction[0])
            else:
                target = target.view(-1, 1)
                loss = loss_fn(prediction[0], target)
                prediction = sigmoid(prediction[0])
            predictions.extend(prediction.cpu().detach())
            loss.backward()
            moli_optimiser.step()
    y_true = torch.FloatTensor(y_true)
    predictions = torch.FloatTensor(predictions)
    auroc = roc_auc_score(y_true, predictions)
    return auroc


class BceWithTripletsToss:
    def __init__(self, gamma, triplet_selector, trip_criterion, semi_hard_triplet):
        self.gamma = gamma
        self.trip_criterion = trip_criterion
        self.triplet_selector = triplet_selector
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()
        self.semi_hard_triplet = semi_hard_triplet
        super(BceWithTripletsToss, self).__init__()

    def __call__(self, predictions, target, last_epoch):
        prediction, zt = predictions
        if not last_epoch and self.semi_hard_triplet:
            triplets = self.triplet_selector[0].get_triplets(zt, target)
        elif last_epoch and self.semi_hard_triplet:
            triplets = self.triplet_selector[1].get_triplets(zt, target)
        else:
            triplets = self.triplet_selector.get_triplets(zt, target)
        target = target.view(-1, 1)
        loss = self.gamma * self.trip_criterion(zt[triplets[:, 0], :], zt[triplets[:, 1], :],
                                                zt[triplets[:, 2], :]) + self.bce_with_logits(prediction, target)
        return loss


def read_and_transpose_csv(path):
    csv_data = pd.read_csv(path, sep="\t", index_col=0, decimal=',')
    return pd.DataFrame.transpose(csv_data)


def calculate_mean_and_std_auc(result_dict, result_file, drug_name):
    result_file.write(f'\tMean Result for {drug_name}:\n\n')
    for result_name, result_value in result_dict.items():
        mean = np.mean(result_value)
        std = np.std(result_value)
        max_value = np.max(result_value)
        min_value = np.min(result_value)
        result_file.write(f'\t\t{result_name} max: {max_value}\n')
        result_file.write(f'\t\t{result_name} min: {min_value}\n')
        result_file.write(f'\t\t{result_name} mean: {mean}\n')
        result_file.write(f'\t\t{result_name} std: {std}\n')
        result_file.write('\n')


def test(moli_model, scaler, x_test_e, x_test_m, x_test_c, test_y, device):
    x_test_e = torch.FloatTensor(scaler.transform(x_test_e)).to(device)
    x_test_m = torch.FloatTensor(x_test_m).to(device)
    x_test_c = torch.FloatTensor(x_test_c).to(device)
    test_y = torch.FloatTensor(test_y.astype(int))
    moli_model.eval()
    logits, _ = moli_model.forward(x_test_e, x_test_m, x_test_c)
    probabilities = sigmoid(logits)
    auc_validate = roc_auc_score(test_y, probabilities.cpu().detach().numpy())
    auprc_validate = average_precision_score(test_y, probabilities.cpu().detach().numpy())
    return auc_validate, auprc_validate


def create_data_loader(x_test_e, x_test_m, x_test_c, test_y, train_batch_size, pin_memory, sampler=None):
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e),
                                             torch.FloatTensor(x_test_m),
                                             torch.FloatTensor(x_test_c), torch.FloatTensor(test_y))
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=False,
                                         num_workers=8, pin_memory=pin_memory, drop_last=True,
                                         sampler=sampler)
    return loader


def test_ensemble(moli_model_list, scaler_list, x_test_e, x_test_m, x_test_c, test_y, device):
    x_test_m = torch.FloatTensor(x_test_m)
    x_test_c = torch.FloatTensor(x_test_c)
    test_y = torch.FloatTensor(test_y.astype(int))
    prediction_lists = []
    y_true_list = None
    for model, scaler in zip(moli_model_list, scaler_list):
        y_true_list, prediction_list = test(model, scaler, x_test_e, x_test_m, x_test_c, test_y, device)
        prediction_lists.append(prediction_list)
    return y_true_list, prediction_lists


def get_triplet_selector(margin, semi_hard_triplet):
    return [SemihardNegativeTripletSelector(margin), HardestNegativeTripletSelector(margin)] if semi_hard_triplet \
        else AllTripletSelector()


def get_loss_fn(margin, gamma, triplet_selector, semi_hard_triplet):
    if triplet_selector is not None and gamma > 0:
        trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
        return BceWithTripletsToss(gamma, triplet_selector, trip_criterion, semi_hard_triplet)
    else:
        return torch.nn.BCEWithLogitsLoss()


def feature_selection(gdsce, gdscm, gdscc):
    selector = VarianceThreshold(0.05 * 20)
    selector.fit_transform(gdsce)
    gdsce = gdsce[gdsce.columns[selector.get_support(indices=True)]]

    selector = VarianceThreshold(0.00001 * 15)
    selector.fit_transform(gdscm)
    gdscm = gdscm[gdscm.columns[selector.get_support(indices=True)]]

    selector = VarianceThreshold(0.01 * 20)
    selector.fit_transform(gdscc)
    gdscc = gdscc[gdscc.columns[selector.get_support(indices=True)]]

    return gdsce, gdscm, gdscc


def create_sampler(y_train):
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight),
                                    replacement=True)
    return sampler


def train_encoder(supervised_encoder_epoch, optimizer, triplet_selector, device, supervised_encoder, train_loader,
                  trip_loss_fun, semi_hard_triplet, omic_number):
    supervised_encoder.train()
    for epoch in trange(supervised_encoder_epoch):
        last_epochs = False if epoch < supervised_encoder_epoch - 2 else True
        for data in train_loader:
            target = data[-1]
            x = data[omic_number]
            optimizer.zero_grad()
            x = x.to(device)
            encoded_data = supervised_encoder(x)
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


def train_classifier(bce_loss_function, classifier_epoch, classifier_optimizer, device, final_c_supervised_encoder,
                     final_e_supervised_encoder, final_m_supervised_encoder, classifier, train_loader):
    classifier.train()
    for _ in trange(classifier_epoch):
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