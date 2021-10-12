import numpy as np
import torch
import torch.utils.data
import torch.nn
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector

sigmoid = torch.nn.Sigmoid()


def create_dataloader(x_expression, x_mutation, x_cna, y_response, mini_batch, pin_memory, sampler=None,
                      drop_last=False):
    dataset = torch.utils.data.TensorDataset(torch.Tensor(x_expression), torch.Tensor(x_mutation),
                                             torch.Tensor(x_cna), torch.Tensor(y_response))
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=mini_batch, shuffle=False,
                                       num_workers=8, sampler=sampler, pin_memory=pin_memory, drop_last=drop_last)


def train(train_loader, moli_model, moli_optimiser, loss_fn, device, gamma):
    y_true = []
    use_amp = False if device == torch.device('cpu') else True

    predictions = []
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    moli_model.train()
    for (data_e, data_m, data_c, target) in train_loader:
        moli_optimiser.zero_grad()
        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            y_true.extend(target)
            data_e = data_e.to(device)
            data_m = data_m.to(device)
            data_c = data_c.to(device)
            target = target.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                prediction = moli_model.forward(data_e, data_m, data_c)
            if gamma > 0:
                loss = loss_fn(prediction, target)
                prediction = sigmoid(prediction[0])
            else:
                target = target.view(-1, 1)
                loss = loss_fn(prediction[0], target)
                prediction = sigmoid(prediction)
            predictions.extend(prediction.cpu().detach())
            scaler.scale(loss).backward()
            scaler.step(moli_optimiser)
            scaler.update()
    y_true = torch.FloatTensor(y_true)
    predictions = torch.FloatTensor(predictions)
    auroc = roc_auc_score(y_true, predictions)
    return auroc


def validate(data_loader, moli_model, device, return_predictions=False):
    y_true = []
    predictions = []
    moli_model.eval()
    with torch.no_grad():
        for (data_e, data_m, data_c, target) in data_loader:
            validate_e = data_e.to(device)
            validate_m = data_m.to(device)
            validate_c = data_c.to(device)
            y_true.extend(target.numpy())
            logits, _ = moli_model.forward(validate_e, validate_m, validate_c)
            probabilities = sigmoid(logits)
            predictions.extend(probabilities.cpu().detach().numpy())
    auc_validate = roc_auc_score(y_true, predictions)
    auprc_validate = average_precision_score(y_true, predictions)
    if return_predictions:
        return y_true, predictions
    else:
        return auc_validate, auprc_validate


class BceWithTripletsToss:
    def __init__(self, gamma, triplet_selector, trip_criterion):
        self.gamma = gamma
        self.trip_criterion = trip_criterion
        self.triplet_selector = triplet_selector
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()
        super(BceWithTripletsToss, self).__init__()

    def __call__(self, predictions, target):
        prediction, zt = predictions
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


def test(moli_model, scaler, x_test_e, x_test_m, x_test_c, test_y, device, pin_memory):
    train_batch_size = 512
    x_test_e = torch.FloatTensor(scaler.transform(x_test_e))
    x_test_m = torch.FloatTensor(x_test_m)
    x_test_c = torch.FloatTensor(x_test_c)
    test_y = torch.FloatTensor(test_y.astype(int))

    test_loader = create_data_loader(x_test_e, x_test_m, x_test_c, test_y, train_batch_size, False, pin_memory)
    auc_test, auprc = validate(test_loader, moli_model, device)
    return auc_test, auprc


def create_data_loader(x_test_e, x_test_m, x_test_c, test_y, train_batch_size, drop_last, pin_memory, sampler=None):
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e),
                                                  torch.FloatTensor(x_test_m),
                                                  torch.FloatTensor(x_test_c), torch.FloatTensor(test_y))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=train_batch_size, shuffle=False,
                                              num_workers=8, pin_memory=pin_memory, drop_last=drop_last,
                                              sampler=sampler)
    return test_loader


def test_ensemble(moli_model_list, scaler_list, x_test_e, x_test_m, x_test_c, test_y, device, pin_memory):
    train_batch_size = 512
    x_test_m = torch.FloatTensor(x_test_m)
    x_test_c = torch.FloatTensor(x_test_c)
    test_y = torch.FloatTensor(test_y.astype(int))
    prediction_lists = []
    y_true_list = None
    for model, scaler in zip(moli_model_list, scaler_list):
        x_test_e = torch.FloatTensor(scaler.transform(x_test_e))
        test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_test_e),
                                                      torch.FloatTensor(x_test_m),
                                                      torch.FloatTensor(x_test_c), torch.FloatTensor(test_y))
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=train_batch_size, shuffle=False,
                                                  num_workers=8, pin_memory=pin_memory)
        y_true_list, prediction_list = validate(test_loader, model, device, True)
        prediction_lists.append(prediction_list)
    return y_true_list, prediction_lists


def get_triplet_selector(margin, triplet_selector_type):
    if triplet_selector_type == 'all':
        triplet_selector = AllTripletSelector()
    elif triplet_selector_type == 'hardest':
        triplet_selector = HardestNegativeTripletSelector(margin)
    elif triplet_selector_type == 'random':
        triplet_selector = RandomNegativeTripletSelector(margin)
    elif triplet_selector_type == 'semi_hard':
        triplet_selector = SemihardNegativeTripletSelector(margin)
    else:
        triplet_selector = None
    return triplet_selector


def get_loss_fn(margin, gamma, triplet_selector):
    if triplet_selector is not None and gamma > 0:
        trip_criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)
        return BceWithTripletsToss(gamma, triplet_selector, trip_criterion)
    else:
        return torch.nn.BCEWithLogitsLoss()
