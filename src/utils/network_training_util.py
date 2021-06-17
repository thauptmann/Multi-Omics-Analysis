import torch.utils.data
import torch.nn
from sklearn.metrics import roc_auc_score
import pandas as pd


def create_dataloader(x_expression, x_mutation, x_cna, y_response, mini_batch, pin_memory, sampler=None,
                      drop_last=False):
    dataset = torch.utils.data.TensorDataset(torch.Tensor(x_expression), torch.Tensor(x_mutation),
                                             torch.Tensor(x_cna), torch.Tensor(y_response))
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=mini_batch, shuffle=False,
                                       num_workers=8, sampler=sampler, pin_memory=pin_memory, drop_last=drop_last)


def train(train_loader, moli_model, moli_optimiser, bce_with_triplets_loss, device, gamma):
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

            prediction, zt = moli_model.forward(data_e, data_m, data_c)
            if gamma > 0:
                loss = bce_with_triplets_loss((prediction, zt), target)
            else:
                bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()
                target = target.view(-1, 1)
                loss = bce_with_logits_loss(prediction, target)
            sigmoid = torch.nn.Sigmoid()
            prediction = sigmoid(prediction)
            predictions.extend(prediction.cpu().detach())
            loss.backward()
            moli_optimiser.step()
    y_true = torch.FloatTensor(y_true)
    predictions = torch.FloatTensor(predictions)
    auc = roc_auc_score(y_true, predictions)
    return auc


def validate(data_loader, moli_model, device):
    y_true = []
    predictions = []
    moli_model.eval()
    with torch.no_grad():
        for (data_e, data_m, data_c, target) in data_loader:
            validate_e = data_e.to(device)
            validate_m = data_m.to(device)
            validate_c = data_c.to(device)
            y_true.extend(target)
            prediction, _ = moli_model.forward(validate_e, validate_m, validate_c)
            sigmoid = torch.nn.Sigmoid()
            prediction = sigmoid(prediction)
            predictions.extend(prediction.cpu().detach())

    auc_test = roc_auc_score(y_true, predictions)
    return auc_test


class BceWithTripletsToss:
    def __init__(self, gamma, triplet_selector, trip_criterion):
        self.gamma = gamma
        self.trip_criterion = trip_criterion
        self.triplet_selector = triplet_selector
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss()
        super(BceWithTripletsToss, self).__init__()

    def __call__(self, predictions, target, ):
        prediction, zt = predictions
        triplets = self.triplet_selector.get_triplets(zt, target)
        target = target.view(-1, 1)
        loss = self.gamma * self.trip_criterion(zt[triplets[:, 0], :], zt[triplets[:, 1], :],
                                                zt[triplets[:, 2], :]) + self.bce_with_logits(prediction, target)
        return loss


def read_and_transpose_csv(path):
    csv_data = pd.read_csv(path, sep="\t", index_col=0, decimal=',')
    return pd.DataFrame.transpose(csv_data)
