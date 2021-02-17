from contextlib import nullcontext
import torch
import torch.utils.data
import torch.nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def create_dataloader(x_expression, x_mutation, x_cna, y_response, mini_batch, pin_memory, sampler=None,
                      drop_last=False):
    dataset = torch.utils.data.TensorDataset(torch.Tensor(x_expression), torch.Tensor(x_mutation),
                                             torch.Tensor(x_cna), torch.Tensor(y_response))
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=mini_batch, shuffle=False,
                                       num_workers=8, sampler=sampler, pin_memory=pin_memory, drop_last=drop_last)


def train(train_loader, moli_model, moli_optimiser, triplet_selector, trip_criterion, bce_with_logits, device, gamma):
    running_loss = 0
    y_true = []
    predictions = []
    if device == "cuda:0":
        scaler = GradScaler()  # Creates a GradScaler once at the beginning of training.
    else:
        scaler = None
    for (data_e, data_m, data_c, target) in train_loader:
        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            moli_model.train()
            y_true.extend(target)

            data_e = data_e.to(device)
            data_m = data_m.to(device)
            data_c = data_c.to(device)
            target = target.to(device)

            with use_autocast(device):
                prediction, zt = moli_model.forward(data_e, data_m, data_c)
                triplets = triplet_selector.get_triplets(zt, target)
                target = target.view(-1, 1)
                loss = gamma * trip_criterion(zt[triplets[:, 0], :], zt[triplets[:, 1], :],
                                              zt[triplets[:, 2], :]) + bce_with_logits(prediction, target)
                sigmoid = torch.nn.Sigmoid()
                prediction = sigmoid(prediction)
                predictions.extend(prediction.cpu().detach())

            moli_optimiser.zero_grad()
            running_loss = loss.item()
            if not (scaler is None):
                scaler.scale(loss).backward()
                scaler.step(moli_optimiser)
                scaler.update()  # Updates the scale for next iteration.
            else:
                loss.backward()
                moli_optimiser.step()
    auc = roc_auc_score(y_true, predictions)
    return auc, running_loss


def validate(data_loader, moli_model, device):
    y_true = []
    predictions = []
    moli_model.eval()
    for (data_e, data_m, data_c, target) in data_loader:
        validate_e = data_e.to(device)
        validate_m = data_m.to(device)
        validate_c = data_c.to(device)
        y_true.extend(target)
        with torch.no_grad():
            moli_model.eval()
            with use_autocast(device):
                prediction, _ = moli_model.forward(validate_e, validate_m, validate_c)
                sigmoid = torch.nn.Sigmoid()
                prediction = sigmoid(prediction)
                predictions.extend(prediction.cpu().detach())

    auc_test = roc_auc_score(y_true, predictions)
    return auc_test


def use_autocast(device):
    return autocast() if device == 'cuda:0' else nullcontext()


def read_and_transpose_csv(path):
    csv_data = pd.read_csv(path, sep="\t", index_col=0, decimal=',')
    return pd.DataFrame.transpose(csv_data)
