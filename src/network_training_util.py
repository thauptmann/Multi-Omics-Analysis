from contextlib import nullcontext
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast


def create_dataloader(x_expression, x_mutation, x_cna, y_response, mini_batch, pin_memory, sampler=None,
                      drop_last=False):
    dataset = torch.utils.data.TensorDataset(torch.Tensor(x_expression), torch.Tensor(x_mutation),
                                             torch.Tensor(x_cna), torch.Tensor(y_response))
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=mini_batch, shuffle=False,
                                       num_workers=8, sampler=sampler, pin_memory=pin_memory, drop_last=drop_last)


def train(train_loader, moli_model, moli_optimiser, triplet_selector, trip_criterion, cross_entropy, device, gamma):
    auc = 0
    loss = None
    if device == "cuda:0":
        scaler = GradScaler()  # Creates a GradScaler once at the beginning of training.
    else:
        scaler = None
    for (data_e, data_m, data_c, target) in train_loader:
        moli_model.train()

        data_e = data_e.to(device)
        data_m = data_m.to(device)
        data_c = data_c.to(device)
        target = target.to(device)

        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            with use_autocast(device):
                y_prediction, zt = moli_model.forward(data_e, data_m, data_c)
                triplets = triplet_selector.get_triplets(zt, target)
                target = target.view(-1, 1)
                loss = gamma * trip_criterion(zt[triplets[:, 0], :], zt[triplets[:, 1], :],
                                          zt[triplets[:, 2], :]) + cross_entropy(y_prediction, target)

            auc = roc_auc_score(target.detach().cpu(), y_prediction.detach().cpu())

            moli_optimiser.zero_grad()

            if not (scaler is None):
                scaler.scale(loss).backward()
                scaler.step(moli_optimiser)
                scaler.update()  # Updates the scale for next iteration.
            else:
                loss.backward()
                moli_optimiser.step()

    return auc, loss.item()


def validate(data_loader, moli_model, device):
    y_true_test = []
    prediction_test = []
    moli_model.eval()
    for (data_e, data_m, data_c, target) in data_loader:
        validate_e = torch.FloatTensor(data_e).to(device)
        validate_m = torch.FloatTensor(data_m).to(device)
        validate_c = torch.FloatTensor(data_c).to(device)
        y_true_test.extend(target.view(-1, 1))
        with torch.no_grad():
            moli_model.eval()
            with use_autocast(device):
                prediction, _ = moli_model.forward(validate_e, validate_m, validate_c)
                prediction_test.extend(prediction.cpu().detach())

    auc_test = roc_auc_score(y_true_test, prediction_test)
    return auc_test


def use_autocast(device):
    return autocast() if device == 'cuda:0' else nullcontext()
