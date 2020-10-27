import torch
from sklearn.metrics import roc_auc_score


def create_dataloader(x_expression, x_mutation, x_cna, y_response, mini_batch, pin_memory, sampler=None,
                      drop_last=False):
    dataset = torch.utils.data.TensorDataset(torch.Tensor(x_expression), torch.Tensor(x_mutation),
                                             torch.Tensor(x_cna), torch.Tensor(y_response))
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=mini_batch, shuffle=False,
                                       num_workers=8, sampler=sampler, pin_memory=pin_memory, drop_last=drop_last)


def train(train_loader, moli_model, moli_optimiser, triplet_selector, trip_criterion, cross_entropy, device, gamma):
    for (data_e, data_m, data_c, target) in train_loader:
        moli_model.train()

        data_e = data_e.to(device)
        data_m = data_m.to(device)
        data_c = data_c.to(device)
        target = target.to(device)

        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            y_pred, zt = moli_model.forward(data_e, data_m, data_c)
            triplets = triplet_selector.get_triplets(zt, target)
            target = target.view(-1, 1)
            loss = gamma * trip_criterion(zt[triplets[:, 0], :], zt[triplets[:, 1], :],
                                          zt[triplets[:, 2], :]) + cross_entropy(y_pred, target)

            auc = roc_auc_score(target.detach().cpu(), y_pred.detach().cpu())

            moli_optimiser.zero_grad()
            loss.backward()
            moli_optimiser.step()

    return auc, loss.item()


def validate(data_loader, moli_model, device):
    y_true_test = []
    prediction_test = []
    moli_model.eval()
    for (data_e, data_m, data_c, target) in data_loader:
        tx_test_e = torch.FloatTensor(data_e).to(device)
        tx_test_m = torch.FloatTensor(data_m).to(device)
        tx_test_c = torch.FloatTensor(data_c).to(device)
        y_true_test.extend(target.view(-1, 1))
        with torch.no_grad():
            prediction, _ = moli_model.forward(tx_test_e, tx_test_m, tx_test_c)
            prediction_test.extend(prediction.cpu().detach())

    auc_test = roc_auc_score(y_true_test, prediction_test)
    return auc_test