import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm
from models.moma_model import Moma
from utils.network_training_util import create_sampler, create_data_loader
from scipy.stats import sem

best_auroc = -1
cv_splits_inner = 5


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def optimise_hyperparameter(parameterization, x_e, x_m, x_c, y, device, pin_memory):
    mini_batch = parameterization['mini_batch']
    h_dim_classifier = parameterization['h_dim_classifier']
    modules = parameterization['modules']
    lr_expression = parameterization['lr_expression']
    lr_mutation = parameterization['lr_mutation']
    lr_cna = parameterization['lr_cna']
    lr_classifier = parameterization['lr_classifier']
    weight_decay = parameterization['weight_decay']
    epochs = parameterization['epochs']

    aucs_validate = []
    iteration = 1
    skf = StratifiedKFold(n_splits=cv_splits_inner)
    for train_index, validate_index in tqdm(skf.split(x_e, y), total=skf.get_n_splits(),
                                            desc="k-fold"):
        x_train_e = x_e[train_index]
        x_train_m = x_m[train_index]
        x_train_c = x_c[train_index]
        y_train = y[train_index]

        x_validate_e = x_e[validate_index]
        x_validate_m = x_m[validate_index]
        x_validate_c = x_c[validate_index]
        y_validate = y[validate_index]

        scaler_gdsc = StandardScaler()
        x_train_e = scaler_gdsc.fit_transform(x_train_e)

        # Initialisation
        sampler = create_sampler(y_train)
        train_loader = create_data_loader(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                          torch.FloatTensor(x_train_c),
                                          torch.FloatTensor(y_train), mini_batch, pin_memory, sampler)

        loss_fn = torch.nn.BCELoss()
        e_in = x_train_e.shape[-1]
        m_in = x_train_m.shape[-1]
        c_in = x_train_c.shape[-1]
        moma_model = Moma(e_in, m_in, c_in, h_dim_classifier, modules).to(device)

        moma_optimiser = torch.optim.Adagrad([
            {'params': moma_model.expression_FC1_x.parameters(), 'lr': lr_expression},
            {'params': moma_model.expression_FC1_y.parameters(), 'lr': lr_expression},
            {'params': moma_model.mutation_FC1_x.parameters(), 'lr': lr_mutation},
            {'params': moma_model.mutation_FC1_y.parameters(), 'lr': lr_mutation},
            {'params': moma_model.cna_FC1_x.parameters(), 'lr': lr_cna},
            {'params': moma_model.cna_FC1_y.parameters(), 'lr': lr_cna},
           # {'params': moma_model.expression_FC2.parameters(), 'lr': lr_classifier},
            {'params': moma_model.expression_FC3.parameters(), 'lr': lr_classifier},
           # {'params': moma_model.mutation_FC2.parameters(), 'lr': lr_classifier},
            {'params': moma_model.mutation_FC3.parameters(), 'lr': lr_classifier},
           # {'params': moma_model.cna_FC2.parameters(), 'lr': lr_classifier},
            {'params': moma_model.cna_FC3.parameters(), 'lr': lr_classifier}],
            weight_decay=weight_decay)

        for _ in trange(epochs, desc='Epoch'):
            train_moma(train_loader, moma_model, moma_optimiser, loss_fn, device)

        # validate
        auc_validate, _ = test_moma(moma_model, scaler_gdsc, torch.FloatTensor(x_validate_e),
                                    torch.FloatTensor(x_validate_m),
                                    torch.FloatTensor(x_validate_c), y_validate, device)
        aucs_validate.append(auc_validate)

        if iteration < cv_splits_inner:
            open_folds = cv_splits_inner - iteration
            remaining_best_results = np.ones(open_folds)
            best_possible_mean = np.mean(np.concatenate([aucs_validate, remaining_best_results]))
            if check_best_auroc(best_possible_mean):
                print('Skip remaining folds.')
                break
        iteration += 1

    mean = np.mean(aucs_validate)
    set_best_auroc(mean)
    standard_error_of_mean = sem(aucs_validate)

    return {'auroc': (mean, standard_error_of_mean)}


def check_best_auroc(best_reachable_auroc):
    global best_auroc
    return best_reachable_auroc < best_auroc


def set_best_auroc(new_auroc):
    global best_auroc
    if new_auroc > best_auroc:
        best_auroc = new_auroc


def train_final(parameterization, x_train_e, x_train_m, x_train_c, y_train, device, pin_memory):
    mini_batch = parameterization['mini_batch']
    h_dim_classifier = parameterization['h_dim_classifier']
    modules = parameterization['modules']
    lr_expression = parameterization['lr_expression']
    lr_mutation = parameterization['lr_mutation']
    lr_cna = parameterization['lr_cna']
    lr_classifier = parameterization['lr_classifier']
    weight_decay = parameterization['weight_decay']
    epochs = parameterization['epochs']

    train_scaler_gdsc = StandardScaler()
    train_scaler_gdsc.fit(x_train_e)
    x_train_e = train_scaler_gdsc.transform(x_train_e)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    e_in = x_train_e.shape[-1]
    m_in = x_train_m.shape[-1]
    c_in = x_train_c.shape[-1]
    moma_model = Moma(e_in, m_in, c_in, h_dim_classifier, modules).to(device)

    moma_optimiser = torch.optim.Adagrad([
        {'params': moma_model.expression_FC1_x.parameters(), 'lr': lr_expression},
        {'params': moma_model.expression_FC1_y.parameters(), 'lr': lr_expression},
        {'params': moma_model.mutation_FC1_x.parameters(), 'lr': lr_mutation},
        {'params': moma_model.mutation_FC1_y.parameters(), 'lr': lr_mutation},
        {'params': moma_model.cna_FC1_x.parameters(), 'lr': lr_cna},
        {'params': moma_model.cna_FC1_y.parameters(), 'lr': lr_cna},
        #{'params': moma_model.expression_FC2.parameters(), 'lr': lr_classifier},
        {'params': moma_model.expression_FC3.parameters(), 'lr': lr_classifier},
        #{'params': moma_model.mutation_FC2.parameters(), 'lr': lr_classifier},
        {'params': moma_model.mutation_FC3.parameters(), 'lr': lr_classifier},
        #{'params': moma_model.cna_FC2.parameters(), 'lr': lr_classifier},
        {'params': moma_model.cna_FC3.parameters(), 'lr': lr_classifier}],
        weight_decay=weight_decay)

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)
    train_loader = create_data_loader(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                      torch.FloatTensor(x_train_c),
                                      torch.FloatTensor(y_train), mini_batch, pin_memory, sampler)

    for epoch in range(epochs):
        train_moma(train_loader, moma_model, moma_optimiser, loss_fn, device)
    return moma_model, train_scaler_gdsc


def train_moma(train_loader, model, optimiser, loss_fn, device):
    y_true = []
    model.train()
    for (data_e, data_m, data_c, target) in train_loader:
        if torch.mean(target) != 0. and torch.mean(target) != 1.:
            optimiser.zero_grad()
            y_true.extend(target)

            data_e = data_e.to(device)
            data_m = data_m.to(device)
            data_c = data_c.to(device)
            target = target.to(device)
            expression_logit, mutation_logit, cna_logit = model.forward(data_e, data_m, data_c)
            loss = loss_fn(torch.squeeze(expression_logit), target) + loss_fn(torch.squeeze(mutation_logit), target) \
                   + loss_fn(torch.squeeze(cna_logit), target)
            loss.backward()
            optimiser.step()

sigmoid = torch.nn.Sigmoid()


def test_moma(model, scaler, extern_e, extern_m, extern_c, test_r, device):
    model = model.to(device)
    extern_e = torch.FloatTensor(scaler.transform(extern_e)).to(device)
    extern_m = torch.FloatTensor(extern_m).to(device)
    extern_c = torch.FloatTensor(extern_c).to(device)
    test_y = torch.FloatTensor(test_r.astype(int))
    model.eval()
    with torch.no_grad():
        expression_logit, mutation_logit, cna_logit = model.forward(extern_e, extern_m, extern_c)
    stacked = np.stack([expression_logit.cpu(), mutation_logit.cpu(), cna_logit.cpu()])
    mean_logits = np.mean(stacked, axis=0)
    probabilities = sigmoid(mean_logits)
    auc_validate = roc_auc_score(test_y, probabilities)
    auprc_validate = average_precision_score(test_y, probabilities)
    return auc_validate, auprc_validate
