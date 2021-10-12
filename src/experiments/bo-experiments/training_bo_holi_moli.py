import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import trange, tqdm
from models.bo_holi_moli_model import AdaptiveMoli
from siamese_triplet.utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector
from utils import network_training_util
from utils.network_training_util import BceWithTripletsToss, get_triplet_selector, get_loss_fn, create_data_loader
from scipy.stats import sem

best_auroc = 0


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def train_and_validate(parameterization, x_e, x_m, x_c, y, device, pin_memory, deactivate_skip_bad_iterations,
                       triplet_selector_type):
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
    iteration = 1
    skf = StratifiedKFold(n_splits=cv_splits)
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
        x_validate_e = scaler_gdsc.transform(x_validate_e)

        # Initialisation
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_train])

        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        train_loader = create_data_loader(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                          torch.FloatTensor(x_train_c),
                                          torch.FloatTensor(y_train), True, pin_memory, sampler)

        validation_loader = create_data_loader(torch.FloatTensor(x_validate_e),
                                               torch.FloatTensor(x_validate_m),
                                               torch.FloatTensor(x_validate_c),
                                               torch.FloatTensor(y_validate), mini_batch*4, False, pin_memory)

        n_sample_e, ie_dim = x_train_e.shape
        _, im_dim = x_train_m.shape
        _, ic_dim = x_train_c.shape

        triplet_selector = get_triplet_selector(margin, triplet_selector_type)
        loss_fn = get_loss_fn(margin, gamma, triplet_selector)

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

        for _ in trange(epochs, desc='Epoch'):
            network_training_util.train(train_loader, moli_model, moli_optimiser, loss_fn, device, gamma)

        # validate
        auc_validate, _ = network_training_util.validate(validation_loader, moli_model, device)
        aucs_validate.append(auc_validate)

        if not deactivate_skip_bad_iterations:
            open_folds = cv_splits - iteration
            remaining_best_results = np.ones(open_folds)
            best_possible_mean = np.mean(np.concatenate([aucs_validate, remaining_best_results]))
            if best_possible_mean < best_auroc:
                print('Skip remaining folds.')
                break

    mean = np.mean(aucs_validate)
    standard_error_of_mean = sem(aucs_validate)

    return {'auroc': (mean, standard_error_of_mean)}


def check_best_auroc(new_auroc):
    global best_auroc
    if new_auroc < best_auroc:
        best_auroc = new_auroc


def train_final(parameterization, x_train_e, x_train_m, x_train_c, y_train, device, pin_memory,
                triplet_selector_type='all'):
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

    triplet_selector = get_triplet_selector(margin, triplet_selector_type)
    loss_fn = get_loss_fn(margin, gamma, triplet_selector)

    depths = [depth_1, depth_2, depth_3, depth_4, depth_5]
    input_sizes = [ie_dim, im_dim, ic_dim]
    dropout_rates = [dropout_rate_e, dropout_rate_m, dropout_rate_c, dropout_rate_middle, dropout_rate_clf]
    output_sizes = [h_dim1, h_dim2, h_dim3, h_dim4, h_dim5]
    moli_model = AdaptiveMoli(input_sizes, output_sizes, dropout_rates, combination, depths).to(device)

    moli_optimiser = torch.optim.Adagrad([
        {'params': moli_model.left_encoder.parameters(), 'lr': lr_middle,},
        {'params': moli_model.expression_encoder.parameters(), 'lr': lr_e},
        {'params': moli_model.mutation_encoder.parameters(), 'lr': lr_m},
        {'params': moli_model.cna_encoder.parameters(), 'lr': lr_c},
        {'params': moli_model.classifier.parameters(), 'lr': lr_cl}], weight_decay=weight_decay
    )

    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight),
                                    replacement=True)
    train_loader = create_data_loader(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                      torch.FloatTensor(x_train_c),
                                      torch.FloatTensor(y_train), mini_batch, True, pin_memory, sampler)
    for _ in range(epochs):
        network_training_util.train(train_loader, moli_model, moli_optimiser, loss_fn, device, gamma)
    return moli_model, train_scaler_gdsc
