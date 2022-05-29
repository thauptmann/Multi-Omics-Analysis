import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from models.stacking_model import StackingSplittedModel
from siamese_triplet.utils import AllTripletSelector
from utils.network_training_util import create_data_loader, create_sampler, \
    train_encoder, train_validate_classifier, train_classifier
from scipy.stats import sem
from models.stacking_model import Encoder, AutoEncoder

best_auroc = -1
cv_splits_inner = 5


def reset_best_auroc_splitted():
    global best_auroc
    best_auroc = 0


def optimise_hyperparameter_splitted(parameterization, x_e, x_m, x_c, y, device, pin_memory, stacking_type):
    mini_batch = parameterization['mini_batch']
    h_dim_e_encode = parameterization['h_dim_e_encode']
    h_dim_m_encode = parameterization['h_dim_m_encode']
    h_dim_c_encode = parameterization['h_dim_c_encode']
    lr_e = parameterization['lr_e']
    lr_m = parameterization['lr_m']
    lr_c = parameterization['lr_c']
    lr_clf = parameterization['lr_clf']
    dropout_e = parameterization['dropout_e']
    dropout_m = parameterization['dropout_m']
    dropout_c = parameterization['dropout_c']
    dropout_clf = parameterization['dropout_clf']
    weight_decay = parameterization['weight_decay']
    epochs_e = parameterization['epochs_e']
    epochs_m = parameterization['epochs_m']
    epochs_c = parameterization['epochs_c']
    epochs_clf = parameterization['epochs_clf']
    margin = parameterization['margin']
    triplet_selector = AllTripletSelector()
    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)

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
        x_validate_e = torch.FloatTensor(scaler_gdsc.transform(x_validate_e)).to(device)

        # Initialisation
        sampler = create_sampler(y_train)
        train_loader = create_data_loader(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                          torch.FloatTensor(x_train_c),
                                          torch.FloatTensor(y_train), mini_batch, pin_memory, sampler)
        IE_dim = x_train_e.shape[-1]
        IM_dim = x_train_m.shape[-1]
        IC_dim = x_train_c.shape[-1]

        e_encoder = Encoder(IE_dim, h_dim_e_encode, dropout_e).to(device)
        m_encoder = Encoder(IM_dim, h_dim_m_encode, dropout_m).to(device)
        c_encoder = Encoder(IC_dim, h_dim_c_encode, dropout_c).to(device)

        E_optimizer = torch.optim.Adagrad(e_encoder.parameters(), lr=lr_e, weight_decay=weight_decay)
        M_optimizer = torch.optim.Adagrad(m_encoder.parameters(), lr=lr_m, weight_decay=weight_decay)
        C_optimizer = torch.optim.Adagrad(c_encoder.parameters(), lr=lr_c, weight_decay=weight_decay)

        # train each Supervised_Encoder with triplet loss
        train_encoder(epochs_e, E_optimizer, triplet_selector, device, e_encoder, train_loader, trip_loss_fun, 0)
        train_encoder(epochs_m, M_optimizer, triplet_selector, device, m_encoder, train_loader, trip_loss_fun, 1)
        train_encoder(epochs_c, C_optimizer, triplet_selector, device, c_encoder, train_loader, trip_loss_fun, 2)

        # train classifier
        classifier_input_dimensions = [h_dim_e_encode, h_dim_m_encode, h_dim_c_encode]
        classifier = StackingSplittedModel(classifier_input_dimensions, dropout_clf, stacking_type)
        optimiser = torch.optim.Adagrad(classifier.parameters(), lr=lr_clf,
                                        weight_decay=weight_decay)
        val_auroc = train_validate_classifier(epochs_clf, device, e_encoder,
                                              m_encoder, c_encoder, train_loader,
                                              optimiser,
                                              x_validate_e, x_validate_m, x_validate_c,
                                              y_validate, classifier)
        aucs_validate.append(val_auroc)

        if iteration < cv_splits_inner:
            open_folds = cv_splits_inner - iteration
            remaining_best_results = np.ones(open_folds)
            best_possible_mean = np.mean(np.concatenate([aucs_validate, remaining_best_results]))
            if check_best_auroc(best_possible_mean):
                print('Skip remaining folds.')
                break
        iteration += 1

    mean = np.mean(aucs_validate)
    set_best_auroc_splitted(mean)
    standard_error_of_mean = sem(aucs_validate)

    return {'auroc': (mean, standard_error_of_mean)}


def check_best_auroc(best_reachable_auroc):
    global best_auroc
    return best_reachable_auroc < best_auroc


def set_best_auroc_splitted(new_auroc):
    global best_auroc
    if new_auroc > best_auroc:
        best_auroc = new_auroc


def train_final_splitted(parameterization, x_train_e, x_train_m, x_train_c, y_train, device,
                         stacking_type):
    mini_batch = parameterization['mini_batch']
    h_dim_e_encode = parameterization['h_dim_e_encode']
    h_dim_m_encode = parameterization['h_dim_m_encode']
    h_dim_c_encode = parameterization['h_dim_c_encode']
    lr_e = parameterization['lr_e']
    lr_m = parameterization['lr_m']
    lr_c = parameterization['lr_c']
    lr_clf = parameterization['lr_clf']
    dropout_e = parameterization['dropout_e']
    dropout_m = parameterization['dropout_m']
    dropout_c = parameterization['dropout_c']
    dropout_clf = parameterization['dropout_clf']
    weight_decay = parameterization['weight_decay']
    epochs_e = parameterization['epochs_e']
    epochs_m = parameterization['epochs_m']
    epochs_c = parameterization['epochs_c']
    epochs_clf = parameterization['epochs_clf']
    margin = parameterization['margin']

    trip_loss_fun = torch.nn.TripletMarginLoss(margin=margin, p=2)
    sampler = create_sampler(y_train)
    final_scaler = StandardScaler()
    x_train_e = final_scaler.fit_transform(x_train_e)
    trainDataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_train_e), torch.FloatTensor(x_train_m),
                                                  torch.FloatTensor(x_train_c),
                                                  torch.FloatTensor(y_train.astype(int)))
    train_loader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=mini_batch, shuffle=False,
                                               num_workers=8, sampler=sampler, drop_last=True)
    IE_dim = x_train_e.shape[-1]
    IM_dim = x_train_m.shape[-1]
    IC_dim = x_train_c.shape[-1]
    final_E_encoder = Encoder(IE_dim, h_dim_e_encode, dropout_e).to(device)
    final_M_encoder = Encoder(IM_dim, h_dim_m_encode, dropout_m).to(device)
    final_C_encoder = Encoder(IC_dim, h_dim_c_encode, dropout_c).to(device)
    E_optimizer = torch.optim.Adagrad(final_E_encoder.parameters(), lr=lr_e, weight_decay=weight_decay)
    M_optimizer = torch.optim.Adagrad(final_M_encoder.parameters(), lr=lr_m, weight_decay=weight_decay)
    C_optimizer = torch.optim.Adagrad(final_C_encoder.parameters(), lr=lr_c, weight_decay=weight_decay)
    triplet_selector = AllTripletSelector()
    classifier_input_dimensions = [h_dim_e_encode, h_dim_m_encode, h_dim_c_encode]
    final_classifier = StackingSplittedModel(classifier_input_dimensions, dropout_clf, stacking_type)
    classifier_optimizer = torch.optim.Adagrad(final_classifier.parameters(), lr=lr_clf, weight_decay=weight_decay)

    # train each Supervised_Encoder with triplet loss
    train_encoder(epochs_e, E_optimizer, triplet_selector, device, final_E_encoder, train_loader,
                  trip_loss_fun, 0)
    train_encoder(epochs_m, M_optimizer, triplet_selector, device, final_M_encoder, train_loader,
                  trip_loss_fun, 1)
    train_encoder(epochs_c, C_optimizer, triplet_selector, device, final_C_encoder, train_loader,
                  trip_loss_fun, 2)

    # train classifier
    train_classifier(final_classifier, epochs_clf, train_loader, classifier_optimizer, final_E_encoder,
                     final_M_encoder, final_C_encoder, device)
    return final_E_encoder, final_M_encoder, final_C_encoder, final_classifier, final_scaler


def splitted_test(x_test_e, x_test_m, x_test_c, y_test, device, final_C_Supervised_Encoder,
                  final_classifier, final_E_Supervised_Encoder, final_M_Supervised_Encoder,):
    encoded_test_E = final_E_Supervised_Encoder(torch.FloatTensor(x_test_e).to(device))
    encoded_test_M = final_M_Supervised_Encoder(torch.FloatTensor(x_test_m).to(device))
    encoded_test_C = final_C_Supervised_Encoder(torch.FloatTensor(x_test_c).to(device))
    test_predictions = final_classifier(encoded_test_E, encoded_test_M, encoded_test_C)
    test_y_true = y_test
    test_y_predictions = test_predictions.cpu().detach().numpy()
    test_AUC = roc_auc_score(test_y_true, test_y_predictions)
    test_AUCPR = average_precision_score(test_y_true, test_y_predictions)
    return test_AUC, test_AUCPR


def compute_splitted_metrics(x_test_e, x_test_m, x_test_c, x_train_validate_e, x_train_validate_m,
                             x_train_validate_c, best_parameters, device, extern_e, extern_m,
                             extern_c, extern_r, y_test, y_train_validate,
                             stacking_type):
    # retrain best
    final_E_Supervised_Encoder, final_M_Supervised_Encoder, final_C_Supervised_Encoder, final_Classifier, \
    final_scaler_gdsc = train_final_splitted(best_parameters, x_train_validate_e, x_train_validate_m,
                                             x_train_validate_c, y_train_validate, device,
                                             stacking_type)
    x_test_e = torch.FloatTensor(final_scaler_gdsc.transform(x_test_e))
    # Test
    test_AUC, test_AUCPR = splitted_test(x_test_e, x_test_m, x_test_c, y_test, device, final_C_Supervised_Encoder,
                                         final_Classifier, final_E_Supervised_Encoder, final_M_Supervised_Encoder)
    # Extern
    external_AUC, external_AUCPR = splitted_test(extern_e, extern_m, extern_c, extern_r, device,
                                                 final_C_Supervised_Encoder,
                                                 final_Classifier, final_E_Supervised_Encoder,
                                                 final_M_Supervised_Encoder)
    return external_AUC, external_AUCPR, test_AUC, test_AUCPR
