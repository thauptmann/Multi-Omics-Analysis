from pathlib import Path

import numpy as np
import yaml
from ax import Models, optimize
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from scipy.stats import sem
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.searchspaces import create_random_forest_search_space
best_auroc = -1
with open(Path('../../config/hyperparameter.yaml'), 'r') as stream:
    parameter = yaml.safe_load(stream)


def reset_best_auroc():
    global best_auroc
    best_auroc = 0


def check_best_auroc(best_reachable_auroc):
    global best_auroc
    return best_reachable_auroc < best_auroc


def set_best_auroc(new_auroc):
    global best_auroc
    if new_auroc > best_auroc:
        best_auroc = new_auroc


def train_validate_hyperparameter_set(x_train_val_concat, y_train_val, parameterization):
    skf = StratifiedKFold(n_splits=parameter['cv_splits'])
    all_validation_aurocs = []
    max_depth = parameterization['max_depth']
    if max_depth == 0:
        max_depth = None
    n_estimators = parameterization['n_estimators']
    min_samples_split = parameterization['min_samples_split']
    min_samples_leaf = parameterization['min_samples_leaf']

    iteration = 1
    for train_index, validate_index in tqdm(skf.split(x_train_val_concat, y_train_val), total=skf.get_n_splits(),
                                            desc="k-fold"):
        X_train_concat = x_train_val_concat[train_index]
        x_val_concat = x_train_val_concat[validate_index]
        Y_train = y_train_val[train_index]
        y_val = y_train_val[validate_index]
        scalerGDSC = StandardScaler()
        X_trainE = scalerGDSC.fit_transform(X_train_concat)
        x_val_e = scalerGDSC.transform(x_val_concat)

        random_forest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                               min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
        random_forest.fit(X_trainE, Y_train)
        predictions = random_forest.predict(x_val_e)
        val_auroc = roc_auc_score(y_val, predictions)
        all_validation_aurocs.append(val_auroc)

        if iteration < parameter['cv_splits']:
            open_folds = parameter['cv_splits'] - iteration
            remaining_best_results = np.ones(open_folds)
            best_possible_mean = np.mean(np.concatenate([all_validation_aurocs, remaining_best_results]))
            if check_best_auroc(best_possible_mean):
                print('Skip remaining folds.')
                break
        iteration += 1

    val_auroc = np.mean(all_validation_aurocs)
    standard_error_of_mean = sem(all_validation_aurocs)

    return {'auroc': (val_auroc, standard_error_of_mean)}


def optimise_random_forest_parameter(search_iterations, x_train_val_concat, y_train_val):
    evaluation_function = lambda parameterization: train_validate_hyperparameter_set(x_train_val_concat,
                                                                                     y_train_val,
                                                                                     parameterization)
    generation_strategy = GenerationStrategy(
        steps=[
            GenerationStep(model=Models.SOBOL, num_trials=-1),
        ],
        name="Sobol"
    )
    search_space = create_random_forest_search_space()
    best_parameters, values, experiment, model = optimize(
        total_trials=search_iterations,
        experiment_name='Super.FELT',
        objective_name='auroc',
        parameters=search_space,
        evaluation_function=evaluation_function,
        minimize=False,
        generation_strategy=generation_strategy,
    )
    return best_parameters, experiment


def compute_random_forest_metrics(x_test_concat, x_train_val_concat, best_parameters, extern_concat,
                                  extern_r, y_test, y_train_val):
    # retrain best
    max_depth = best_parameters['max_depth']
    n_estimator = best_parameters['n_estimator']
    min_samples_split = best_parameters['min_samples_split']
    min_samples_leaf = best_parameters['min_samples_leaf']

    final_scaler = StandardScaler()
    x_train_val_concat = final_scaler.fit_transform(x_train_val_concat)
    x_test_concat = final_scaler.fit_transform(x_test_concat)
    extern_concat = final_scaler.fit_transform(extern_concat)

    random_forest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimator,
                                           min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    random_forest.fit(x_train_val_concat, y_train_val)

    # Test
    test_Pred = random_forest.predict(x_test_concat)
    test_AUC = roc_auc_score(y_test, test_Pred)
    test_AUCPR = average_precision_score(y_test, test_Pred)

    # Extern
    extern_Pred = random_forest.predict(extern_concat)
    extern_AUC = roc_auc_score(extern_r, extern_Pred)
    extern_AUCPR = average_precision_score(extern_r, extern_Pred)
    return extern_AUC, extern_AUCPR, test_AUC, test_AUCPR



