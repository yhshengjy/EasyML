from sklearn.model_selection import cross_val_score
from sklearn.metrics import *

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.ensemble import *
from sklearn.linear_model import *
from .utils import *

from hpsklearn import HyperoptEstimator
from hyperopt import tpe
from .hyperp_space import *

import matplotlib.pyplot as plt

'''
def load_hyperparameter_space(filename):
    with open(filename, 'r') as file:
        hyperparameter_space = json.load(file)
    return hyperparameter_space
'''


def train_classification_model(X_train,
                               y_train,
                               # X_test,
                               # y_test,
                               model_names,
                               group=None,
                               scoring='accuracy',
                               search_methods='bayesian',
                               n_iter=100,
                               cv=5,
                               random_state=0,
                               n_jobs=-1):

    if model_names in ['any_classifier', 'any_regressor'] and search_methods != 'bayesian':
        raise ValueError("Invalid combination of model_names and search_methods")

    # hyperparameter_space = load_hyperparameter_space(hyperparameter_space_file)
    # hyperparameter_space = Hyperparameters.param_grid_space

    if 'XGBoostClassifier' in model_names:
        X_train.columns = clean_feature_names(X_train.columns)
        # X_test.columns = clean_feature_names(X_test.columns)

    all_models = {}

    for model_name in model_names:

        if search_methods == 'grid':
            # param_space = hyperparameter_space[model_name]
            param_space = Hyperparameters.get_space('cls_grid', model_name)
            estimator = eval(param_space['estimator'])
            param_grid = param_space['param_grid']

            grid_search = GridSearchCV(estimator,
                                       param_grid,
                                       cv=cv,
                                       scoring=scoring,
                                       n_jobs=n_jobs)

            grid_search.fit(X_train, y_train, groups=group)
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_

        if search_methods == 'random':
            # param_space = hyperparameter_space[model_name]
            param_space = Hyperparameters.get_space('cls_grid', model_name)
            estimator = eval(param_space['estimator'])
            param_grid = param_space['param_grid']

            # random_state
            random_search = RandomizedSearchCV(estimator,
                                               param_grid,
                                               n_iter=n_iter,
                                               cv=cv,
                                               scoring=scoring,
                                               n_jobs=n_jobs,
                                               random_state=random_state)

            random_search.fit(X_train, y_train, groups=group)
            best_model = random_search.best_estimator_
            best_score = random_search.best_score_

        if search_methods == 'bayesian':
            # clf = hp.choice('t2dclf', [random_classier_ht()])
            if model_name == 'RandomForest':
                clf = Hyperparameters.random_classier_ht()
            elif model_name == 'ExtraTrees':
                clf = Hyperparameters.extratrees_classifier_ht()
            elif model_name == 'LogisticRegression':
                clf = Hyperparameters.logreg_ht()
            elif model_name == 'GradientBoosting':
                clf = Hyperparameters.gbc_ht()
            elif model_name == 'XGBoost':
                clf = Hyperparameters.xgbc_ht()
            elif model_name == 'SVC':
                clf = Hyperparameters.svc_ht()
            elif model_name == 'all':
                clf = Hyperparameters.all_clf()
            else:
                ValueError("Invalid model_name. Please choose a valid model.")

            estim = HyperoptEstimator(classifier=clf,
                                      algo=tpe.suggest,
                                      preprocessing=[],
                                      loss_fn=map_metric_to_function(scoring),
                                      max_evals=n_iter,
                                      trial_timeout=60,
                                      n_jobs=n_jobs)

            estim.fit(X_train, y_train, n_folds=5, cv_shuffle=True)
            best_model = estim.best_model()['learner']
            # best_score = estim.score(X_test, y_test)
            best_score = cross_val_score(best_model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs).mean()

        all_models[model_name] = {'model': best_model, 'score': best_score}

    return all_models


def train_regression_model(X_train,
                           y_train,
                           model_names,
                           scoring='r2_score',
                           search_methods='bayesian',
                           n_iter=100,
                           cv=5,
                           random_state=0,
                           n_jobs=-1):

    # hyperparameter_space = load_hyperparameter_space(hyperparameter_space_file)
    hyperparameter_space = Hyperparameters._param_reg_grid_space

    all_models = {}

    for model_name in model_names:

        if search_methods == 'grid':
            param_space = Hyperparameters.get_space('reg_grid', model_name)
            estimator = eval(param_space['estimator'])
            param_grid = param_space['param_grid']
            grid_search = GridSearchCV(estimator,
                                       param_grid,
                                       cv=cv,
                                       scoring=scoring,
                                       n_jobs=n_jobs)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_

        if search_methods == 'random':
            param_space = Hyperparameters.get_space('reg_grid', model_name)
            estimator = eval(param_space['estimator'])
            param_grid = param_space['param_grid']
            random_search = RandomizedSearchCV(estimator,
                                               param_grid,
                                               n_iter=n_iter,
                                               cv=cv,
                                               scoring=scoring,
                                               n_jobs=n_jobs,
                                               random_state=random_state)
            random_search.fit(X_train, y_train)
            best_model = random_search.best_estimator_
            best_score = random_search.best_score_

        if search_methods == 'bayesian':
            if model_name == 'RandomForest':
                ref = Hyperparameters.randomforest_regressor_ht()
            elif model_name == 'ExtraTrees':
                ref = Hyperparameters.extratrees_regressor_ht()
            elif model_name == 'Lasso':
                ref = Hyperparameters.lasso_ht()
            elif model_name == 'Ridge':
                ref = Hyperparameters.ridge_ht
            elif model_name == 'GradientBoosting':
                ref = Hyperparameters.gbr_ht()
            elif model_name == 'XGBoost':
                ref = Hyperparameters.xgbr_ht()
            else:
                ValueError("Invalid model_name. Please choose a valid model.")

            estim = HyperoptEstimator(classifier=ref,
                                      algo=tpe.suggest,
                                      preprocessing=[],
                                      loss_fn=map_metric_to_function(scoring),
                                      max_evals=n_iter,
                                      trial_timeout=60)

            estim.fit(X_train, y_train, n_folds=5, cv_shuffle=True)
            best_model = estim.best_model()['learner']
            # best_score = estim.score(X_test, y_test)

        all_models[model_name] = {'model': best_model, 'score': best_score}

    # plot_roc_pr_curves_from_dict(all_models, X_test, y_test)
    # plot_regression_results(model_dict, X_test, y_test, metrics=scoring)

    return all_models


def plot_regression_results(model_dict, X_test, y_test, metrics=['R2', 'MAE']):
    """
    Plot regression results for models stored in a dictionary.

    Parameters:
    - model_dict: Dictionary containing each model and its name.
    - X_test, y_test: Test data.
    - metrics: List of additional metrics to display on the plot.
    """

    plt.figure(figsize=(10, 8))

    for model_name, model_data in model_dict.items():
        model = model_data['model']
        y_pred = model.predict(X_test)

        # Scatter plot of true vs. predicted values
        plt.scatter(y_test, y_pred, label=model_name)

    # Plot diagonal line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='gray')

    # Calculate and display additional metrics
    if 'R2' in metrics:
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.9, f'R2: {r2:.2f}', transform=plt.gca().transAxes)
    if 'MAE' in metrics:
        mae = mean_absolute_error(y_test, y_pred)
        plt.text(0.05, 0.85, f'MAE: {mae:.2f}', transform=plt.gca().transAxes)
    if 'RMSE' in metrics:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        plt.text(0.05, 0.8, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes)

    # Plot settings
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Results')
    plt.legend()
    plt.grid(True)

    plt.show()


def compute_roc(estimator, X_train, X_test, y_train, y_test):
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    y_train_prob = estimator.predict_proba(X_train)[:, 1]

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    auc_train = roc_auc_score(y_train, y_train_prob)

    y_test_prob = estimator.predict_proba(X_test)[:, 1]

    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    auc_test = roc_auc_score(y_test, y_test_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.2f}')
    plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


def acc_loss(target, pred):
    return 1 - accuracy_score(target, pred)


def f1_loss(target, pred):
    return 1 - f1_score(target, pred)


def auroc_loss(target, pred):
    return 1 - roc_auc_score(target, pred)


def mcc_loss(target, pred):
    return 1 - matthews_corrcoef(target, pred)


def mse_loss(target, pred):
    return mean_squared_error(target, pred)


def rmse_loss(target, pred):
    return mean_squared_error(target, pred, squared=False)


def mae_loss(target, pred):
    return mean_absolute_error(target, pred)


def r2_loss(target, pred):
    return 1 - r2_score(target, pred)


def map_metric_to_function(metric_name):
    metric_name = metric_name.lower()

    metric_functions = {
        'accuracy': acc_loss,
        'roc_auc': auroc_loss,
        'f1': f1_loss,
        'mcc': mcc_loss,
        'mse': mse_loss,
        'rmse': rmse_loss,
        'mae': mae_loss,
        'r2': r2_loss
    }

    return metric_functions.get(metric_name, None)
