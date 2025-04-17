import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
import numpy as np
from .utils import check_model_name, return_model
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import matplotlib as mpl
from scipy import interp
from sklearn.metrics import make_scorer, confusion_matrix, RocCurveDisplay

def _specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def _evaluate_classification_model(model, X_test, y_test, scoring, cv=None, group=None):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
        matthews_corrcoef

    if cv is not None:
        #print(cv)
        scores = {}
        for metric in scoring:
            # 没有random_state
            if metric == 'specificity':
                specificity_scorer = make_scorer(_specificity_score)
                metric_scores = cross_val_score(model, X_test, y_test, scoring=specificity_scorer, cv=cv, groups=group)
            elif metric == 'mcc':
                mr = 'matthews_corrcoef'
                metric_scores = cross_val_score(model, X_test, y_test, scoring=mr, cv=cv, groups=group)
            else:
                metric_scores = cross_val_score(model, X_test, y_test, scoring=metric, cv=cv, groups=group)
            #计算均值，可以加上sd
            scores[metric] = np.mean(metric_scores)
        return scores
    else:
        y_pred = model.predict(X_test)
        scores = {}
        for metric in scoring:

            if metric == 'accuracy':
                scores[metric] = accuracy_score(y_test, y_pred)
            elif metric == 'precision':
                scores[metric] = precision_score(y_test, y_pred)
            elif metric == 'recall':
                scores[metric] = recall_score(y_test, y_pred)
            elif metric == 'specificity':
                #tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                #scores[metric] = tn / (tn + fp)
                scores[metric] = _specificity_score(y_test, y_pred)
            elif metric == 'f1':
                scores[metric] = f1_score(y_test, y_pred)
            elif metric == 'roc_auc':
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                scores[metric] = roc_auc_score(y_test, y_pred_proba)
            elif metric == 'mcc':
                scores[metric] = matthews_corrcoef(y_test, y_pred)

        return scores


def _evaluate_regression_model(model, X_test, y_test, scoring, cv=None):
    """
    评估回归模型性能，支持交叉验证和单次评估。

    参数：
        model: 已训练的回归模型
        X_test: 测试数据特征
        y_test: 真实目标值
        scoring: 评估指标列表，可选 ['mae', 'rmse', 'mse', 'r2']
        cv: 交叉验证折数，若为 None 则进行单次评估

    返回：
        dict: 评估指标及其均值
    """

    if cv is not None:
        scores = {}
        for metric in scoring:
            if metric == 'rmse':
                rmse_scorer = make_scorer(mean_squared_error, squared=False)  # RMSE scorer
                metric_scores = cross_val_score(model, X_test, y_test, scoring=rmse_scorer, cv=cv)
            elif metric == 'mse':
                metric_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_squared_error', cv=cv)
                metric_scores = -metric_scores  # 由于sklearn计算的是负的 MSE，因此转换回来
            elif metric == 'mae':
                metric_scores = cross_val_score(model, X_test, y_test, scoring='neg_mean_absolute_error', cv=cv)
                metric_scores = -metric_scores  # Sklearn计算的是负的 MAE，需要转换回来
            elif metric == 'r2':
                metric_scores = cross_val_score(model, X_test, y_test, scoring='r2', cv=cv)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            scores[metric] = np.mean(metric_scores)  # 计算均值
        return scores

    else:
        y_pred = model.predict(X_test)
        scores = {}
        for metric in scoring:
            if metric == 'mse':
                scores[metric] = mean_squared_error(y_test, y_pred)
            elif metric == 'rmse':
                scores[metric] = mean_squared_error(y_test, y_pred, squared=False)
            elif metric == 'mae':
                scores[metric] = mean_absolute_error(y_test, y_pred)
            elif metric == 'r2':
                scores[metric] = r2_score(y_test, y_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return scores


def evaluate_model(model, X_test, y_test, scoring, cv=None, group=None, task='classification'):

    results_dict = {}

    if isinstance(model, dict):
        for model_name, model_data in model.items():
            eval_model = model_data['model']
            metrics_dict = {}

            if task == 'classification':
                metrics_dict = _evaluate_classification_model(eval_model, X_test, y_test, scoring, cv, group=group)
            else:
                metrics_dict = _evaluate_regression_model(eval_model, X_test, y_test, scoring, cv)

            results_dict[model_name] = metrics_dict
    else:
        # Single model

        if task == 'classification':
            metrics_dict = _evaluate_classification_model(model, X_test, y_test, scoring, cv)
        else:
            metrics_dict = _evaluate_regression_model(model, X_test, y_test, scoring, cv)

        mn = check_model_name(model)
        results_dict[mn] = metrics_dict

    return results_dict


def plot_roc_pr_curves(model, X, y, cv=None, plot_savedir=None, random_state=None):

    plt.figure(figsize=(12, 5))

    if isinstance(model, dict):
        # plt.subplot(1, 2, 1)
        for model_name, model_data in model.items():
            eval_model = model_data['model']
            if cv is not None:
                _plot_roc_curve_cv(eval_model, X, y, cv=cv, plot_savedir=plot_savedir, random_state=random_state)
            else:
                plt.subplot(1, 2, 1)
                _plot_roc_curve_single(eval_model, X, y, model_name)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves')
                plt.legend(loc='lower right')

        # plt.subplot(1, 2, 2)
        for model_name, model_data in model.items():
            eval_model = model_data['model']
            if cv is not None:
                _plot_pr_curve_cv(eval_model, X, y, cv=cv, plot_savedir=plot_savedir, random_state=random_state)
            else:
                plt.subplot(1, 2, 2)
                _plot_pr_curve_single(eval_model, X, y, model_name)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curves')
                plt.legend(loc='lower left')

    else:
        # plt.subplot(1, 2, 1)
        model_name = check_model_name(model)
        if cv is not None:
            _plot_roc_curve_cv(model, X, y, cv=cv, random_state=random_state)
        else:
            plt.subplot(1, 2, 1)
            _plot_roc_curve_single(model, X, y, model_name)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves')
            plt.legend(loc='lower right')

        # plt.subplot(1, 2, 2)
        if cv is not None:
            _plot_pr_curve_cv(model, X, y, cv=cv, random_state=random_state)
        else:
            plt.subplot(1, 2, 2)
            _plot_pr_curve_single(model, X, y, model_name)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend(loc='lower left')

    plt.tight_layout()
    if plot_savedir is not None:
        fn = plot_savedir + 'Test_roc_pr_curve.png'
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.show()


def _plot_roc_curve_single(model, X, y, model_name):
    y_score = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (model_name, roc_auc))


def _plot_roc_curve_cv(model,
                       X,
                       y,
                       cv,
                       plot_savedir,
                       random_state=None):
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    params = model.get_params()
    n_model = return_model(model)
    n_model.set_params(**params)
    model_name = check_model_name(model)

    # cv = cv
    model = model
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for index, (train, test) in enumerate(cv.split(X, y)):
        n_model.fit(X.iloc[train], y[train])
        plot = RocCurveDisplay.from_estimator(
            n_model, X.iloc[test], y[test],
            name="ROC fold {}".format(index),
            alpha=0.5,
            lw=1,
            ax=ax,
        )

        interp_tpr = np.interp(mean_fpr, plot.fpr, plot.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(plot.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f)" % (mean_auc),
        lw=2,
        alpha=0.8,
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve with CV (%s)" % (model_name),
    )
    ax.legend(loc="lower right")
    fn = plot_savedir + "ROC Curve with CV (%s)" % (model_name)
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.show()


def _plot_pr_curve_single(model, X, y, model_name):
    '''
    y_score = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_score)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label='%s (AUC = %0.2f)' % (model_name, pr_auc))
    
    y_score = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_score)
    pr_auc = auc(recall, precision)
    '''

    y_score = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_score)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label='%s (AUC = %0.2f)' % (model_name, pr_auc))


def _plot_pr_curve_cv(model,
                      X,
                      y,
                      cv,
                      plot_savedir,
                      random_state=None
                      ):
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1

    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    params = model.get_params()
    n_model = return_model(model)
    n_model.set_params(**params)
    model_name = check_model_name(model)

    prs = []
    aucs = []
    mean_recall = np.linspace(0, 1, 100)

    # plt.figure(figsize=(18 , 13))
    i = 0
    for train, test in cv.split(X, y):
        n_model.fit(X.iloc[train], y[train])
        probas_ = n_model.predict_proba(X.iloc[test])
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:, 1])
        prs.append(interp(mean_recall, precision, recall))
        pr_auc = auc(recall, precision)
        aucs.append(pr_auc)
        plt.plot(recall, precision, lw=1, alpha=0.5, label='Fold %d (AUCPR = %0.2f)' % (i + 1, pr_auc))
        i += 1

    mean_precision = np.mean(prs, axis=0)
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = np.std(aucs)
    plt.plot(mean_precision, mean_recall, color='b',
             label=r'Mean (AUCPR = %0.2f)' % (mean_auc),
             lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title("PR Curve with CV (%s)" % (model_name))
    plt.legend(loc=0)
    fn = plot_savedir + "PR Curve with CV (%s)" % (model_name)
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.show()

def _plot_regression_results(model_dict, X_test, y_test, metrics=['R2', 'MAE']):
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


