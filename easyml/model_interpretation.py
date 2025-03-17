import shap
from sklearn.inspection import permutation_importance
from sklearn.ensemble import *
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, RepeatedKFold, KFold
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from .utils import *
import matplotlib.pyplot as plt


def interpret_model(model,
                    X_test,
                    y_test,
                    cv_f=None,
                    method='permutation',
                    n_repeats=100,
                    n_features=20,
                    use_whole_dataset=False,
                    index=[0],
                    plotpath='./',
                    show=True,
                    random_state=0):

    if method not in ['permutation', 'shap']:
        raise ValueError("Invalid method. Supported methods are 'permutation' and 'shap'.")

    if cv_f is not None:
        if method == 'permutation':
            # Permutation feature importance
            feature_importances = _calculate_permutation_feature_importance_cv(model,
                                                                               X_test,
                                                                               y_test,
                                                                               cv_f,
                                                                               n_repeats=n_repeats,
                                                                               n_features=n_features,
                                                                               plotpath=plotpath,
                                                                               random_state=random_state)
            return feature_importances

        elif method == 'shap':
            # SHAP values
            if use_whole_dataset:
                shap_values = _calculate_shap_values(model,
                                                     X_test,
                                                     index,
                                                     plotpath,
                                                     show)
            else:
                if isinstance(cv_f, KFold) or isinstance(cv_f, StratifiedKFold):
                    shap_values = _calculate_shap_values_cv(model,
                                                            X_test,
                                                            y_test,
                                                            cv_f,
                                                            index,
                                                            plotpath)

                elif isinstance(cv_f, RepeatedKFold) or isinstance(cv_f, RepeatedStratifiedKFold):
                    shap_values = _calculate_shap_values_repcv(model,
                                                               X_test,
                                                               y_test,
                                                               cv_f,
                                                               index,
                                                               plotpath)
            return shap_values

    else:
        if method == 'permutation':
            # Permutation feature importance
            feature_importances = _calculate_permutation_feature_importance(model=model,
                                                                            X_test=X_test,
                                                                            y_test=y_test,
                                                                            n_repeats=n_repeats,
                                                                            n_features=n_features,
                                                                            plotpath=plotpath,
                                                                            random_state=random_state)
            return feature_importances

        elif method == 'shap':
            # SHAP values
            shap_values = _calculate_shap_values(model,
                                                 X_test,
                                                 index,
                                                 plotpath,
                                                 show)
            return shap_values


def _calculate_permutation_feature_importance(model,
                                              X_test,
                                              y_test,
                                              n_repeats,
                                              n_features,
                                              plotpath,
                                              random_state):

    perm_imp = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=random_state)

    filtered_features = X_test.columns[perm_imp.importances_mean > 0]
    filtered_importances = perm_imp.importances_mean[perm_imp.importances_mean > 0]

    sorted_indices = filtered_importances.argsort()[::-1]
    sorted_features = filtered_features[sorted_indices]
    sorted_importances = filtered_importances[sorted_indices]

    if n_features is not None:
        sorted_features = sorted_features[:n_features]
        sorted_importances = sorted_importances[:n_features]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Permutation Feature Importance')
    plt.gca().invert_yaxis()  # 确保图中特征按重要性从大到小显示

    if plotpath is not None:
        fn = plotpath + 'permutation_feature_importance'
        plt.savefig(fn, dpi=300, bbox_inches='tight')

    plt.show()

    return perm_imp.importances_mean

def _calculate_permutation_feature_importance_cv(model,
                                                 X,
                                                 y,
                                                 kf,
                                                 n_repeats,
                                                 n_features,
                                                 plotpath,
                                                 random_state):

    import numpy as np

    perm_importances = []

    # Cross-validation
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        params = model.get_params()
        model1 = return_model(model)
        model1.set_params(**params)

        model1.fit(X_train, y_train)

        result = permutation_importance(model1, X_val, y_val, n_repeats=n_repeats, random_state=random_state)
        perm_importances.append(result.importances_mean)

    mean_importances = np.mean(perm_importances, axis=0)
    std_importances = np.std(perm_importances, axis=0)

    feature_names = X.columns if hasattr(X, 'columns') else np.arange(X.shape[1])

    sorted_indices = np.argsort(mean_importances)[::-1]
    sorted_features = np.array(feature_names)[sorted_indices]
    sorted_importances = mean_importances[sorted_indices]
    sorted_std_importances = std_importances[sorted_indices]

    if n_features is not None:
        sorted_features = sorted_features[:n_features]
        sorted_importances = sorted_importances[:n_features]
        sorted_std_importances = sorted_std_importances[:n_features]

    # Plot bar chart of feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, xerr=sorted_std_importances)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Permutation Feature Importance with Cross-Validation')
    plt.gca().invert_yaxis()
    plt.show()

    if plotpath is not None:
        fn = plotpath + 'permutation_feature_importance'
        plt.savefig(fn, dpi=300, bbox_inches='tight')

    return sorted_features, sorted_importances, sorted_std_importances

def _calculate_permutation_feature_importance_cv(model,
                                                 X,
                                                 y,
                                                 kf,
                                                 n_repeats,
                                                 plotpath,
                                                 random_state):

    import numpy as np

    perm_importances = []

    # Cross-validation
    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        params = model.get_params()
        model1 = return_model(model)
        model1.set_params(**params)

        model1.fit(X_train, y_train)

        result = permutation_importance(model1, X_val, y_val, n_repeats=n_repeats, random_state=random_state)
        perm_importances.append(result.importances_mean)

    mean_importances = np.mean(perm_importances, axis=0)
    std_importances = np.std(perm_importances, axis=0)

    feature_names = X.columns if hasattr(X, 'columns') else np.arange(X.shape[1])

    # Sort features by importance
    sorted_indices = np.argsort(mean_importances)[::-1]
    sorted_features = np.array(feature_names)[sorted_indices]
    sorted_importances = mean_importances[sorted_indices]
    sorted_std_importances = std_importances[sorted_indices]

    # Plot bar chart of feature importances
    plt.figure(figsize=(10, 6))
    # plt.barh(sorted_features, sorted_importances, xerr=sorted_std_importances)
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Permutation Feature Importance with Cross-Validation')
    plt.gca().invert_yaxis()
    plt.show()

    if plotpath is not None:
        fn = plotpath + 'permutation_feature_importance'
        plt.savefig(fn, dpi=300, bbox_inches='tight')

    return sorted_features, sorted_importances, sorted_std_importances


def _calculate_shap_values(model, X, index, plotpath, show):
    if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
        X.columns = clean_feature_names(X.columns)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        # 另一个接口
        shap_values1 = explainer.shap_values(X)

        shap.summary_plot(shap_values1, X, show=show)
        plt.savefig(f'{plotpath}/summary_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()

        shap.summary_plot(shap_values1, X, plot_type="bar", show=show)
        plt.savefig(f'{plotpath}/summary_plot_bar.png', dpi=300, bbox_inches='tight')
        plt.clf()

        if index is not None:
            for i in index:
                shap.plots.waterfall(shap_values[i], show=show)
                plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
                plt.clf()

    elif isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        # 另一个接口
        shap_values1 = explainer.shap_values(X)
        # shap.plots.beeswarm(shap_values[:,:,1])
        shap.summary_plot(shap_values1[1], X, show=show)
        plt.savefig(f'{plotpath}/summary_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()

        shap.summary_plot(shap_values1[1], X, plot_type="bar", show=show)
        plt.savefig(f'{plotpath}/summary_plot_bar.png', dpi=300, bbox_inches='tight')
        plt.clf()

        if index is not None:
            for i in index:
                shap.plots.waterfall(shap_values[i, :, 1], show=show)
                plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
                plt.clf()

    elif isinstance(model, ExtraTreesClassifier) or isinstance(model, ExtraTreesRegressor):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        # 另一个接口
        shap_values1 = explainer.shap_values(X)
        # shap.plots.beeswarm(shap_values[:,:,1])
        shap.summary_plot(shap_values1[1], X, show=show)
        plt.savefig(f'{plotpath}/summary_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()

        shap.summary_plot(shap_values1[1], X, plot_type="bar", show=show)
        plt.savefig(f'{plotpath}/summary_plot_bar.png', dpi=300, bbox_inches='tight')
        plt.clf()

        if index is not None:
            for i in index:
                shap.plots.waterfall(shap_values[i, :, 1])
                plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
                plt.clf()

    elif isinstance(model, SVC) or isinstance(model, SVR):
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer(X)
        # 另一个接口
        shap_values1 = explainer.shap_values(X)
        shap.summary_plot(shap_values1, X)
        shap.summary_plot(shap_values1, X, plot_type="bar")
        if index is not None:
            for i in index:
                shap.plots.waterfall(shap_values[i])
                plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
                plt.clf()

    elif isinstance(model, GradientBoostingClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        # 另一个接口
        shap_values1 = explainer.shap_values(X)

        shap.summary_plot(shap_values1, X,  show=show)
        plt.savefig(f'{plotpath}/summary_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()

        shap.summary_plot(shap_values1, X, plot_type="bar", show=show)
        plt.savefig(f'{plotpath}/summary_plot_bar.png', dpi=300, bbox_inches='tight')
        plt.clf()

        if index is not None:
            for i in index:
                shap.plots.waterfall(shap_values[i], show=show)
                plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
                plt.clf()

    elif isinstance(model, LogisticRegression):
        masker = shap.maskers.Independent(data=X)
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        # 另一个接口
        shap_values1 = explainer.shap_values(X)

        shap.summary_plot(shap_values1, X, show=show)
        plt.savefig(f'{plotpath}/summary_plot.png', dpi=300, bbox_inches='tight')
        plt.clf()

        shap.summary_plot(shap_values1, X, plot_type="bar", show=show)
        plt.savefig(f'{plotpath}/summary_plot_bar.png', dpi=300, bbox_inches='tight')
        plt.clf()

        if index is not None:
            for i in index:
                shap.plots.waterfall(shap_values[i], show=show)
                plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
                plt.clf()

    else:
        explainer = shap.KernelExplainer(model.predict, X)
        shap_values = explainer(X)
        # 另一个接口
        shap_values1 = explainer.shap_values(X)
        shap.summary_plot(shap_values1, X, show=show)
        shap.summary_plot(shap_values1, X, plot_type="bar", show=show)
        if index is not None:
            for i in index:
                shap.plots.waterfall(shap_values[i], show=show)
                plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
                plt.clf()

    return shap_values


def _calculate_shap_value1(model, X, index):
    if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
        X.columns = clean_feature_names(X.columns)
        explainer = shap.TreeExplainer(model)
        shap_values1 = explainer.shap_values(X)
        expected_value = explainer.expected_value

    elif isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
        explainer = shap.TreeExplainer(model)
        shap_values1 = explainer.shap_values(X)
        expected_value = explainer.expected_value[1]

    elif isinstance(model, ExtraTreesClassifier) or isinstance(model, ExtraTreesRegressor):
        explainer = shap.TreeExplainer(model)
        shap_values1 = explainer.shap_values(X)
        expected_value = explainer.expected_value[1]

    elif isinstance(model, GradientBoostingClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values1 = explainer.shap_values(X)
        expected_value = explainer.expected_value

    elif isinstance(model, LogisticRegression):
        masker = shap.maskers.Independent(data=X)
        explainer = shap.Explainer(model, X)
        shap_values1 = explainer.shap_values(X)
        expected_value = explainer.expected_value

    return shap_values1, expected_value


def _calculate_shap_values_cv(model,
                              X,
                              y,
                              cv_f,
                              index,
                              plotpath='./'):
    ix_training, ix_test = [], []
    #print(cv_f)
    for fold in cv_f.split(X, y):
        ix_training.append(fold[0]), ix_test.append(fold[1])

    SHAP_values_per_fold, fold_ecv = [], []

    for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):

        X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
        y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

        params = model.get_params()
        model1 = return_model(model)
        model1.set_params(**params)

        model1.fit(X_train, y_train)

        shap_values, expected_value = _calculate_shap_value1(model1, X_test, index)

        m = check_model_name(model)
        if m in ['LogisticRegression', 'GBDT', 'XGBoost']:
            s1 = shap_values
        else:
            s1 = shap_values[1]

        if m == 'GBDT':
            s2 = float(expected_value)
        else:
            s2 = expected_value

        for SHAPs in s1:
            SHAP_values_per_fold.append(SHAPs)
            fold_ecv.append(s2)

    new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
    # fold_ecv = fold_ecv[new_index]

    X1 = X.reindex(X.index[new_index])
    shap.summary_plot(np.array(SHAP_values_per_fold), X1)
    plt.savefig(f'{plotpath}/summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()

    if index is not None:
        for oi in index:
        #for i in index:
            i = new_index.index(oi)
            #shap.force_plot(fold_ecv[i], SHAP_values_per_fold[i], X1.iloc[i, :])
            shap.plots._waterfall.waterfall_legacy(fold_ecv[i],
                                                   SHAP_values_per_fold[i],
                                                   feature_names=X1.columns,
                                                   show=False)
            plt.title(f'index{i}')
            plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.clf()

    return SHAP_values_per_fold, fold_ecv


def _calculate_shap_values_repcv(model,
                                 X,
                                 y,
                                 cv_f,
                                 index,
                                 plotpath='./'):
    n_splits = int(cv_f.get_n_splits() / cv_f.n_repeats)
    CV_repeats = cv_f.n_repeats

    random_states = np.random.randint(10000, size=CV_repeats)

    shap_values_per_cv = dict()
    ex_per_cv = dict()

    for sample in X.index:
        shap_values_per_cv[sample] = {}
        ex_per_cv[sample] = {}

        for CV_repeat in range(CV_repeats):
            shap_values_per_cv[sample][CV_repeat] = {}
            ex_per_cv[sample][CV_repeat] = {}

    for i, CV_repeat in enumerate(range(CV_repeats)):

        if isinstance(cv_f, RepeatedStratifiedKFold):
            CV = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_states[i])
        elif isinstance(cv_f, RepeatedKFold):
            CV = KFold(n_splits=n_splits, shuffle=True, random_state=random_states[i])

        ix_training, ix_test = [], []

        for fold in CV.split(X, y):
            ix_training.append(fold[0]), ix_test.append(fold[1])

        for i, (train_outer_ix, test_outer_ix) in enumerate(zip(ix_training, ix_test)):

            X_train, X_test = X.iloc[train_outer_ix, :], X.iloc[test_outer_ix, :]
            y_train, y_test = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

            params = model.get_params()
            model1 = return_model(model)
            model1.set_params(**params)

            model1.fit(X_train, y_train)

            shap_values, expected_value = _calculate_shap_value1(model1, X_test, index)

            m = check_model_name(model)
            if m in ['LogisticRegression', 'GBDT', 'XGBoost']:
                s1 = shap_values
            else:
                s1 = shap_values[1]

            if m == 'GBDT':
                s2 = float(expected_value)
            else:
                s2 = expected_value

            # Extract SHAP information per fold per sample 
            for i, test_index in enumerate(test_outer_ix):
                shap_values_per_cv[X.index[test_index]][CV_repeat] = s1[i]
                ex_per_cv[X.index[test_index]][CV_repeat] = s2

    average_shap_values, stds, ranges, aver_ex = [], [], [], []

    for i in range(0, len(X)):
        df_per_obs = pd.DataFrame.from_dict(shap_values_per_cv[X.index[i]])  # Get all SHAP values for sample number i
        # Get relevant statistics for every sample 
        average_shap_values.append(df_per_obs.mean(axis=1).values)
        stds.append(df_per_obs.std(axis=1).values)
        ranges.append(df_per_obs.max(axis=1).values - df_per_obs.min(axis=1).values)

        di = ex_per_cv[X.index[i]]
        mean_value = sum(di.values()) / len(di)
        aver_ex.append(mean_value)

    shap.summary_plot(np.array(average_shap_values), X, show=False)
    plt.title('Average SHAP values after cross-validation')
    plt.savefig(f'{plotpath}/summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.clf()
    #plt.title('Average SHAP values after cross-validation')
    X1 = X.applymap(lambda x: round(x, 2))

    if index is not None:
        for i in index:
            #shap.force_plot(aver_ex[i], np.array(average_shap_values)[i, :], X1.iloc[i, :], matplotlib=True)
            shap.plots._waterfall.waterfall_legacy(aver_ex[i],
                                                   np.array(average_shap_values)[i, :],
                                                   feature_names=X1.columns,
                                                   show=False)
            plt.title(f'index{i}')
            plt.savefig(f'{plotpath}/waterfall_plot_{i}.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.clf()
    return average_shap_values, aver_ex