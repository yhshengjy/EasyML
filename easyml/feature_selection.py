#特征筛选
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, RFE, RFECV
from BorutaShap import BorutaShap
import pandas as pd

def feature_selection(X,
                      y,
                      method='RFECV',
                      task='classification',
                      n_features_to_select=None,
                      estimator=None,
                      step=1,
                      cv=None,
                      min_features_to_select=50,
                      importance_measure='shap',
                      n_trials=100,
                      n_jobs=-1,
                      random_state=1):
    """
    Perform feature selection using specified method.

    Parameters:
    - X: Feature matrix
    - y: Target variable
    - method: Method for feature selection ('Mutual_info', 'RFE', 'RFECV' and 'BorutaShap')
    - task: Type of task ('classification' or 'regression')
    - n_features_to_select: Number of features to select (only used for 'rfe')
    - estimator: Estimator for feature selection (required for 'rfe' and 'rfecv')
    - cv: int, cross-validation generator or an iterable (only used for 'rfecv')
    - step: number of features to remove at each iteration. default=1
    - min_features_to_select: The minimum number of features to be selected. default=5 (only used for 'rfecv')
    
    Returns:
    - Selected features
    
    Example usage:
    estimator = RandomForestClassifier()  
    selected_features = feature_selection(X_train,
                                          y_train,
                                          estimator=estimator
                                          method='rfe',
                                          task='classification',
                                          n_features_to_select=5,estimator=estimator)
    """
    
    if method == 'Mutual_info':
        if task == 'classification':
            scores = mutual_info_classif(X, y, random_state=random_state) 
        elif task == 'regression':
            scores = mutual_info_regression(X, y, random_state=random_state)
        else:
            raise ValueError("Invalid task type. Choose from 'classification' or 'regression'.")
        
        selected_features = X.columns[scores.argsort()[-n_features_to_select:][::-1]]
    
    elif method == 'RFE':
        if estimator is None:
            raise ValueError("Estimator is required for 'rfe' method.")
        
        if task == 'classification':
            selector = RFE(estimator, n_features_to_select=1, step=step)
        elif task == 'regression':
            selector = RFE(estimator, n_features_to_select=1, step=step)
        else:
            raise ValueError("Invalid task type. Choose from 'classification' or 'regression'.")
        
        selector = selector.fit(X, y)
        rfe_res = pd.Series(selector.feature_names_in_, index=selector.ranking_)
        rfe_res.sort_index(inplace=False)
        selected_features = rfe_res[:n_features_to_select].to_list()
        #selected_features = X.columns[selector.support_]

    elif method == 'RFECV':
        if estimator is None:
            raise ValueError("Estimator is required for 'rfecv' method.")
        
        if task == 'classification':
            selector = RFECV(estimator, cv=cv, min_features_to_select=min_features_to_select, step=step, n_jobs=n_jobs)
        elif task == 'regression':
            selector = RFECV(estimator, cv=cv, min_features_to_select=min_features_to_select, step=step, n_jobs=n_jobs)
        else:
            raise ValueError("Invalid task type. Choose from 'classification' or 'regression'.")
        
        selector = selector.fit(X, y)
        selected_features = X.columns[selector.support_]
        
    elif method == 'BorutaShap':
        if task == 'classification':
            selector = BorutaShap(estimator, importance_measure=importance_measure, classification=True)
        elif task == 'regression':
            selector = BorutaShap(estimator, importance_measure=importance_measure, classification=False)
        else:
            raise ValueError("Invalid task type. Choose from 'classification' or 'regression'.")
        selector.fit(X=X, y=y, n_trials=n_trials, random_state=random_state, verbose=True)
        selector.TentativeRoughFix()
        selector.plot(which_features='all')
        sf_df = selector.Subset()
        selected_features = sf_df.columns.to_list()

    else:
        raise ValueError("Invalid method. Choose from 'Mutual_info', 'RFE', 'RFECV' or 'BorutaShap'.")

    return selected_features