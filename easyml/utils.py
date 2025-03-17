import os
import numpy as np
import pandas as pd
from sklearn.ensemble import *
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

'''
def encode_target(target, case, df):
  
    if target not in df.columns:
        raise ValueError(f"The target column '{target}' does not exist in the DataFrame.")

    df[target + '_encoded'] = df[target].apply(lambda x: 1 if x == case else 0)
    
    return df
'''


def clean_feature_names(feature_names):
    import re
    cleaned_names = []
    for name in feature_names:
        cleaned_name = re.sub(r'[\[\]<]', '', name)
        cleaned_names.append(cleaned_name)
    return cleaned_names


def lognorm(table, pseudocount):
    average = table.values.sum() / table.shape[0]
    table = table.div(table.sum(axis=1), axis=0)
    table = np.log10(table * average + pseudocount)
    return table


def CLR_normalize(pd_dataframe, pseudocount):
    """
    Centered Log Ratio
    Aitchison, J. (1982). 
    The statistical analysis of compositional data. 
    Journal of the Royal Statistical Society: 
    Series B (Methodological), 44(2), 139-160.
    """
    d = pd_dataframe
    d = d + pseudocount
    step1_1 = d.apply(np.log, 1)
    step1_2 = step1_1.apply(np.average, 1)
    step1_3 = step1_2.apply(np.exp)
    step2 = d.divide(step1_3, 0)
    step3 = step2.apply(np.log, 1)
    return (step3)


def check_model_name(model):
    if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
        return ('XGBoost')

    elif isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
        return ('RandomForest')

    elif isinstance(model, ExtraTreesClassifier) or isinstance(model, ExtraTreesRegressor):
        return ('ExtraTrees')

    elif isinstance(model, SVC) or isinstance(model, SVR):
        return ('SVM')

    elif isinstance(model, GradientBoostingClassifier):
        return ('GBDT')

    elif isinstance(model, LogisticRegression):
        return ('LogisticRegression')


def return_model(model):
    if isinstance(model, xgb.XGBClassifier) or isinstance(model, xgb.XGBRegressor):
        nm = xgb.XGBClassifier()

    elif isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
        nm = RandomForestClassifier()

    elif isinstance(model, ExtraTreesClassifier) or isinstance(model, ExtraTreesRegressor):
        nm = ExtraTreesClassifier()

    elif isinstance(model, GradientBoostingClassifier):
        nm = GradientBoostingClassifier()

    elif isinstance(model, LogisticRegression):
        nm = LogisticRegression()

    return (nm)


def find_matching_rows(oX, X, rows_oX):
    matching_rows = []
    for index in rows_oX:
        row_oX = oX.iloc[index]
        for i, row_X in X.iterrows():
            if row_oX.equals(row_X):
                matching_rows.append(i)
                break
    return matching_rows
