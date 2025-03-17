from sklearn.ensemble import *
from sklearn.linear_model import *
from xgboost import XGBClassifier
from hpsklearn import *
from hyperopt.pyll.base import scope


class Hyperparameters:

    def frange(start, stop, step):
        while start < stop:
            yield round(start, 10)
            start += step

    _param_grid_space = {
        "RandomForest": {
            "estimator": "RandomForestClassifier()",
            "param_grid": {
                "n_estimators": [*range(100, 1000, 100)],
                "max_depth": [*range(10, 30, 1)],
                "max_features": ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }
        },
        "ExtraTrees": {
            "estimator": "ExtraTreesClassifier()",
            "param_grid": {
                "n_estimators": [*range(100, 1000, 100)],
                "max_depth": [*range(10, 30, 1)],
                "max_features": ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }
        },
        "LogisticRegression": {
            "estimator": "LogisticRegression()",
            "param_grid": {
                "penalty": ["l1", "l2"],
                "C": [*map(lambda x: round(x, 2), [*frange(0.01, 1, 0.01)])]
            }
        },
        "XGBoost": {
            "estimator": "XGBClassifier()",
            "param_grid": {
                "n_estimators": [*range(100, 800, 50)],
                "learning_rate": [*map(lambda x: round(x, 2), [*frange(0.01, 0.31, 0.02)])],
                "max_depth": [*range(2, 15, 1)],
                "subsample": [*map(lambda x: round(x, 2), [*frange(0.5, 1, 0.1)])]
            }
        },
        "GradientBoosting": {
            "estimator": "GradientBoostingClassifier()",
            "param_grid": {
                "n_estimators": [*range(100, 800, 100)],
                "learning_rate": [*map(lambda x: round(x, 2), [*frange(0.01, 0.31, 0.01)])],
                "max_features": ["log2", "sqrt", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "max_depth": [*range(5, 15, 1)],
                "subsample": [*map(lambda x: round(x, 2), [*frange(0.5, 1, 0.1)])]
            }
        }
    }

    _param_reg_grid_space = {
        "RandomForest": {
            "estimator": "RandomForestRegressor()",
            "param_grid": {
                "n_estimators": [*range(100, 1000, 100)],
                "max_depth": [*range(10, 30, 1)],
                "max_features": ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }
        },
        "ExtraTrees": {
            "estimator": "ExtraTreesRegressor()",
            "param_grid": {
                "n_estimators": [*range(100, 1000, 100)],
                "max_depth": [*range(10, 30, 1)],
                "max_features": ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }
        },
        "Ridge": {
            "estimator": "Ridge()",
            "param_grid": {
                "alpha": [*map(lambda x: round(x, 2), [*frange(0.01, 1, 0.01)])]
            }
        },
        "Lasso": {
            "estimator": "Lasso()",
            "param_grid": {
                "alpha": [*map(lambda x: round(x, 2), [*frange(0.01, 1, 0.01)])]
            }
        },
        "XGBoost": {
            "estimator": "XGBRegressor()",
            "param_grid": {
                "n_estimators": [*range(100, 800, 50)],
                "learning_rate": [*map(lambda x: round(x, 2), [*frange(0.01, 0.2, 0.02)])],
                "max_depth": [*range(2, 10, 1)],
                "subsample": [*map(lambda x: round(x, 2), [*frange(0.7, 1, 0.05)])]
            }
        },
        "GradientBoosting": {
            "estimator": "GradientBoostingRegressor()",
            "param_grid": {
                "n_estimators": [*range(100, 800, 100)],
                "learning_rate": [*map(lambda x: round(x, 2), [*frange(0.01, 0.2, 0.01)])],
                "max_features": [None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "max_depth": [*range(5, 15, 1)],
                "subsample": [*map(lambda x: round(x, 2), [*frange(0.5, 1, 0.1)])]
            }
        }
    }

    _param_hpopt_space = {
        'RandomForestClassifier':
            {'n_estimators': hp.choice("n_estimators", [*range(100, 1000, 100)]),
             'max_depth': hp.choice("max_depth", [*range(10, 30, 1)]),
             "max_features": hp.choice("max_features", ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
             "min_impurity_decrease": 0,  # d
             "criterion": "gini",  # d
             "min_samples_leaf": 1,  # d
             "min_samples_split": 2,  # d
             'min_weight_fraction_leaf': 0.0,  # d
             'max_leaf_nodes': None # d
             },
        "RandomForestRegressor": {
            'n_estimators': hp.choice("n_estimators", [*range(100, 1000, 100)]),
            'max_depth': hp.choice("max_depth", [*range(10, 30, 1)]),
            "max_features": hp.choice("max_features",
                                      ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            "min_impurity_decrease": 0,  # d
            "criterion": "squared_error",  # d
            "min_samples_leaf": 1,  # d
            "min_samples_split": 2,  # d
            'min_weight_fraction_leaf': 0.0,  # d
            'max_leaf_nodes': None  # d
        },
        'ExtraTreesClassifier':
            {'n_estimators': hp.choice("n_estimators", [*range(100, 1000, 100)]),
             'max_depth': hp.choice("max_depth", [*range(10, 30, 1)]),
             "max_features": hp.choice("max_features", ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
             "min_impurity_decrease": 0,  # d
             "criterion": "gini",  # d
             "min_samples_leaf": 1,  # d
             "min_samples_split": 2,  # d
             'min_weight_fraction_leaf': 0.0,  # d
             'max_leaf_nodes': None # d
             },
        'ExtraTreesRegressor':
            {'n_estimators': hp.choice("n_estimators", [*range(100, 1000, 100)]),
             'max_depth': hp.choice("max_depth", [*range(10, 30, 1)]),
             "max_features": hp.choice("max_features", ["sqrt", "log2", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
             "min_impurity_decrease": 0,  # d
             "criterion": "squared_error",  # d
             "min_samples_leaf": 1,  # d
             "min_samples_split": 2,  # d
             'min_weight_fraction_leaf': 0.0,  # d
             'max_leaf_nodes': None  # d
             },
        "LogisticRegression": {
            "penalty": hp.choice("penalty", ["l1", "l2"]),
             "C": hp.quniform("C", 0.01, 1, 0.01),
             'tol': 0.0001  # d
             #"class_weight": hp.choice("class_weight", ["balanced", None])
             },
        "Ridge": {
            "alpha": hp.quniform("alpha", 0.01, 1, 0.01),
            "tol": 0.0001
        },
        "Lasso": {
            "alpha": hp.quniform("alpha", 0.01, 1, 0.01),  # Lasso 回归的正则化强度
            "tol": 0.0001
        },
        "XGBRegressor": {
            "n_estimators": hp.choice("n_estimators", [*range(100, 800, 50)]),
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.2, 0.02),  # 限制在 0.01 - 0.2 之间
            "max_depth": hp.choice("max_depth", [*range(2, 10, 1)]),  # 降低最大深度，减少过拟合
            "subsample": hp.quniform("subsample", 0.7, 1, 0.05),
            'colsample_bylevel': 1,
            'colsample_bytree': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            "min_child_weight": 1,
        },
        # gamma, min_child_weight, n_jobs, random_state
        "XGBClassifier": {
            "n_estimators": hp.choice("n_estimators", [*range(100, 800, 50)]),
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.31, 0.02),
            "max_depth": hp.choice("max_depth", [*range(2, 15, 1)]),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.1),
            'colsample_bylevel': 1,
            'colsample_bytree': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            "min_child_weight": 1,
        },
        "GradientBoostingRegressor": {
            "n_estimators": hp.choice("n_estimators", [*range(100, 800, 100)]),
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.2, 0.01),
            "max_features": hp.choice("max_features", [None, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            "max_depth": hp.choice("max_depth", [*range(5, 15, 1)]),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.1),
            'min_samples_split': 2,  # d
            'max_leaf_nodes': None,  # d
            'min_impurity_decrease': 0,  # d
            'min_samples_leaf': 1,  # d
            'criterion': 'squared_error',  # d
        },
        "GradientBoostingClassifier": {
            "n_estimators": hp.choice("n_estimators", [*range(100, 800, 100)]),
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.31, 0.01),
            "max_features": hp.choice("max_features", ["log2", "sqrt", None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            "max_depth": hp.choice("max_depth", [*range(5, 15, 1)]),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.1),
            #"loss": hp.choice("loss", ['log_loss', 'exponential']), #hyperopt-sklearn中已定义
            'min_samples_split': 2,  # d
            'max_leaf_nodes': None,  # d
            'min_impurity_decrease': 0,  # d
            'min_samples_leaf': 1,  # d
            'criterion': 'friedman_mse',  # d
        }
    }

    @classmethod
    def get_space(cls, space_type, model_name):
        if space_type == "cls_grid":
            return cls._param_grid_space.get(model_name, {})
        elif space_type == "reg_grid":
            return cls._param_reg_grid_space.get(model_name, {})
        elif space_type == "hyperopt":
            return cls._param_hpopt_space.get(model_name, {})

    @classmethod
    def set_space(cls, space_type, model_name, new_space):
        if space_type == "cls_grid":
            cls._param_grid_space[model_name] = new_space
        elif space_type == "reg_grid":
            cls._param_reg_grid_space[model_name] = new_space
        elif space_type == "hyperopt":
            cls._param_hpopt_space[model_name] = new_space

    @classmethod
    def add_param(cls, space_type, model_name, param_name, values):
        space = cls.get_space(space_type, model_name)
        space["param_grid"][param_name] = values
        cls.set_space(space_type, model_name, space)

    @classmethod
    def remove_param(cls, space_type, model_name, param_name):
        space = cls.get_space(space_type, model_name)
        if param_name in space["param_grid"]:
            del space["param_grid"][param_name]
            cls.set_space(space_type, model_name, space)

    @classmethod
    def random_classier_ht(name='nyh', params=_param_hpopt_space):
        n_estimators = params['RandomForestClassifier']['n_estimators']
        max_depth = params['RandomForestClassifier']['max_depth']
        max_features = params['RandomForestClassifier']['max_features']

        min_impurity_decrease = params['RandomForestClassifier']['min_impurity_decrease']
        criterion = params['RandomForestClassifier']['criterion']
        min_samples_leaf = params['RandomForestClassifier']['min_samples_leaf']
        min_samples_split = params['RandomForestClassifier']['min_samples_split']
        min_weight_fraction_leaf = params['RandomForestClassifier']['min_weight_fraction_leaf']
        max_leaf_nodes = params['RandomForestClassifier']['max_leaf_nodes']

        n_jobs = -1

        return random_forest_classifier(
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            n_jobs=n_jobs
        )

    @classmethod
    def random_regressor_ht(name='nyh', params=_param_hpopt_space):
        n_estimators = params['RandomForestRegressor']['n_estimators']
        max_depth = params['RandomForestRegressor']['max_depth']
        max_features = params['RandomForestRegressor']['max_features']

        min_impurity_decrease = params['RandomForestRegressor']['min_impurity_decrease']
        criterion = params['RandomForestRegressor']['criterion']
        min_samples_leaf = params['RandomForestRegressor']['min_samples_leaf']
        min_samples_split = params['RandomForestRegressor']['min_samples_split']
        min_weight_fraction_leaf = params['RandomForestRegressor']['min_weight_fraction_leaf']
        max_leaf_nodes = params['RandomForestRegressor']['max_leaf_nodes']

        n_jobs = -1

        return random_forest_regressor(
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            n_jobs=n_jobs
        )

    @classmethod
    def extratrees_classifier_ht(name='nyh', params=_param_hpopt_space):
        n_estimators = params['RandomForestClassifier']['n_estimators']
        max_depth = params['RandomForestClassifier']['max_depth']
        max_features = params['RandomForestClassifier']['max_features']

        min_impurity_decrease = params['RandomForestClassifier']['min_impurity_decrease']
        criterion = params['RandomForestClassifier']['criterion']
        min_samples_leaf = params['RandomForestClassifier']['min_samples_leaf']
        min_samples_split = params['RandomForestClassifier']['min_samples_split']
        min_weight_fraction_leaf = params['RandomForestClassifier']['min_weight_fraction_leaf']
        max_leaf_nodes = params['RandomForestClassifier']['max_leaf_nodes']

        n_jobs = -1

        return extra_trees_classifier(
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            n_jobs=n_jobs
        )

    @classmethod
    def extratrees_regressor_ht(name='nyh', params=_param_hpopt_space):
        n_estimators = params['RandomForestClassifier']['n_estimators']
        max_depth = params['RandomForestClassifier']['max_depth']
        max_features = params['RandomForestClassifier']['max_features']

        min_impurity_decrease = params['RandomForestClassifier']['min_impurity_decrease']
        criterion = params['RandomForestClassifier']['criterion']
        min_samples_leaf = params['RandomForestClassifier']['min_samples_leaf']
        min_samples_split = params['RandomForestClassifier']['min_samples_split']
        min_weight_fraction_leaf = params['RandomForestClassifier']['min_weight_fraction_leaf']
        max_leaf_nodes = params['RandomForestClassifier']['max_leaf_nodes']

        n_jobs = -1

        return extra_trees_regressor(
            name=name,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            criterion=criterion,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_leaf_nodes=max_leaf_nodes,
            n_jobs=n_jobs
        )

    @classmethod
    def lasso_ht(name='nyh', params=_param_hpopt_space):
        alpha = params['lasso']['alpha']
        return lasso(name, alpha=alpha)

    @classmethod
    def ridge_ht(name='nyh', params=_param_hpopt_space):
        alpha = params['ridge']['alpha']
        return ridge(name, alpha=alpha)

    @classmethod
    def logreg_ht(name='nyh', params=_param_hpopt_space):
        penalty = params['LogisticRegression']['penalty']
        C = params['LogisticRegression']['C']
        tol = params['LogisticRegression']['tol']
        solver = 'liblinear'
        return logistic_regression(name,
                                   penalty=penalty,
                                   C=C,
                                   tol=tol,
                                   solver=solver)

    @classmethod
    def gbc_ht(name='nyh', init=None, params=_param_hpopt_space):
        n_estimators = params['GradientBoostingClassifier']['n_estimators']
        learning_rate = params['GradientBoostingClassifier']['learning_rate']
        init = init
        max_features = params['GradientBoostingClassifier']['max_features']
        subsample = params['GradientBoostingClassifier']['subsample']
        #loss = params['GradientBoostingClassifier']['loss']
        max_depth = params['GradientBoostingClassifier']['max_depth']
        min_samples_split = params['GradientBoostingClassifier']['min_samples_split']
        max_leaf_nodes = params['GradientBoostingClassifier']['max_leaf_nodes']
        min_impurity_decrease = params['GradientBoostingClassifier']['min_impurity_decrease']
        min_samples_leaf = params['GradientBoostingClassifier']['min_samples_leaf']
        criterion = params['GradientBoostingClassifier']['criterion']

        return gradient_boosting_classifier(
            name=name,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_features=max_features,
            subsample=subsample,
            #loss=loss,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion
        )

    @classmethod
    #huigui
    def gbr_ht(name='nyh', init=None, params=_param_hpopt_space):
        n_estimators = params['GradientBoostingRegressor']['n_estimators']
        learning_rate = params['GradientBoostingRegressor']['learning_rate']
        init = init
        max_features = params['GradientBoostingRegressor']['max_features']
        subsample = params['GradientBoostingRegressor']['subsample']
        max_depth = params['GradientBoostingRegressor']['max_depth']
        min_samples_split = params['GradientBoostingRegressor']['min_samples_split']
        max_leaf_nodes = params['GradientBoostingRegressor']['max_leaf_nodes']
        min_impurity_decrease = params['GradientBoostingRegressor']['min_impurity_decrease']
        min_samples_leaf = params['GradientBoostingRegressor']['min_samples_leaf']
        criterion = params['GradientBoostingRegressor']['criterion']

        return gradient_boosting_regressor(
            name=name,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_features=max_features,
            subsample=subsample,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion
        )

    @classmethod
    def xgbc_ht(self, name='nyh'):
        params = self._param_hpopt_space
        n_estimators = params['XGBClassifier']['n_estimators']
        learning_rate = params['XGBClassifier']['learning_rate']
        subsample = params['XGBClassifier']['subsample']
        gamma = params['XGBClassifier']['gamma']
        reg_alpha = params['XGBClassifier']['reg_alpha']
        reg_lambda = params['XGBClassifier']['reg_lambda']
        colsample_bytree = params['XGBClassifier']['colsample_bytree']
        colsample_bylevel = params['XGBClassifier']['colsample_bylevel']
        max_depth = params['XGBClassifier']['max_depth']
        min_child_weight = params['XGBClassifier']['min_child_weight']
        return xgboost_classification(name,
                                      n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      subsample=subsample,
                                      gamma=gamma,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      colsample_bytree=colsample_bytree,
                                      colsample_bylevel=colsample_bylevel,
                                      max_depth=max_depth,
                                      min_child_weight=min_child_weight)

    @classmethod
    #huigui
    def xgbr_ht(self, name='nyh'):
        params = self._param_hpopt_space
        n_estimators = params['XGBRegressor']['n_estimators']
        learning_rate = params['XGBRegressor']['learning_rate']
        subsample = params['XGBRegressor']['subsample']
        gamma = params['XGBRegressor']['gamma']
        reg_alpha = params['XGBRegressor']['reg_alpha']
        reg_lambda = params['XGBRegressor']['reg_lambda']
        colsample_bytree = params['XGBRegressor']['colsample_bytree']
        colsample_bylevel = params['XGBRegressor']['colsample_bylevel']
        max_depth = params['XGBRegressor']['max_depth']
        min_child_weight = params['XGBRegressor']['min_child_weight']
        return xgboost_regression(name,
                                      n_estimators=n_estimators,
                                      learning_rate=learning_rate,
                                      subsample=subsample,
                                      gamma=gamma,
                                      reg_alpha=reg_alpha,
                                      reg_lambda=reg_lambda,
                                      colsample_bytree=colsample_bytree,
                                      colsample_bylevel=colsample_bylevel,
                                      max_depth=max_depth,
                                      min_child_weight=min_child_weight)

    @classmethod
    def all_clf(name='nyh'):
        return any_classifier(name)
