import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
import pickle

#from easyml.evaluate_model import _specificity_score
from .feature_selection import *
from .train_model import *
from .evaluate_model import *
from .model_interpretation import *
from .utils import *


class easyML:
    def __init__(self,
                 filename,
                 case,
                 target,
                 split=True,
                 test_size=0.3,
                 task='classification',
                 transformation_method=None,
                 balance_method=None,
                 pseudocount=0.00001,
                 sampling_strategy='auto',
                 stratify=False,
                 group=None,
                 scaling_method=None,
                 var_filter=0,
                 cor_filter=None,
                 n_jobs=-1):
        """
              Initializes the easyML class for automation machine learning.

              Parameters:
              ----------
              filename : str
                  The path to the dataset file to be loaded.

              case : str
                  A descriptive name or identifier for the case or project.

              target : str
                  The name of the target variable in the dataset.

              split : bool, default=True
                  If True, the dataset will be split into training and testing sets.

              test_size : float, default=0.3
                  The proportion of the dataset to include in the test split. Ignored if split is False.

              task : str, default='classification'
                  The type of machine learning task. Supported tasks are:
                  - 'classification'
                  - 'regression'

              transformation_method : str or None, default=None
                  The method to apply for data transformation. Supported methods include:
                  - 'CLR'
                  - 'Relative_Abundance'
                  - 'Hellinger'
                  - 'Lognorm'

                  Detailed information can be found in the article:
                  Proportion-based normalizations outperform compositional data transformations
                  in machine learning applications Microbiome (2024) 12:45.

              balance_method : str or None, default=None
                  The method to use for balancing the classes in the target variable. Supported methods include:
                  - 'Under_Sampling'
                  - 'Over_Sampling'
                  - 'SMOTE'
                  - 'ADASYN'
                  - 'ENN'

                  Refer to the imbalanced-learn package for more details.

              pseudocount : float, default=0.00001
                  A small value to add to avoid division by zero or log of zero in data transformation.

              sampling_strategy : str, default='auto'
                  The sampling strategy to use for balancing. Relevant only if balance_method is specified.

              stratify : bool, default=False
                  If True, the data split will be stratified according to the target variable.

              scaling_method : str or None, default=None
                  The method to use for scaling features. Supported methods include:
                  - 'standard': Standard Scaling
                  - 'minmax': Min-Max Scaling

              n_jobs : int, default=-1
                  The number of CPU cores to use for computations. -1 means using all processors.

              """

        self.loader = DataLoader(filename, target)

        if isinstance(filename, str):
            self.data = self.loader.load_data()
        elif isinstance(filename, pd.DataFrame):
            self.data = filename
        else:
            raise ValueError("Invalid input. It should be either a filenames or DataFrame.")

        #data去掉group
        if group is not None:
            self.group = self.data[group]
            self.data = self.data.drop(columns=[group])
        else:
            self.group = None

        self.target = target
        self.case = case

        self.split = split
        self.test_size = test_size

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_fs = None
        self.X_test_fs = None
        # self.model_names=model_names
        # self.scoring=scoring
        # self.search_methods=search_methods
        # self.n_iter=n_iter
        self.cv_f = None
        self.holdout_X = None
        self.holdout_y = None
        # self.fold = None
        self.task = task
        self.transformation_method = transformation_method
        self.balance_method = balance_method
        self.pseudocount = pseudocount
        self.sampling_strategy = sampling_strategy
        self.stratify = stratify
        self.scaling_method = scaling_method
        self.o_Xtrain = None
        self.o_ytrain = None

        # self.trained_model=None
        self.selected_features = None
        self.final_model = None
        self.model_performance = None
        self.model_fs_performance = None

        self.varf = var_filter
        self.corf = cor_filter

        # self.random_state=random_state
        self.n_jobs = n_jobs

    def get_data(self):
        return self.data

    def preprocess_data(self, random_state=None):
        """
            Preprocess the data for machine learning.

            This method applies various preprocessing steps to the data,
            including transformations, balancing methods, dataset splitting, and feature scaling.

            Parameters:
            ----------
            random_state : int or None, default=None

        """

        preprocessor = EncodeSplit(self.data, self.target, self.task, self.case, random_state)
        # if self.task=='classification':
        preprocessor.encode_target()

        preprocessor.split_data(split=self.split,
                                test_size=self.test_size,
                                stratify=self.stratify,
                                varf=self.varf,
                                corf=self.corf
                                )

        r_train = preprocessor.X_train.index.to_list()
        c_train = preprocessor.X_train.columns.to_list()
        if self.split:
            r_test = preprocessor.X_test.index.to_list()
            c_test = preprocessor.X_test.columns.to_list()

        if self.balance_method is not None:
            self.o_Xtrain = self.X_train
            self.o_ytrain = self.y_train

        preprocessor.preprocess(split=self.split,
                                transformation_method=self.transformation_method,
                                balance_method=self.balance_method,
                                pseudocount=self.pseudocount,
                                sampling_strategy=self.sampling_strategy,
                                scaling_method=self.scaling_method
                                )

        if self.scaling_method is not None and self.balance_method is None:
            preprocessor.X_train = pd.DataFrame(preprocessor.X_train,
                                                index=r_train,
                                                columns=c_train)
            if self.split:
                preprocessor.X_test = pd.DataFrame(preprocessor.X_test,
                                                   index=r_test,
                                                   columns=c_test)
        if self.scaling_method is not None and self.balance_method is not None:
            preprocessor.X_train = pd.DataFrame(preprocessor.X_train,
                                                # index=r_train,
                                                columns=c_train)
            if self.split:
                preprocessor.X_test = pd.DataFrame(preprocessor.X_test,
                                                   index=r_test,
                                                   columns=c_test)
        self.X_train, self.X_test, self.y_train, self.y_test = preprocessor.X_train, \
                                                               preprocessor.X_test, \
                                                               preprocessor.y_train, \
                                                               preprocessor.y_test

    def feature_selection(self,
                          model=None,
                          method='RFECV',
                          # task='classification',
                          n_features_to_select=50,
                          step=5,
                          cv=None,
                          fold=5,
                          repeat=5,
                          min_features_to_select=10,
                          importance_measure='shap',
                          n_trials=100,
                          random_state=None):
        """
          Perform feature selection on the dataset.

          This method applies various feature selection techniques to identify the most relevant features for the model.

          Parameters:
          ----------
          method : str, default='RFECV'
              The feature selection method to use. Supported methods include:
              - 'Mutual_info': Mutual Information.
              - 'RFE': Recursive Feature Elimination.
              - 'RFECV': Recursive Feature Elimination with Cross-Validation.
              - 'BorutaShap': using SHAP Values as the feature selection method in Boruta.

          task : str, default='classification'
              The type of machine learning task. Supported tasks are:
              - 'classification'
              - 'regression'

          n_features_to_select : int, default=50
              Number of features to select, used for 'Mutual_info' and 'RFE'.

          step : int, default=5
              The number of features to remove at each iteration. Used only for 'RFE' and 'RFECV'.

          cv : str, default='None'
               Cross-validation method. If `self.cv_f` is not None, it will be used instead.
               Supported cross-validation methods include:
                - 'KFold'
                - 'StratifiedKFold'
                - 'RepeatedKFold'
                - 'RepeatedStratifiedKFold'

          fold : int, default=5
              Number of folds for cross-validation.

          repeat : int, default=5
              Number of times cross-validation is repeated.

          min_features_to_select : int, default=10
              The minimum number of features to select. Used only for 'RFECV'.

          importance_measure : str, default='shap'
              The importance measure to use for feature selection. Relevant only for 'BorutaShap'.

          n_trials : int, default=100
              The number of trials for feature selection. Relevant only for 'BorutaShap'.

          random_state : int or None, default=None.

          Returns:
          -------
          selected_features : list
              A list of the selected feature names.

          """

        if self.cv_f is not None:
            cvf = self.cv_f
        elif cv is not None:
            cvf = _create_cv_object(cv=cv, fold=fold, repeat=repeat, random_state=random_state)
        if self.group is not None:
            cvf = cvf.split(self.X_train, self.y_train, self.group)

        if model is not None:
            estimator = model
        else:
            estimator = self.final_model

        self.selected_features = feature_selection(X=self.X_train,
                                                   y=self.y_train,
                                                   method=method,
                                                   task=self.task,
                                                   n_features_to_select=n_features_to_select,
                                                   estimator=estimator,
                                                   step=step,
                                                   cv=cvf,
                                                   min_features_to_select=min_features_to_select,
                                                   importance_measure=importance_measure,
                                                   n_trials=n_trials,
                                                   n_jobs=self.n_jobs,
                                                   random_state=random_state)

        # if method in ['Var', 'Corr', 'Mutual_info']:
        #    self.X_train_fs = self.X_train[self.selected_features]
        # elif method in ['RFE', 'RFECV', 'BorutaShap']:
        self.X_train_fs = self.X_train[self.selected_features]
        if self.split:
            self.X_test_fs = self.X_test[self.selected_features]

    def train_model(self,
                    model_names=['RandomForest'],
                    scoring='roc_auc',
                    n_iter=10,
                    search_methods='bayesian',
                    cv='StratifiedKFold',
                    fold=5,
                    repeat=5,
                    use_selected_features=False,
                    random_state=None):
        """
            Train a machine learning model.

            Parameters:
            ----------
            model_names : list, default=['RandomForest']
                Supported model names include:
                    - 'RandomForest': Random Forest classifier.
                    - 'ExtraTrees': Extra Trees classifier.
                    - 'LogisticRegression':Logistic Regression model.
                    - 'GradientBoosting': Gradient Boosting classifier.
                    - 'XGBoost': XGBoost classifier.

            scoring : str, default='roc_auc'
                Metric for evaluating model performance, include:
                - 'accuracy'
                - 'precision'
                - 'recall'
                - 'f1'
                - 'roc_auc'
                - mcc

            n_iter : int, default=10
                Number of iterations for random search and hyperparameter tuning.

            search_methods : str, default='bayesian'
                Hyperparameter search method. Supported methods include:
                - 'grid' (Grid Search)
                - 'random' (Random Search)
                - 'bayesian' (Bayesian Optimization)

            cv : str, default='StratifiedKFold'
                Cross-validation method. Supported cross-validation methods include:
                - 'KFold'
                - 'StratifiedKFold'
                - 'StratifiedGroupKFold'
                - 'RepeatedKFold'
                - 'RepeatedStratifiedKFold'

            fold : int, default=5
                Number of folds for cross-validation.

            use_selected_features : bool, default=False
                Whether to use selected features for training. If True, only the selected features will be used.

            random_state : int, default=None
                Random seed for reproducibility.

            Returns:
            ----------
            trained_models : dict
                Returns a dictionary containing the best models. The keys are the model names, and the values are the trained model objects.

            """

        if use_selected_features:
            if 'XGBoost' in model_names:
                self.X_train_fs.columns = clean_feature_names(self.X_train_fs.columns)
                if self.split:
                    self.X_test_fs.columns = clean_feature_names(self.X_test_fs.columns)
            X_train = self.X_train_fs
            y_train = self.y_train
        else:
            if 'XGBoost' in model_names:
                self.X_train.columns = clean_feature_names(self.X_train.columns)
                if self.split:
                    self.X_test.columns = clean_feature_names(self.X_test.columns)
            X_train = self.X_train
            y_train = self.y_train

        self.cv_f = _create_cv_object(cv=cv, fold=fold, repeat=repeat, random_state=random_state)
        #if self.group is not None:
        #    self.cv_f = self.cv_f.split(X_train, y_train, self.group)

        if self.task == 'classification':
            trained_model = train_classification_model(X_train=X_train,
                                                       y_train=y_train,
                                                       # X_test=self.X_test,
                                                       # y_test=self.y_test,
                                                       model_names=model_names,
                                                       group=self.group,
                                                       scoring=scoring,
                                                       n_iter=n_iter,
                                                       search_methods=search_methods,
                                                       cv=self.cv_f,
                                                       random_state=random_state,
                                                       n_jobs=self.n_jobs)

        if self.task == 'regression':
            trained_model = train_regression_model(X_train=X_train,
                                                   y_train=y_train,
                                                   model_names=model_names,
                                                   scoring=scoring,
                                                   n_iter=n_iter,
                                                   search_methods=search_methods,
                                                   cv=self.cv_f,
                                                   random_state=random_state,
                                                   n_jobs=self.n_jobs)

        if use_selected_features:
            self.trained_fs_model = trained_model
        else:
            self.trained_model = trained_model

    def evaluate_model(self,
                       holdout=None,
                       target_column=None,
                       model=None,
                       scoring=['roc_auc'],
                       cv=None,
                       fold=5,
                       repeat=5,
                       plot_model=True,
                       plot_savedir='./',
                       use_selected_features=False,
                       random_state=None):
        """
        Evaluate a machine learning model using specified metrics and cross-validation.

        Parameters:
        ----------
        holdout : pandas.DataFrame or None, default=None
            Holdout dataset to evaluate the model on.
            If None, the model is evaluated using train_test_split to generate a test set or using cross-validation.

        model : object or None, default=None
            The trained machine learning model to be evaluated. If None, the default model from self is used.

        scoring : list of str, default=['roc_auc']
            List of scoring metrics to evaluate the model. Common metrics include:
            - 'accuracy'
            - 'precision'
            - 'recall'
            - 'f1'
            - 'roc_auc'
            - Other metrics supported by sklearn

        cv : str or None, default=None
            Cross-validation method to be used. If None, the method specified in self is used. Supported methods include:
            - 'KFold'
            - 'StratifiedKFold'
            - 'StratifiedGroupKFold'
            - 'GroupShuffleSplit'

        fold : int, default=5
            Number of folds for cross-validation. If None, the number of folds specified in self is used.

        plot_model : bool, default=False
            Whether to plot ROC and Precision-Recall curves for the model.

        use_selected_features : bool, default=False
            Whether to use selected features for evaluation. If True, only the selected features will be used.

        plot_savedir : str or None, default='./'
           Directory to save the plots.

        random_state : int or None, default=None
            Random seed for reproducibility.

        Returns:
        ----------
        evaluation_results : dict
            A dictionary containing the evaluation results for the specified metrics.

        """
        from IPython.display import display

        if holdout == 'test':
            if self.split:
                '''
                if cv is not None:
                    if not use_selected_features:
                        X = self.X_train
                        y = self.y_train
                    else:
                        X = self.X_train_fs
                        y = self.y_train
                else:
                '''
                if not use_selected_features:
                    X = self.X_test
                    y = self.y_test
                else:
                    X = self.X_test_fs
                    y = self.y_test
                cve = None
                modele = self.final_model
            else:
                raise ValueError("Test dataset not available because 'split' is False.")
        elif holdout is None:
            if cv is not None:
                cve = _create_cv_object(cv=cv, fold=fold, repeat=repeat, random_state=random_state)
                if not use_selected_features:
                    X = self.X_train
                    y = self.y_train
                else:
                    X = self.X_train_fs
                    y = self.y_train
            else:
                cve = self.cv_f
                if not use_selected_features:
                    X = self.X_train
                    y = self.y_train
                else:
                    X = self.X_train_fs
                    y = self.y_train
            if use_selected_features:
                modele = self.trained_fs_model
            else:
                modele = self.trained_model
        else:
            if isinstance(holdout, pd.DataFrame):
                holdout_df = holdout
            elif isinstance(holdout, str):
                holdout_df = pd.read_csv(holdout, index_col=0)

            ho = EncodeSplit(holdout_df,
                             target_column=target_column,
                             task=self.task,
                             case=self.case)
            ho.encode_target()
            ho.split_data(split=False)
            ho.preprocess(split=False,
                          transformation_method=self.transformation_method,
                          pseudocount=self.pseudocount,
                          scaling_method=self.scaling_method
                          )
            X = ho.X_train
            y = ho.y_train
            cve = None
            modele = self.final_model

        if model is not None:
            modele = model

        if not use_selected_features:
            self.model_performance = evaluate_model(model=modele,
                                                    X_test=X,
                                                    y_test=y,
                                                    scoring=scoring,
                                                    cv=cve,
                                                    group=self.group,
                                                    task=self.task)
            mp = pd.DataFrame.from_dict(self.model_performance, orient='index')
            display(mp)
        else:
            self.model_fs_performance = evaluate_model(model=modele,
                                                       X_test=X,
                                                       y_test=y,
                                                       scoring=scoring,
                                                       cv=cve,
                                                       group=self.group,
                                                       task=self.task)
            mp = pd.DataFrame.from_dict(self.model_fs_performance, orient='index')
            display(mp)

        if plot_model:
            if self.task == 'classification':
                plot_roc_pr_curves(modele, X, y, cv=cve, plot_savedir=plot_savedir, random_state=random_state)
            # if self.task == 'regression':
            #    _plot_regression_results(model_dict, X_test, y_test, metrics)

    def select_model(self, model_name, use_selected_features=False):
        """
        Select a trained machine learning model as the final model from multiple trained models.

        Parameters:
        ----------
        model_name : str
            The name of the model to select from the trained models. Supported model names include:
            - 'RandomForest'
            - 'ExtraTrees'
            - 'LogisticRegression'
            - 'GradientBoosting'
            - 'XGBoost'
            - 'Lasso'
            - 'Ridge'

        use_selected_features : bool, default=False
            Whether to use the model trained with selected features.

        Raises:
        ------
        ValueError
            If the model_name is not found in trained models.
        """

        supported_models = {
            "RandomForest", "ExtraTrees", "LogisticRegression",
            "GradientBoosting", "XGBoost", "Lasso", "Ridge"
        }

        if model_name not in supported_models:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {supported_models}")

        if use_selected_features:
            if model_name not in self.trained_fs_model:
                raise ValueError(f"Model '{model_name}' is not available in trained feature-selected models.")
            self.final_model = self.trained_fs_model[model_name]['model']
        else:
            if model_name not in self.trained_model:
                raise ValueError(f"Model '{model_name}' is not available in trained models.")
            self.final_model = self.trained_model[model_name]['model']

    def auto_select_model(self, metric='roc_auc', use_selected_features=False):
        """
            Select the best machine learning model based on a specified evaluation metric.

            Parameters:
            ----------
            metric : str
                The evaluation metric to use for selecting the best model, include:
                - 'accuracy'
                - 'precision'
                - 'recall'
                - 'f1'
                - 'roc_auc'
                - mcc
            """

        best_model = None
        best_score = float('-inf')
        if not use_selected_features:
            mp = self.model_performance.items()
        else:
            mp = self.model_fs_performance.items()

        for model, metrics in mp:
            if metric not in metrics:
                raise ValueError(f"The metric '{metric}' is not found in the metrics for model '{model}'")

            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model

        if best_model is None:
            raise ValueError(f"No valid metric '{metric}' found in any model metrics")

        if use_selected_features:
            self.final_model = self.trained_fs_model[best_model]['model']
        else:
            self.final_model = self.trained_model[best_model]['model']

    def interpret_model(self,
                        method='shap',
                        n_repeats=100,
                        n_features=20,
                        index=[0],
                        use_whole_dataset=False,
                        use_selected_features=False,
                        plot_savedir='./',
                        show=True,
                        random_state=1):
        """
            Interprets the machine learning model using the specified interpretation method.

            Parameters:
            ----------
            method : str, default='shap'
                The method used for model interpretation. Currently supports:
                - 'shap': SHapley Additive exPlanations
                - 'permutation': Permutation Feature Importance

            n_repeats : int, default=100
                The number of times to repeat the permutation importance calculation.
                Only relevant if the method is 'permutation'.

            index : list of int, default=[0]
                The indices of the samples for which to generate SHAP waterfall plots.
                Only relevant if method is 'shap'.

            use_whole_dataset : bool, default=False
                If True, the entire dataset will be used for interpretation.
                If False, only a subset of the data will be used.

            use_selected_features : bool, default=False
                If True, the model built from a subset of features selected through feature selection techniques
                 will be used for interpretation.
                If False, the model built with all features will be used.

            plot_savedir : str, default='./'
                The directory where plots will be saved.

            random_state : int, default=1
                The random seed for reproducibility.

            """

        if self.split:

            if use_selected_features == False:
                X_test = self.X_test
                y_test = self.y_test
            else:
                X_test = self.X_test_fs
                y_test = self.y_test

            self.shap_permutation_value = interpret_model(model=self.final_model,
                                                          X_test=X_test,
                                                          y_test=y_test,
                                                          method=method,
                                                          n_repeats=n_repeats,
                                                          n_features=n_features,
                                                          index=index,
                                                          plotpath=plot_savedir,
                                                          show=show,
                                                          random_state=random_state)
        elif self.holdout_X is not None:
            self.shap_permutation_value = interpret_model(model=self.final_model,
                                                          X_test=self.holdout_X,
                                                          y_test=self.holdout_y,
                                                          method=method,
                                                          n_repeats=n_repeats,
                                                          n_features=n_features,
                                                          index=index,
                                                          plotpath=plot_savedir,
                                                          random_state=random_state)
        else:
            if use_selected_features == False:
                X_train = self.X_train
                y_train = self.y_train
            else:
                X_train = self.X_train_fs
                y_train = self.y_train

            if self.balance_method is None:
                if self.scaling_method is not None:
                    index1 = find_matching_rows(self.o_Xtrain, self.X_train, index)
            else:
                index1 = index

            self.shap_permutation_value = interpret_model(model=self.final_model,
                                                          X_test=X_train,
                                                          y_test=y_train,
                                                          cv_f=self.cv_f,
                                                          method=method,
                                                          n_repeats=n_repeats,
                                                          n_features=n_features,
                                                          use_whole_dataset=use_whole_dataset,
                                                          index=index1,
                                                          plotpath=plot_savedir,
                                                          show=show,
                                                          random_state=random_state)

    def model_specificity(self,
                          disease,
                          target_column='disease',
                          scoring=['roc_auc'],
                          use_packagedata=True,
                          file=None,
                          plot_path=None
                          ):
        """
            Evaluate the specificity of a machine learning model.

            This function is designed to test the specificity of a machine learning model for different diseases.
            The diseases can be predefined ('ACVD', 'IBD', 'T2D', 'T1D', 'CRC', 'PD') or user-defined through a custom file.

            Parameters:
            ----------
            disease : list
                A list of diseases to evaluate. Can include predefined diseases ['ACVD', 'IBD', 'T2D', 'T1D', 'CRC', 'PD']
                or custom diseases provided by the user.

            target_column : str, default='disease'
                The name of the target column in the gut microbiota data file.

            scoring : list, optional, default=['roc_auc']
                A list of scoring metrics to evaluate the model's performance. Default is ['roc_auc'].

            use_packagedata : bool, optional, default=True
                Whether to use the packaged dataset. If False, a user-defined file must be provided.

            file : str, optional, default=None
                The name of the folder containing the user-defined data file. Required when `use_packagedata` is False.

            plot_path : str, optional, default=None
                The path where the result plots will be saved. If None, plots will not be saved.

            Returns:
            -------
            results : dict
                A dictionary containing the evaluation results for the specified diseases and scoring metrics.

            Example:
            --------
            model.model_specificity(disease=['ACVD', 'T2D'], target_column='label', scoring=['roc_auc', 'f1'], use_packagedata=True)
        """

        auroc_list = []
        for i in disease:

            if use_packagedata:
                fn = os.path.join(os.path.dirname(__file__), 'datasets', i + '_specificity.csv')
                data1 = pd.read_csv(fn, index_col=0)
            else:
                if file is None:
                    raise ValueError("When use_packagedata is False, 'file' argument must be provided.")

                fn = file + i + '_specificity.csv'
                data1 = pd.read_csv(fn, index_col=0)

            data1['study'] = i
            data2 = self.X_train.copy()

            data2['study'] = self.case
            df_list = [data1, data2]
            tot_df = pd.concat(df_list, axis=0, join='outer', ignore_index=False).fillna(0)
            df_file = tot_df[tot_df['study'] == i]
            # df_file = df_file.drop('study', axis=1)
            index_list = [item for item in data2.columns if item != 'study']
            index_list.append(target_column)
            df_file = df_file.loc[:, index_list]

            mp = model_specificity_assessment(model=self.final_model,
                                              data=df_file,
                                              case=i,
                                              target_column=target_column,
                                              transformation_method=self.transformation_method,
                                              pseudocount=self.pseudocount,
                                              scaling_method=self.scaling_method,
                                              scoring=scoring)
            mpd = {i: mp}
            auroc_list.append(mpd)

        _plotbar_ms_metrics(auroc_list, plot_path)

        return auroc_list

    def predict_with_model(self, new_data):
        """
            Predict using the trained model on new data.

            Parameters:
            ----------
            new_data : pd.DataFrame or str
                The new data to make predictions on. This can either be:
                - A pandas DataFrame containing the new data.
                - A string specifying the file path to the dataset.

        """
        try:
            if isinstance(new_data, pd.DataFrame):
                nw_df = new_data
            elif isinstance(new_data, str):
                nw_df = pd.read_csv(new_data, index_col=0)
            tc = 'mp'
            nw_df['mp'] = 'mp'

            nw = EncodeSplit(nw_df,
                             target_column=tc,
                             task=self.task,
                             case=self.case)
            # ho.encode_target()
            nw.split_data(split=False)
            nw.preprocess(split=False,
                          transformation_method=self.transformation_method,
                          pseudocount=self.pseudocount,
                          scaling_method=self.scaling_method
                          )
            n_test = nw.X_train
            predictions = self.final_model.predict(n_test)
            return predictions
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None

    def save_model(self, save_path):
        """
            Save a easyml instance to a file.

            Parameters:
            - save_path: The file path where the model will be saved.
        """
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)
        print(f"Model saved to {save_path}")


class DataLoader:
    def __init__(self, filename, target=None):
        self.filename = filename
        self.target = target

    def load_data(self):
        try:
            if self.target is not None:
                target_column = self.target
            else:
                raise ValueError("Target column not specified.")

            data = pd.read_csv(self.filename, index_col=0)
            print("Data loaded successfully.")

            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in the loaded data.")

            return data
        except FileNotFoundError:
            raise ValueError("File not found. Please check the file path.")


class EncodeSplit:
    def __init__(self,
                 data,
                 target_column,
                 task,
                 case,
                 random_state=None,
                 ):

        self.data = data
        self.target_column = target_column
        self.task = task
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.case = case
        self.random_state = random_state

        # self.transformation_method=transformation_method
        # self.balance_method=None

    def encode_target(self):
        if self.target_column in self.data.columns:
            if self.case not in self.data[self.target_column].to_list():
                raise ValueError(f"The case is not in the specified column.")
            if all(isinstance(item, (int, str)) for item in self.data[self.target_column]):
                task_type = 'classification'
            elif all(isinstance(item, (int, float)) for item in self.data[self.target_column]):
                task_type = 'regression'
            if task_type == self.task:
                if self.task == 'classification':
                    self.data[self.target_column] = self.data[self.target_column].apply(
                        lambda x: 1 if x == self.case else 0)
                    print("Target column encoded successfully.")
            else:
                raise ValueError(f"The task type inferred from the target variable does not match the specified task.")
        else:
            raise ValueError(f"Target column '{self.target_column}' not found in the data.")

    def split_data(self, split, test_size=0.2, stratify=False, varf=None, corf=None):
        if split == False:
            # self.X_train = self.data.drop(columns=[self.target_column])
            # self.y_train = self.data[self.target_column]
            X = self.data.drop(columns=[self.target_column])
            self.y_train = self.data[self.target_column]
            feature_filter = FeatureFilter(threshold_variance=varf, threshold_correlation=corf)
            self.X_train = feature_filter.filter_features(X)
        else:
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            if stratify:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                        y,
                                                                                        test_size=test_size,
                                                                                        stratify=y,
                                                                                        random_state=self.random_state
                                                                                        )

                feature_filter = FeatureFilter(threshold_variance=varf, threshold_correlation=corf)
                self.X_train = feature_filter.fit_transform(self.X_train)
                self.X_test = feature_filter.transform(self.X_test)
            else:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                                        y,
                                                                                        test_size=test_size,
                                                                                        random_state=self.random_state
                                                                                        )
                feature_filter = FeatureFilter(threshold_variance=varf, threshold_correlation=corf)
                self.X_train = feature_filter.fit_transform(self.X_train)
                self.X_test = feature_filter.transform(self.X_test)

    def preprocess(self,
                   split,
                   transformation_method=None,
                   pseudocount=0.000001,
                   balance_method=None,
                   sampling_strategy='auto',
                   scaling_method=None
                   ):

        if transformation_method is not None:
            if split:
                self.X_train, self.X_test = self.microbiome_transformations(
                    split=split,
                    transformation_method=transformation_method,
                    pseudocount=pseudocount)
            else:
                self.X_train = self.microbiome_transformations(
                    split=split,
                    transformation_method=transformation_method,
                    pseudocount=pseudocount)

        if scaling_method is not None:
            if split:
                self.X_train, self.X_test = self.feature_scaling(split, scaling_method)
            else:
                self.X_train = self.feature_scaling(split, scaling_method)

        if balance_method == 'Under_Sampling':
            self.under_sampling(sampling_strategy=sampling_strategy, random_state=self.random_state)
        elif balance_method == 'Over_Sampling':
            self.over_sampling(sampling_strategy=sampling_strategy, random_state=self.random_state)
        elif balance_method == 'SMOTE':
            self.smote(sampling_strategy=sampling_strategy, random_state=self.random_state)
        elif balance_method == 'ADASYN':
            self.adasyn(sampling_strategy=sampling_strategy, random_state=self.random_state)
        elif balance_method == 'ENN':
            self.enn(sampling_strategy=sampling_strategy, random_state=self.random_state)

    def microbiome_transformations(self, split, transformation_method, pseudocount):
        if transformation_method == 'CLR':
            transformer = FunctionTransformer(func=CLR_normalize, kw_args={'pseudocount': pseudocount}, validate=False)

        elif transformation_method == 'Relative_Abundance':
            transformer = FunctionTransformer(lambda x: x.div(x.sum(axis=1), axis=0), validate=False)

        elif transformation_method == 'Hellinger':
            transformer = FunctionTransformer(lambda x: np.sqrt(x.div(x.sum(axis=1), axis=0)), validate=False)

        elif transformation_method == 'Lognorm':
            transformer = FunctionTransformer(func=lognorm, kw_args={'pseudocount': pseudocount}, validate=False)

        if split:
            X_train = transformer.fit_transform(self.X_train)
            X_test = transformer.transform(self.X_test)
            return X_train, X_test
        else:
            return transformer.fit_transform(self.X_train)

    def under_sampling(self, sampling_strategy, random_state):
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        self.X_train, self.y_train = sampler.fit_resample(self.X_train, self.y_train)

    def over_sampling(self, sampling_strategy, random_state):
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
        self.X_train, self.y_train = sampler.fit_resample(self.X_train, self.y_train)

    def smote(self, sampling_strategy, random_state):
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)

    def adasyn(self, sampling_strategy, random_state):
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
        self.X_train, self.y_train = adasyn.fit_resample(self.X_train, self.y_train)

    def enn(self, sampling_strategy, random_state):
        enn = EditedNearestNeighbours(sampling_strategy=sampling_strategy)
        self.X_train, self.y_train = enn.fit_resample(self.X_train, self.y_train)

    def feature_scaling(self, split, method):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        if split:
            if method == 'minmax':
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(self.X_train)
                X_test = scaler.transform(self.X_test)
                return X_train, X_test
            if method == 'standard':
                scaler = StandardScaler()
                X_train = scaler.fit_transform(self.X_train)
                X_test = scaler.transform(self.X_test)
                return X_train, X_test
        else:
            if method == 'minmax':
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(self.X_train)
                return X_train
            if method == 'standard':
                scaler = StandardScaler()
                X_train = scaler.fit_transform(self.X_train)
                return X_train


class FeatureFilter:
    def __init__(self, threshold_variance=None, threshold_correlation=None):
        self.threshold_variance = threshold_variance
        self.threshold_correlation = threshold_correlation
        self.selected_features = None

    def filter_by_variance(self, X):
        if self.threshold_variance is not None:
            variances = X.var(axis=0)
            return X.loc[:, variances > self.threshold_variance]
        return X

    def filter_by_correlation(self, X):
        if self.threshold_correlation is not None:
            correlation_matrix = X.corr().abs()
            upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if
                       any(upper_triangle[column] > self.threshold_correlation)]
            return X.drop(columns=to_drop)
        return X

    def filter_features(self, X):
        X_filtered = self.filter_by_variance(X)
        X_filtered = self.filter_by_correlation(X_filtered)
        return X_filtered

    def fit(self, X):
        X_filtered = self.filter_features(X)
        self.selected_features = X_filtered.columns
        return self

    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("FeatureFilter must be fitted before calling transform.")
        return X.loc[:, self.selected_features]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def _create_cv_object(cv, fold, repeat, random_state):
    if cv == 'KFold':
        return KFold(n_splits=fold, shuffle=True, random_state=random_state)
    elif cv == 'StratifiedKFold':
        return StratifiedKFold(n_splits=fold, shuffle=True, random_state=random_state)
    elif cv == 'StratifiedGroupKFold':
        return StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=random_state)
    elif cv == 'RepeatedKFold':
        return RepeatedKFold(n_splits=fold, n_repeats=repeat, random_state=random_state)
    elif cv == 'RepeatedStratifiedKFold':
        return RepeatedStratifiedKFold(n_splits=fold, n_repeats=repeat, random_state=random_state)
    else:
        raise ValueError("Unsupported cross-validation method: {}".format(cv))


def _plot_heatmap(evaluation_results, filepath, annot=True, cmap='YlGnBu'):
    import seaborn as sns

    data = [[k[0], k[1], v] for k, v in evaluation_results.items()]
    evaluation_df = pd.DataFrame(data, columns=['Source Dataset', 'Target Dataset', 'value'])
    pivot_df = evaluation_df.pivot(index='Source Dataset', columns='Target Dataset', values='value')

    plt.figure(figsize=(10, 8))

    sns.heatmap(pivot_df, annot=annot, fmt=".2f", cmap=cmap, square=True, linewidths=.5)

    plt.title(f"Study-to-Study Transfer Validation")
    plt.xlabel("Target Dataset")
    plt.ylabel("Source Dataset")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    filename = filepath + 'Study-to-Study'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def _plotbar_ms_metrics(data, pp):
    diseases = [list(d.keys())[0] for d in data]
    # 从第一个疾病中提取第一个评估器名
    first_disease = data[0][diseases[0]]
    evaluator_name = list(first_disease.keys())[0]  # 评估器名称，例如 'RandomForest'

    metrics = list(first_disease[evaluator_name].keys())  # 提取指标

    scores = {metric: [] for metric in metrics}

    for d in data:
        disease = list(d.keys())[0]
        evaluator = d[disease]  # 获取当前疾病的评估器
        for metric in metrics:
            scores[metric].append(evaluator[evaluator_name][metric])  # 提取指标值

    x = np.arange(len(diseases))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots()
    rects = []

    for i, metric in enumerate(metrics):
        rect = ax.bar(x + i * width, scores[metric], width, label=metric)
        rects.append(rect)

    ax.set_ylabel('Scores')
    ax.set_title('Model Metrics')
    ax.set_xticks(x + width * len(metrics) / 2 - width / 2)
    ax.set_xticklabels(diseases)
    ax.legend()

    if pp is not None:
        fn = pp + 'Model_Specificity.png'
        plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.show()


def model_specificity_assessment(model,
                                 data,
                                 case,
                                 target_column,
                                 transformation_method=None,
                                 pseudocount=0.00001,
                                 scaling_method=None,
                                 scoring=['roc_auc']
                                 ):
    nd = EncodeSplit(data,
                     task='classification',
                     target_column=target_column,
                     case=case)

    nd.encode_target()
    nd.split_data(split=False)
    nd.preprocess(split=False,
                  transformation_method=transformation_method,
                  pseudocount=pseudocount,
                  scaling_method=scaling_method
                  )
    n_X = nd.X_train
    n_y = nd.y_train
    # predictions = model.predict(n_test)
    mp = evaluate_model(model=model,
                        X_test=n_X,
                        y_test=n_y,
                        scoring=scoring,
                        task='classification')
    return mp


def study_to_study_transfer(filename,
                            target,
                            case,
                            model_name='RandomForest',
                            task='classification',
                            split=False,
                            stratify=False,
                            group=None,
                            transformation_method=None,
                            balance_method=None,
                            pseudocount=0.00001,
                            sampling_strategy='auto',
                            scaling_method=None,
                            scoring='roc_auc',
                            n_iter=100,
                            search_methods='random',
                            cv='StratifiedKFold',
                            fold=5,
                            repeat=5,
                            random_state=None,
                            cmap='YlGnBu',
                            plotpath='./',
                            n_jobs=-1):
    """
        Perform study-to-study transfer learning, training models with data from one study
        and applying to another under specific settings.

        Parameters:
        ----------
        filename : str
            Path or name of the data file.

        target : str
            Name of the target variable for prediction.

        case : str
            Specifies the case or scenario being addressed.

        model_name : list, default=['RandomForest']
            List of model names to train. Supported models include:
            - 'RandomForest'
            - 'SVM'
            - 'LogisticRegression'
            - Other supported model names

        task : str, default='classification'
            Type of task ('classification' or 'regression'). Defaults to 'classification'.

        split : bool, default=False
            Whether to split the dataset. Defaults to False.

        stratify : bool, default=False
            Whether to stratify split if split is True. Defaults to False.

        transformation_method : callable, default=None
            Method for data transformation. Defaults to None.

        balance_method : callable, default=None
            Method for balancing the dataset. Defaults to None.

        pseudocount : float, default=0.00001
            Pseudocount to add to features to prevent zero values. Defaults to 0.00001.

        sampling_strategy : str, default='auto'
            Strategy for sampling the data. Defaults to 'auto'.
        scaling_method : callable, default=None
            Method for scaling the data. Defaults to None.
        scoring : str, default='roc_auc'
            Metric for evaluating model performance. Common evaluation metrics include:
            - 'accuracy'
            - 'precision'
            - 'recall'
            - 'f1'
            - 'roc_auc'
            - 'mcc'
        n_iter : int, default=10
            Number of iterations for hyperparameter search. Applicable for search methods like Bayesian optimization and random search.
        search_methods : str, default='bayesian'
            Hyperparameter search method. Supported methods include:
            - 'grid' (Grid Search)
            - 'random' (Random Search)
            - 'bayesian' (Bayesian Optimization)
        cv : str, default='StratifiedKFold'
            Cross-validation method. Supported cross-validation methods include:
            - 'KFold'
            - 'StratifiedKFold'
            - 'StratifiedGroupKFold'
        fold : int, default=5
            Number of folds for cross-validation. Defaults to 5.
        random_state : int, default=None
            Seed for random number generator. Defaults to None.
        cmap : str, default='YlGnBu'
            Colormap for plotting. Defaults to 'YlGnBu'.
        plotpath : str, default='./'
            Path to save the output files. Defaults to './'.
        n_jobs : int, default=-1
            Number of jobs to run in parallel. Defaults to -1.
    """
    from IPython.display import display
    if isinstance(filename, str) and os.path.isdir(filename):
        files = [f for f in os.listdir(filename) if f.endswith('.csv')]
        files = [os.path.join(filename, f) for f in files]
    elif isinstance(filename, list):
        files = filename
    else:
        raise ValueError("Invalid input filename. It should be either a folder name or a list of filenames.")

    fastml_instances = {}
    evaluation_results = {}
    if balance_method is None:
        bm = [None] * len(files)
    if not isinstance(cv, list):
        cv = [cv] * len(files)

    df_list = []
    group_list = []
    for i, file in enumerate(files):
        #print(file)
        # f =  os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file, index_col=0)
        if group in df.columns:
            group_list.append(group)
        else:
            group_list.append(None)
        df['study'] = [os.path.splitext(os.path.basename(file))[0]] * df.shape[0]
        # print(df)
        df_list.append(df)

    tot_df = pd.concat(df_list, axis=0, join='outer', ignore_index=False).fillna(0)

    for i, file in enumerate(files):
        dataset_name = os.path.splitext(os.path.basename(file))[0]
        df_file = tot_df[tot_df['study'] == dataset_name]
        df_file = df_file.drop('study', axis=1)
        group_study = group_list[i]
        balance_m = bm[i]
        cvs = cv[i]
        #print(group_study)
        if group_study is None:
            df_file = df_file.drop(group, axis=1)
        easyml = easyML(filename=df_file,
                        case=case,
                        target=target,
                        task=task,
                        split=split,
                        group=group_list[i],
                        transformation_method=transformation_method,
                        balance_method=balance_m,
                        pseudocount=pseudocount,
                        sampling_strategy=sampling_strategy,
                        stratify=stratify,
                        scaling_method=None,
                        var_filter=None,
                        # random_state=random_state,
                        n_jobs=n_jobs)

        easyml.preprocess_data(random_state=random_state)

        easyml.train_model(model_names=[model_name],
                           scoring=scoring,
                           n_iter=n_iter,
                           search_methods=search_methods,
                           cv=cvs,
                           fold=fold,
                           repeat=repeat,
                           use_selected_features=False,
                           random_state=random_state)

        easyml.evaluate_model(scoring=[scoring], plot_model=False, random_state=random_state)
        easyml.select_model(model_name)

        evaluation_results[(dataset_name, dataset_name)] = easyml.model_performance[model_name][scoring]

        fastml_instances[dataset_name] = easyml

        # 评估模型在其他数据集上的表现
        for j, other_file in enumerate(files):
            if i != j:  # 跳过当前数据集
                other_dataset_name = os.path.splitext(os.path.basename(other_file))[0]
                # df = pd.read_csv(other_file, index_col=0)
                # print(other_dataset_name)
                df = tot_df[tot_df['study'] == other_dataset_name]
                #print(df.shape)
                df = df.drop('study', axis=1)
                if group in df.columns:
                    df = df.drop(group, axis=1)
                # print(df.shape)
                # from sklearn.preprocessing import LabelEncoder
                # X_test = df.drop([target, 'study'], axis=1)
                # y_test = df[target].apply(lambda x: 1 if x == case else 0)

                od = EncodeSplit(df,
                                 target_column=target,
                                 task=task,
                                 case=case,
                                 random_state=random_state
                                 )

                # if task=='classification':
                od.encode_target()

                od.split_data(split=False)

                od.preprocess(split=False,
                              transformation_method=transformation_method,
                              # balance_method=balance_method,
                              pseudocount=pseudocount,
                              scaling_method=scaling_method,
                              # sampling_strategy=sampling_strategy
                              )
                #print(od.X_train.shape)
                X_test = od.X_train
                y_test = od.y_train
                #display(X_test)
                # 使用当前数据集的模型进行预测
                y_pred = easyml.final_model.predict(X_test)

                # 计算评价指标
                if scoring == 'accuracy':
                    evaluation_results[(dataset_name, other_dataset_name)] = accuracy_score(y_test, y_pred)
                elif scoring == 'precision':
                    evaluation_results[(dataset_name, other_dataset_name)] = precision_score(y_test, y_pred)
                elif scoring == 'recall':
                    evaluation_results[(dataset_name, other_dataset_name)] = recall_score(y_test, y_pred)
                elif scoring == 'specificity':
                    evaluation_results[(dataset_name, other_dataset_name)] = _specificity_score(y_test, y_pred)
                elif scoring == 'f1':
                    evaluation_results[(dataset_name, other_dataset_name)] = f1_score(y_test, y_pred)
                elif scoring == 'roc_auc':
                    y_pred_proba = easyml.final_model.predict_proba(X_test)[:, 1]
                    evaluation_results[(dataset_name, other_dataset_name)] = roc_auc_score(y_test, y_pred_proba)

    _plot_heatmap(evaluation_results=evaluation_results, cmap=cmap, filepath=plotpath)

    return evaluation_results, fastml_instances


def leave_one_study_out(filename,
                        target,
                        case,
                        model_name='RandomForest',
                        task='classification',
                        stratify=False,
                        split=False,
                        group=None,
                        transformation_method=None,
                        balance_method=None,
                        pseudocount=0.00001,
                        sampling_strategy='auto',
                        scaling_method=None,
                        scoring='roc_auc',
                        n_iter=100,
                        search_methods='random',
                        cv='StratifiedKFold',
                        fold=5,
                        repeat=5,
                        random_state=None,
                        plotpath='./',
                        n_jobs=-1):
    """
        Performs leave-one-study-out cross-validation to evaluate model performance across different studies.
        This approach is suitable for scenarios where study-specific biases must be controlled.

        Parameters:
        ----------
        filename : str
            Path or name of the data file.

        target : str
            Name of the target variable for prediction.

        case : str
            Specifies the case or scenario being addressed.

        model_name : list, default=['RandomForest']
            List of model names to train. Supported models include:
            - 'RandomForest'
            - 'SVM'
            - 'LogisticRegression'
            - Other supported model names

        task : str, default='classification'
            Type of task ('classification' or 'regression'). Defaults to 'classification'.

        stratify : bool, default=False
            Whether to stratify split if split is True. Defaults to False.

        transformation_method : callable, default=None
            Method for data transformation. Defaults to None.

        balance_method : callable, default=None
            Method for balancing the dataset. Defaults to None.

        pseudocount : float, default=0.00001
            Pseudocount to add to features to prevent zero values. Defaults to 0.00001.

        sampling_strategy : str, default='auto'
            Strategy for sampling the data. Defaults to 'auto'.

        scaling_method : callable, default=None
            Method for scaling the data. Defaults to None.

        scoring : str, default='roc_auc'
            Metric for evaluating model performance. Common evaluation metrics include:
            - 'accuracy'
            - 'precision'
            - 'recall'
            - 'f1'
            - 'roc_auc'

        n_iter : int, default=100
            Number of iterations for hyperparameter search. Defaults to 100.

        search_methods : str, default='random'
            Hyperparameter search method. Default is 'random'.

        cv : str, default='StratifiedKFold'
            Cross-validation method. Supported cross-validation methods include:
            - 'KFold'
            - 'StratifiedKFold'
            - 'GroupKFold'

        fold : int, default=5
            Number of folds for cross-validation. Defaults to 5.

        random_state : int, default=None
            Seed for random number generator. Defaults to None.

        plotpath : str, default='./'
            Path to save the output files. Defaults to './'.

        n_jobs : int, default=-1
            Number of jobs to run in parallel. Defaults to -1.

        """

    if isinstance(filename, str) and os.path.isdir(filename):
        files = [f for f in os.listdir(filename) if f.endswith('.csv')]
        files = [os.path.join(filename, f) for f in files]
    elif isinstance(filename, list):
        files = filename
    else:
        raise ValueError("Invalid input filename. It should be either a folder name or a list of filenames.")

    easyml_instances = {}

    evaluation_results = {}

    df_list = []
    for file in files:
        # print(file)
        # f =  os.path.splitext(os.path.basename(file))[0]
        df = pd.read_csv(file, index_col=0)
        df['study'] = [os.path.splitext(os.path.basename(file))[0]] * df.shape[0]
        # print(df)
        df_list.append(df)

    tot_df = pd.concat(df_list, axis=0, join='outer', ignore_index=False).fillna(0)

    for i, file_test in enumerate(files):

        dataset_name = os.path.splitext(os.path.basename(file_test))[0]
        df_file = tot_df[tot_df['study'] != dataset_name]
        df_file = df_file.drop('study', axis=1)

        easyml = easyML(filename=df_file,
                        case=case,
                        target=target,
                        task=task,
                        split=split,
                        group=group,
                        transformation_method=transformation_method,
                        balance_method=balance_method,
                        pseudocount=pseudocount,
                        sampling_strategy=sampling_strategy,
                        scaling_method=None,
                        stratify=stratify,
                        var_filter=None,
                        cor_filter=None,
                        # random_state=random_state,
                        n_jobs=n_jobs)

        easyml.preprocess_data(random_state=random_state)

        easyml.train_model(model_names=[model_name],
                           scoring=scoring,
                           n_iter=n_iter,
                           search_methods=search_methods,
                           cv=cv,
                           fold=fold,
                           repeat=repeat,
                           use_selected_features=False,
                           random_state=random_state)

        easyml.select_model(model_name)
        easyml_instances[dataset_name] = easyml

        # 评估模型在其他数据集上的表现
        # 加载其他数据集              
        df = tot_df[tot_df['study'] == dataset_name]
        # X_test = df.drop([target, 'study'], axis=1)
        # y_test = df['disease']
        # le = LabelEncoder()
        # y_test = le.fit_transform(y_test)

        # y_test = df[target].apply(lambda x: 1 if x == case else 0)
        # y_pred = easyml.trained_model[0].predict(X_test)
        df = df.drop('study', axis=1)
        if group in df.columns:
            df = df.drop(group, axis=1)
        od = EncodeSplit(df,
                         target_column=target,
                         task=task,
                         case=case,
                         random_state=random_state
                         )

        # if task=='classification':
        od.encode_target()

        od.split_data(split=False)

        od.preprocess(split=False,
                      transformation_method=transformation_method,
                      balance_method=balance_method,
                      pseudocount=pseudocount,
                      sampling_strategy=sampling_strategy,
                      scaling_method=scaling_method
                      )
        # print(od.X_train.shape)
        X_test = od.X_train
        y_test = od.y_train

        y_pred = easyml.final_model.predict(X_test)

        if scoring == 'accuracy':
            evaluation_results[dataset_name] = accuracy_score(y_test, y_pred)
        elif scoring == 'precision':
            evaluation_results[dataset_name] = precision_score(y_test, y_pred)
        elif scoring == 'recall':
            evaluation_results[dataset_name] = recall_score(y_test, y_pred)
        elif scoring == 'f1':
            evaluation_results[dataset_name] = f1_score(y_test, y_pred)
        elif scoring == 'roc_auc':
            y_pred_proba = easyml.final_model.predict_proba(X_test)[:, 1]
            evaluation_results[dataset_name] = roc_auc_score(y_test, y_pred_proba)
            # evaluation_results[dataset_name] = roc_auc_score(y_test, y_pred)
        elif scoring == 'mcc':
            evaluation_results[dataset_name] = matthews_corrcoef(y_test, y_pred)

    _create_bar_chart(evaluation_results, scoring, plotpath)

    return evaluation_results


def oneliner_train_model(filename,
                         target,
                         case,
                         split=True,
                         model_name=['RandomForest'],
                         task='classification',
                         stratify=True,
                         group=None,
                         transformation_method='Relative_Abundance',
                         balance_method=None,
                         pseudocount=0.00001,
                         sampling_strategy='auto',
                         scaling_method=None,
                         var_filter=0,
                         cor_filter=None,
                         scoring='roc_auc',
                         n_iter=100,
                         search_methods='random',
                         cv='StratifiedKFold',
                         fold=5,
                         holdout=None,
                         eval_scoring=['roc_auc', 'f1'],
                         plot_model=True,
                         model_select_metric='roc_auc',
                         plot_savedir='./',
                         random_state=None,
                         n_jobs=-1):
    """
       Train a machine learning model with one line of code.

       Parameters:
       ----------
       filename : str
           The path to the dataset file to be loaded.

       target : str
           The name of the target variable in the dataset.

       case : str
           A descriptive name or identifier for the case.

       split : bool, default=True
           If True, the dataset will be split into training and testing sets.

       model_name : list, default=['RandomForest']
           Supported model names include:
           - 'RandomForest': Random Forest classifier.
           - 'ExtraTrees': Extra Trees classifier.
           - 'LogisticRegression': Logistic Regression model.
           - 'GradientBoosting': Gradient Boosting classifier.
           - 'XGBoost': XGBoost classifier.

       task : str, default='classification'
           The type of machine learning task. Supported tasks are:
           - 'classification'
           - 'regression'

       stratify : bool, default=True
           If True, the data split will be stratified according to the target variable.

       transformation_method : str or None, default=None
           The method to apply for data transformation. Supported methods include:
           - 'CLR'
           - 'Relative_Abundance'
           - 'Hellinger'
           - 'Lognorm'

           Detailed information can be found in the article:
           Proportion-based normalizations outperform compositional data transformations
           in machine learning applications Microbiome (2024) 12:45.

       balance_method : str or None, default=None
           The method to use for balancing the classes in the target variable. Supported methods include:
           - 'Under_Sampling'
           - 'Over_Sampling'
           - 'SMOTE'
           - 'ADASYN'
           - 'ENN'

           Refer to the imbalanced-learn package for more details.

       pseudocount : float, default=0.00001
           A small value to add to avoid division by zero or log of zero in data transformation.

       sampling_strategy : str, default='auto'
           The sampling strategy to use for balancing. Relevant only if balance_method is specified.

       scaling_method : str or None, default=None
           The method to use for scaling features. Supported methods include:
           - 'standard': Standard Scaling
           - 'minmax': Min-Max Scaling

       scoring : str, default='roc_auc'
           Metric for evaluating model performance. Supported metrics include:
           - 'accuracy'
           - 'precision'
           - 'recall'
           - 'f1'
           - 'roc_auc'
           - 'mcc'

       n_iter : int, default=100
           Number of iterations for random search and hyperparameter tuning.

       search_methods : str, default='bayesian'
           Hyperparameter search method. Supported methods include:
           - 'grid' (Grid Search)
           - 'random' (Random Search)
           - 'bayesian' (Bayesian Optimization)

       cv : str, default='StratifiedKFold'
           Cross-validation method. Supported cross-validation methods include:
           - 'KFold'
           - 'StratifiedKFold'
           - 'RepeatedKFold'
           - 'RepeatedStratifiedKFold'

       fold : int, default=5
           Number of folds for cross-validation.

       holdout : float or None, default=None
           dataset to hold out for validation. If None, no holdout set is used.

       eval_scoring : list, default=['roc_auc']
           List of evaluation metrics to report. Supported metrics include:
           - 'accuracy'
           - 'precision'
           - 'recall'
           - 'f1'
           - 'roc_auc'
           - 'mcc'

       plot_model : bool, default=True
           If True, ROC and Precision-Recall curves will be plotted.

       plot_savedir : str or None, default='./'
           Directory to save the plots.

       random_state : int or None, default=None
           Random state for reproducibility.

       n_jobs : int, default=-1
           The number of CPU cores to use for computations. -1 means using all processors.

    """

    if eval_scoring is None:
        eval_scoring = ['roc_auc', 'f1']
    easyml = easyML(filename=filename,
                    case=case,
                    target=target,
                    split=split,
                    task=task,
                    group=group,
                    transformation_method=transformation_method,
                    balance_method=balance_method,
                    pseudocount=pseudocount,
                    sampling_strategy=sampling_strategy,
                    scaling_method=scaling_method,
                    var_filter=var_filter,
                    cor_filter=cor_filter,
                    stratify=stratify,
                    n_jobs=n_jobs)

    easyml.preprocess_data(random_state=random_state)

    easyml.train_model(model_names=model_name,
                       scoring=scoring,
                       n_iter=n_iter,
                       search_methods=search_methods,
                       cv=cv,
                       fold=fold,
                       random_state=random_state)

    easyml.evaluate_model(  # holdout=holdout,
                          scoring=eval_scoring,
                          plot_model=plot_model,
                          plot_savedir=plot_savedir,
                          random_state=random_state)

    easyml.auto_select_model(metric=model_select_metric)

    if split:
        easyml.evaluate_model(holdout='test',
                              scoring=eval_scoring,
                              plot_model=plot_model,
                              plot_savedir=plot_savedir,
                              random_state=random_state)
    elif holdout is not None:
        easyml.evaluate_model(holdout=holdout,
                              scoring=eval_scoring,
                              plot_model=plot_model,
                              plot_savedir=plot_savedir,
                              random_state=random_state)

    return easyml


def _create_bar_chart(data, scoring, plotpath):
    categories = list(data.keys())
    values = list(data.values())

    bar = plt.bar(categories, values)
    plt.title('Leave One Study Out Validation')
    plt.xlabel('Study')
    plt.ylabel(scoring)
    plt.xticks(rotation=45)

    '''
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
    '''
    plt.tight_layout()
    filename = plotpath + 'Leave-one-Study-out'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
