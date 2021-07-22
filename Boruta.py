from dataload import Dataset
import argparse
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from tqdm import tqdm
import shap
import ray

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import binom_test
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

if not ray.is_initialized():
    ray.init(address="auto")
#Source code from https://github.com/Ekeany/Boruta-Shap
#modified by Dongkyun 2021.07.22
@ray.remote
def fitting(obj,X, y, model):
    X = X.copy()
    y = y.copy()
    model =clone(model)
    X_shadow = X.apply(np.random.permutation)
    # append
    obj_col = X_shadow.select_dtypes("object").columns.tolist()
    if obj_col == []:
        pass
    else:
        X_shadow[obj_col] = X_shadow[obj_col].astype("category")

    X_shadow.columns = ['shadow_' + feature for feature in X.columns]

    X_boruta = pd.concat([X, X_shadow], axis=1)
    X_boruta_train, X_boruta_test, y_train, y_test = train_test_split(X_boruta,
                                                                      y,
                                                                      test_size=0.2,
                                                                      random_state=42)
    model.fit(X_boruta_train, y_train)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    shap_values = np.array(explainer.shap_values(X_boruta_test))
    shap_values = np.abs(shap_values).sum(axis=0)
    shap_values = shap_values.mean(0)
    vals = shap_values


    mean_value = np.mean(vals)
    std_value = np.std(vals)
    vals = [(element - mean_value) / std_value for element in vals]

    X_feature_import = vals[:len(X.columns)]
    Shadow_feature_import = vals[len(X_shadow.columns):]
    return X_feature_import, Shadow_feature_import


class BorutaShap:
    def __init__(self, model=None,  pvalue=0.05):

        self.pvalue = pvalue
        self.model = model

    def fit(self, X, y, random_state=0,use_tentative=False, verbose=True):

        """
        The main body of the program this method it computes the following

        1. Extend the information system by adding copies of all variables (the information system
        is always extended by at least 5 shadow attributes, even if the number of attributes in
        the original set is lower than 5).

        2. Shuffle the added attributes to remove their correlations with the response.

        3. Run a random forest classifier on the extended information system and gather the
        Z scores computed.

        4. Find the maximum SHAP score among shadow attributes (MZSA), and then assign a hit to
        every attribute that scored better than MZSA.

        5. For each attribute with undetermined importance perform a two-sided test of equality
        with the MZSA.

        6. Deem the attributes which have importance significantly lower than MZSA as ‘unimportant’
        and permanently remove them from the information system.

        7. Deem the attributes which have importance significantly higher than MZSA as ‘important’.

        8. Remove all shadow attributes.

        9. Repeat the procedure until the importance is assigned for all the attributes, or the
        algorithm has reached the previously set limit of the random forest runs.

        10. Stores results.
        """

        np.random.seed(random_state)

        self.origin_X = X.copy()
        self.ncols = X.shape[1]
        self.all_columns = X.columns.to_numpy()
        self.rejected_columns = []
        self.accepted_columns = []

        pipe = Pipeline([
            ('imputer', SimpleImputer(missing_values=np.nan, strategy="median")),
            ('sampling_under', RandomUnderSampler(random_state=0, sampling_strategy=0.5)),
            ('sampling_over', SMOTE(random_state=0, sampling_strategy=1)),
        ])
        X, y = pipe.fit_resample(X, y)
        X = pd.DataFrame(X, columns=self.all_columns)

        self.hits = np.zeros(self.ncols)
        self.order = dict(zip(X.columns.to_list(), np.arange(X.shape[1])))


        self.history_shadow = np.zeros(self.ncols)
        self.history_x = np.zeros(self.ncols)

        print("Start Boruta filtering--------------------------")
        print("This process usually takes several minutes ~ hours")
        result= ray.get([fitting.remote(self,X, y, self.model) for trial in range(1000)])


        X_feature_importances = np.array(list(map(lambda x:x[0],result)))
        shadow_features_importances = np.array(list(map(lambda x:x[1],result)))

        self.hits = self.calculate_hits(X_feature_importances,shadow_features_importances) #dimension (num_colums,)
        self.test_features(iteration=1000)

        self.store_feature_importance(X_feature_importances,shadow_features_importances)
        self.calculate_rejected_accepted_tentative(verbose=verbose)

        if use_tentative:
            return self.accepted+self.tentative, self.rejected
        else:
            return self.accepted, self.tentative+self.rejected


    def calculate_rejected_accepted_tentative(self, verbose):

        """
        Figures out which features have been either accepted rejeected or tentative

        Returns
        -------
        3 lists

        """

        self.rejected = list(
            set(self.flatten_list(self.rejected_columns)) - set(self.flatten_list(self.accepted_columns)))
        self.accepted = list(set(self.flatten_list(self.accepted_columns)))
        self.tentative = list(set(self.all_columns) - set(self.rejected + self.accepted))

        if verbose:
            print(str(len(self.accepted)) + ' attributes confirmed important: ' + str(self.accepted))
            print(str(len(self.rejected)) + ' attributes confirmed unimportant: ' + str(self.rejected))
            print(str(len(self.tentative)) + ' tentative attributes remains: ' + str(self.tentative))





    def store_feature_importance(self,X_feature_importances,shadow_features_importances):

        """
        Reshapes the columns in the historical feature importance scores object also adds the mean, median, max, min
        shadow feature scores.

        Returns
        -------
        Datframe

        """

        self.history_x = pd.DataFrame(data=X_feature_importances,
                                      columns=self.all_columns)

        self.history_x['Max_Shadow'] = [max(i) for i in shadow_features_importances]
        self.history_x['Min_Shadow'] = [min(i) for i in shadow_features_importances]
        self.history_x['Mean_Shadow'] = [np.nanmean(i) for i in shadow_features_importances]
        self.history_x['Median_Shadow'] = [np.nanmedian(i) for i in shadow_features_importances]
        self.history_x.dropna(axis=0, inplace=True)


    @staticmethod
    def flatten_list(array):
        return [item for sublist in array for item in sublist]



    def calculate_hits(self,X_feature_importances,shadow_features_importances):

        """
        If a features importance is greater than the maximum importance value of all the random shadow
        features then we assign it a hit.

        Parameters
        ----------
        Percentile : value ranging from 0-1
            can be used to reduce value of the maximum value of the shadow features making the algorithm
            more lenient.

        """

        shadow_threshold = shadow_features_importances.max(axis=1,keepdims=True)
        hits = X_feature_importances > shadow_threshold
        hits = np.sum(hits,axis=0)
        return hits


    @staticmethod
    def binomial_H0_test(array, n, p, alternative):
        return [binom_test(x, n=n, p=p, alternative=alternative) for x in array]


    @staticmethod
    def find_index_of_true_in_array(array):
        length = len(array)
        return list(filter(lambda x: array[x], range(length)))

    @staticmethod
    def bonferoni_corrections(pvals, alpha=0.05, n_tests=None):
        """
        used to counteract the problem of multiple comparisons.
        """
        pvals = np.array(pvals)

        if n_tests is None:
            n_tests = len(pvals)
        else:
            pass

        alphacBon = alpha / float(n_tests)
        reject = pvals <= alphacBon
        pvals_corrected = pvals * float(n_tests)
        return reject, pvals_corrected

    def test_features(self, iteration):
        acceptance_p_values = self.binomial_H0_test(self.hits,
                                                    n=iteration,
                                                    p=0.5,
                                                    alternative='greater')

        regect_p_values = self.binomial_H0_test(self.hits,
                                                n=iteration,
                                                p=0.5,
                                                alternative='less')

        # [1] as function returns a tuple
        self.modified_acceptance_p_values = self.bonferoni_corrections(acceptance_p_values,
                                                                  alpha=0.05,
                                                                  n_tests=len(self.hits))[1]

        self.modified_regect_p_values = self.bonferoni_corrections(regect_p_values,
                                                              alpha=0.05,
                                                              n_tests=len(self.hits))[1]

        # Take the inverse as we want true to keep featrues
        rejected_columns = np.array(self.modified_regect_p_values) < self.pvalue
        accepted_columns = np.array(self.modified_acceptance_p_values) < self.pvalue

        rejected_indices = self.find_index_of_true_in_array(rejected_columns)
        accepted_indices = self.find_index_of_true_in_array(accepted_columns)

        rejected_features = self.all_columns[rejected_indices]
        accepted_features = self.all_columns[accepted_indices]


        self.rejected_columns.append(rejected_features)
        self.accepted_columns.append(accepted_features)






if __name__ == "__main__":

    result_diretory = f'./result/Boruta'
    csv_directory_fname = './data/sample.csv'
    data = Dataset(csv_directory_fname, 'Cardio')
    xtrain, xtest, ytrain, ytest, columns = data.load_dataset()



    Feature_Selector = BorutaShap(model=RandomForestClassifier(n_estimators=1000,max_depth=10,random_state=42), pvalue=0.05)
    accepted_columns, rejected_columns = Feature_Selector.fit(X=xtrain, y=ytrain, use_tentative=True,verbose=True)


