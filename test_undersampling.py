
"""Test the module nearmiss."""
from __future__ import print_function

import numpy as np
from sklearn.utils.testing import (assert_array_equal, assert_warns,
                                   assert_raises_regex)
import os
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from us import NearMiss
from IPython.display import Image
from pyspark import SparkContext
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from pyspark import SQLContext
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from MainFile import mainfile
from us import NearMiss
df1=pd.DataFrame.from_csv('cs-training-processed.csv',sep=',')
df_500=df1.sample(1000)
# print(df_500)
msk = np.random.rand(len(df_500))<0.8
training_data_sampled =df_500[msk]
X_training_data =df_500[msk]
test_data_sampled= df_500[~msk]
X_test_data= df_500[~msk]
# training_data['features'] = training_data[['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans'/
#                                            'NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']]
# training_data['Combined_ID'] = training_data.sum(axis=1)
# print(training_data)
# converted = training_data.apply(lambda f : to_number(f[0]) , axis = 1)
# training_data["features"] = training_data.apply(lambda r: (r["RevolvingUtilizationOfUnsecuredLines"] , r["age"],r['NumberOfTime30-59DaysPastDueNotWorse'],r['DebtRatio'],r['MonthlyIncome'],  r['NumberOfOpenCreditLinesAndLoans'],r['NumberOfTimes90DaysLate'],r['NumberRealEstateLoansOrLines']/
#                                                       r['NumberOfTime60-89DaysPastDueNotWorse'],r['NumberOfDependents']) , axis=1)
# test_data["features"] = test_data.apply(lambda r: (r["RevolvingUtilizationOfUnsecuredLines"] , r["age"],r['NumberOfTime30-59DaysPastDueNotWorse'],r['DebtRatio'],r['MonthlyIncome'],  r['NumberOfOpenCreditLinesAndLoans'],r['NumberOfTimes90DaysLate'],r['NumberRealEstateLoansOrLines']/
#                                                       r['NumberOfTime60-89DaysPastDueNotWorse'],r['NumberOfDependents']) , axis=1)


Y_train=X_training_data.pop('SeriousDlqin2yrs')
Y_test=X_test_data.pop('SeriousDlqin2yrs')
# Y_test.columns=['SeriousDlqin2yrs']
X_train = pd.get_dummies(training_data_sampled)
X_test=pd.get_dummies(test_data_sampled)
X_train.columns.difference(X_test.columns)
X_test[X_train.columns.difference(X_test.columns)] = 0
X_test = X_test[X_train.columns]
# Generate a global dataset to use for the 3 version of nearmiss
RND_SEED = 0
X =X_train
Y =Y_train

VERSION_NEARMISS = [1, 2, 3]


# FIXME remove at the end of the deprecation 0.4
def test_nearmiss_deprecation():
    nm = NearMiss(ver3_samp_ngh=3, version=3)
    assert_warns(DeprecationWarning, nm.fit, X, Y)


def test_nearmiss_wrong_version():
    version = 1000
    nm = NearMiss(version=version, random_state=RND_SEED)
    assert_raises_regex(ValueError, "must be 1, 2 or 3",
                        nm.fit_sample, X, Y)


def test_nm_wrong_nn_obj():
    print('--------------------------------------------hey')
    ratio = 'auto'
    nn = 'rnd'
    nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                  version=VERSION_NEARMISS,
                  return_indices=True,
                  n_neighbors=nn)
    assert_raises_regex(ValueError, "has to be one of",
                        nm.fit_sample, X, Y)

    # Create the object
    nn3 = 'rnd'
    nn = NearestNeighbors(n_neighbors=3)
    nm3 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=3, return_indices=True,
                   n_neighbors=nn, n_neighbors_ver3=nn3)
    assert_raises_regex(ValueError, "has to be one of",
                        nm3.fit_sample, X, Y)


def test_nm_fit_sample_auto():
    ratio = 'auto'
    X_gt = X_test

    y_gt =Y_test

    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                      version=version)
        X_resampled, y_resampled = nm.fit_sample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        return(assert_array_equal(y_resampled, y_gt[version_idx]))


def test_nm_fit_sample_auto_indices():
    ratio = 'auto'
    X_gt = [np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [1.17737838, -0.2002118],
                      [-0.60413357, 0.24628718],
                      [0.03142011, 0.12323596],
                      [1.15157493, -1.2981518],
                      [-0.54619583, 1.73009918],
                      [0.99272351, -0.11631728]])]

    y_gt = [np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])]

    idx_gt = [np.array([3, 10, 11, 2, 8, 5, 9, 1, 6]),
              np.array([3, 10, 11, 2, 8, 5, 9, 1, 6]),
              np.array([3, 10, 11, 0, 2, 3, 5, 1, 4])]

    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                      version=version, return_indices=True)
        X_resampled, y_resampled, idx_under = nm.fit_sample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])
        assert_array_equal(idx_under, idx_gt[version_idx])


def test_nm_fit_sample_float_ratio():
    ratio = .7

    X_gt = [np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [1.17737838, -0.2002118],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295],
                      [0.99272351, -0.11631728]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [1.17737838, -0.2002118],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295],
                      [0.99272351, -0.11631728]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [1.17737838, -0.2002118],
                      [-0.60413357, 0.24628718],
                      [0.03142011, 0.12323596],
                      [-0.05903827, 0.10947647],
                      [1.15157493, -1.2981518],
                      [-0.54619583, 1.73009918],
                      [0.99272351, -0.11631728],
                      [0.45713638, 1.31069295]])]

    y_gt = [np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])]

    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                      version=version)
        X_resampled, y_resampled = nm.fit_sample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])


def test_nm_fit_sample_nn_obj():
    ratio = 'auto'
    nn = NearestNeighbors(n_neighbors=3)
    X_gt = [np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [1.17737838, -0.2002118],
                      [-0.60413357, 0.24628718],
                      [0.03142011, 0.12323596],
                      [1.15157493, -1.2981518],
                      [-0.54619583, 1.73009918],
                      [0.99272351, -0.11631728]])]

    y_gt = [np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])]

    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                      version=version, n_neighbors=nn)
        X_resampled, y_resampled = nm.fit_sample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])
