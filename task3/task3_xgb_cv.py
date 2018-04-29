
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectKBest

import numpy as np
import pandas as pd
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

# Constants
FILE_PATH_TRAIN = "./input/train.h5"
FILE_PATH_TEST = "./input/test.h5"
TEST_SIZE = 0.25

# read training file
# test_data = pd.read_hdf(FILE_PATH_TRAIN, "test")
training_data = pd.read_hdf(FILE_PATH_TRAIN, "train")

# training data
# extracting the x-values 
x_values_training = training_data.copy()
x_values_training = x_values_training.drop(labels=['y'], axis=1)
x_component_training = x_values_training.values

# extracting the y-values
y_component_training = training_data['y'].values

# training the scaler
scaler = StandardScaler(with_mean=True, with_std=True)
scaler = scaler.fit(x_component_training)

# scaling the training and test data
x_train_scaled = scaler.transform(x_component_training)

# feature selection 
selector = SelectKBest(f_classif, k=25)
selector = selector.fit(x_train_scaled, y_component_training)
x_train_scaled_new = selector.transform(x_train_scaled)

# splitting the training set into a training & validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_scaled_new, y_component_training, test_size=TEST_SIZE, random_state=42)

# training, evaluation and test data in xgboost DMatrix
xg_train = xgb.DMatrix(x_train, label=y_train)
xg_val = xgb.DMatrix(x_val, label=y_val)

# setup parameters for xgboost
params = {}

# use softmax multi-class classification
params['objective'] = 'multi:softmax'

# scale weight of positive examples
params['silent'] = 0
params['num_class'] = 5
params['tree_method'] = 'auto'
params['seed'] = 42

# number of boosting rounds
rounds = 300

# gridsearch_params = [
#     (max_depth, min_child_weight)
#     for max_depth in range(6,13,2)
#     for min_child_weight in range(4,9,2)
# ]

# print(gridsearch_params)

# best_params = None
# min_error = float("Inf")

# for max_depth, min_child_weight in gridsearch_params:
#     print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))

#     # Update our parameters
#     params['max_depth'] = max_depth
#     params['min_child_weight'] = min_child_weight

#     # Run CV
#     cv_results = xgb.cv(params, xg_train, num_boost_round=rounds, seed=42, nfold=5, metrics={'merror'}, early_stopping_rounds=10, verbose_eval=True)

#     # Update best error
#     mean_error = cv_results['test-merror-mean'].min()
#     boost_rounds = cv_results['test-merror-mean'].argmin()

#     print("\t Multiclass Error {} for {} rounds".format(mean_error, boost_rounds))
#     print()

#     if mean_error < min_error:
#         min_error = mean_error
#         best_params = (max_depth, min_child_weight)

# print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_error))

# # grid search parameters
# gridsearch_params = []

# # tree depth, gamma, learning rate, regularization lambda
# for max_tree_depth in range(6, 11, 1):
#     for gamma in range(0, 13, 2):
#         for learn_rate in [0.3, 0.1, 0.05]:
#             for reg_lambda in [10.0, 1.0, 0.0, 0.1, 0.01]:
#                 gridsearch_params.append((max_tree_depth, gamma, learn_rate, reg_lambda))

# print(gridsearch_params)

gridsearch_params = [
    (max_depth, gamma)
    for max_depth in range(6,13,2)
    for gamma in range(0,13,2)
]

print(gridsearch_params)

best_params = None
min_test_error = float("Inf")
min_train_error = float("Inf")

file = open("output.txt", mode="w+", encoding='utf-8', newline='\n')

for max_depth, gamma in gridsearch_params:
    print("CV with max_depth={}, gamma={}".format(max_depth, gamma))
    file.write("CV with max_depth={}, gamma={}\n".format(max_depth, gamma))

    # Update our parameters
    params['max_depth'] = max_depth
    params['gamma'] = gamma

    # Run CV
    cv_results = xgb.cv(params, xg_train, num_boost_round=rounds, seed=42, nfold=5, metrics={'merror'}, early_stopping_rounds=10, verbose_eval=True)

    # Update best error
    test_error = cv_results['test-merror-mean'].min()
    train_error = cv_results['train-merror-mean'].min()
    boost_rounds = cv_results['test-merror-mean'].argmin()

    print("Multiclass Error {} for {} rounds".format(test_error, boost_rounds))
    print()

    file.write("Multiclass Error - Test: {} - Train: {} for {} rounds\n".format(test_error, train_error, boost_rounds))
    file.write("\n")
    
    if test_error < min_test_error:
        min_test_error = test_error
        min_train_error = train_error
        best_params = (max_depth, gamma)

print("Best params: {}, {}, Test Error: {}, Train Error: {}".format(best_params[0], best_params[1], min_test_error, min_train_error))
file.write("Best params: {}, {}, Test Error: {}, Train Error: {}\n".format(best_params[0], best_params[1], min_test_error, min_train_error))
file.close()
