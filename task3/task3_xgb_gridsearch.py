
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
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
TEST_SIZE = 0.225

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
params['silent'] = 1
params['num_class'] = 5
params['tree_method'] = 'auto'
params['seed'] = 42
# params['eta'] = 0.3
params['lambda'] = 10.0
params['gamma'] = 0
params['max_depth'] = 9
params['min_child_weight'] = 8
params['subsample'] = 0.9
params['colsample_bytree'] = 0.8

# number of boosting rounds
rounds = 3000

# 10 fold cv
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBooster Model
bst = xgb.Booster(params=params)

# grid search parameters
gridsearch_params = [0.01, 0.005]

print(gridsearch_params, "length:", len(gridsearch_params))

best_params = None
min_test_error = float("Inf")
min_train_error = float("Inf")

file = open("cv_gridsearch_learning-rate.txt", mode="w+", encoding='utf-8', newline='\n')

for eta in gridsearch_params:
    print("CV with eta={}".format(eta))
    file.write("CV with eta={}\n".format(eta))

    # Update our parameters
    params['eta'] = eta

    # Run CV
    cv_results = xgb.cv(params, xg_train, num_boost_round=rounds, seed=42, nfold=5, metrics={'merror'}, early_stopping_rounds=25, verbose_eval=True)

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
        best_params = eta

print("Best params: eta {}, Test Error: {}, Train Error: {}".format(best_params, min_test_error, min_train_error))
file.write("Best params: eta {}, Test Error: {}, Train Error: {}\n".format(best_params, min_test_error, min_train_error))
file.close()
