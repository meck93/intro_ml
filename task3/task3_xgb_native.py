
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
FILE_PATH_SAMPLE = "./input/sample.csv"
TEST_SIZE = 0.225

# read files
test_data = pd.read_hdf(FILE_PATH_TEST, "test")
training_data = pd.read_hdf(FILE_PATH_TRAIN, "train")
ids = pd.read_csv(FILE_PATH_SAMPLE, sep=',', header=0, usecols=['Id'])

# training data
# extracting the x-values 
x_values_training = training_data.copy()
x_values_training = x_values_training.drop(labels=['y'], axis=1)
x_component_training = x_values_training.values

# extracting the y-values
y_component_training = training_data['y'].values

# test data
# extracting the x-values
x_values_test = test_data.copy()
x_values_test = x_values_test.values

# training the scaler
scaler = StandardScaler(with_mean=True, with_std=True)
scaler = scaler.fit(x_component_training)

# scaling the training and test data
x_train_scaled = scaler.transform(x_component_training)
x_test_scaled = scaler.transform(x_values_test)

# feature selection 
selector = SelectKBest(f_classif, k=30)
selector = selector.fit(x_train_scaled, y_component_training)
x_train_scaled_new = selector.transform(x_train_scaled)
x_test_scaled_new = selector.transform(x_test_scaled)

# splitting the training set into a training & validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_scaled_new, y_component_training, test_size=TEST_SIZE)

# training, evaluation and test data in xgboost DMatrix
xg_train = xgb.DMatrix(x_train, label=y_train)
xg_val = xgb.DMatrix(x_val, label=y_val)
xg_test = xgb.DMatrix(x_test_scaled_new, label=None)

# setup parameters for xgboost
params = {}

# use softmax multi-class classification
params['objective'] = 'multi:softmax'

# scale weight of positive examples
params['silent'] = 1
params['num_class'] = 5
params['tree_method'] = 'auto'
params['seed'] = 42
params['eta'] = 0.3
params['lambda'] = 10.0
params['gamma'] = 0
params['max_depth'] = 9
params['min_child_weight'] = 8
params['subsample'] = 0.9
params['colsample_bytree'] = 0.8

# metrics to be watched on training and test data set
watchlist = [(xg_train, 'train'), (xg_val, 'test')]

# number of boosting rounds
rounds = 1000

# number of early stopping rounds
e_stop = 25

# evaluation metrics for cv
eval_metric = ['merror']

# train the xgboost model
bst = xgb.train(params=params, dtrain=xg_train, num_boost_round=rounds, evals=watchlist, early_stopping_rounds=e_stop, verbose_eval=True)

print("Best Score:", bst.best_score)
print("Best Iteration:", bst.best_iteration)
print("Best Tree Limit:", bst.best_ntree_limit)
print()

print("Best MAE: {:.2f} with {} rounds".format(bst.best_score, bst.best_iteration+1))

# get prediction on validation set
pred = bst.predict(xg_val)
error_rate = np.sum(pred != y_val) / y_val.shape[0]
print('Test error using softmax = {}'.format(error_rate))
print()

# computing error metrics
print("The first 5 real y-values:", y_val[0:5])
print("The first 5 y-value predictions", pred[0:5])

# current validation prediction accuracy
curr_val_acc = accuracy_score(y_val, pred)

print("XGBoost: Accuracy Score", curr_val_acc)
print()

# get prediction on validation set
test_pred = bst.predict(xg_test)

# preparing to write the coefficients to file
out = {"Id" : ids['Id'].values, "y": test_pred}

# output data frame
out = pd.DataFrame(data=out)

# printing test output
print()
print("Result Written To File:")
print(out.head(5))

# write to csv-file
out.to_csv("./output/xgb/task3_xgb_nativ_[4].csv", sep=';', index=False, header=True)
