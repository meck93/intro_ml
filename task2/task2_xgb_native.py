
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

# personal csv reader module
import reader

FILE_PATH_TRAIN = "./input/train.csv"
FILE_PATH_TEST = "./input/test.csv"
TEST_SIZE = 0.25

# read training file
test_data = reader.read_csv(FILE_PATH_TEST, show_info=False)
training_data = reader.read_csv(FILE_PATH_TRAIN, show_info=False)

# splitting the training data set into x and y components
data_columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']

# test data
# extracting the x-values
x_values_test = test_data[data_columns]
x_values_test = x_values_test.values

# training data
# extracting the x-values 
x_values_training = training_data[data_columns]
x_component_training = x_values_training.values

# extracting the y-values
y_component_training = training_data['y'].values

# training the scaler
scaler = StandardScaler(with_mean=True, with_std=True)
scaler = scaler.fit(x_component_training)

# scaling the training and test data
x_train_scaled = scaler.transform(x_component_training)
x_test_scaled = scaler.transform(x_values_test)

# splitting the training set into a training & validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_scaled, y_component_training, test_size=TEST_SIZE)

# training, evaluation and test data in xgboost DMatrix
xg_train = xgb.DMatrix(x_train, label=y_train)
xg_val = xgb.DMatrix(x_val, label=y_val)
xg_test = xgb.DMatrix(x_test_scaled, label=None)

# setup parameters for xgboost
param = {}

# use softmax multi-class classification
param['objective'] = 'multi:softmax'

# scale weight of positive examples
param['silent'] = 0
param['nthread'] = 4
param['num_class'] = 3
param['tree_method'] = 'exact'
param['subsample'] = 0.8

# regularization
param['alpha'] = 0
param['lambda'] = 1
param['gamma'] = 0
param['eta'] = 0.3
param['max_depth'] = 6

# metrics to be watched on training and test data set
watchlist = [(xg_train, 'train'), (xg_val, 'test')]

# number of boosting rounds
rounds = 1000
# number of early stopping rounds
e_stop = 100

# evaluation metrics for cv
eval_metric = ['mlogloss', 'merror']

# train the xgboost model
bst = xgb.train(params=param, dtrain=xg_train, num_boost_round=rounds, evals=watchlist, early_stopping_rounds=e_stop, verbose_eval=True)

print("Best Score:", bst.best_score)
print("Best Iteration:", bst.best_iteration)
print("Best Tree Limit:", bst.best_ntree_limit)
print()

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
out = {"Id" : test_data['Id'], "y": test_pred}

# output data frame
out = pd.DataFrame(data=out, dtype=np.int16)

# printing test output
print()
print("Result Written To File:")
print(out.head(5))

# write to csv-file
# out.to_csv("./output/task2_xgb_nativ_[9].csv", sep=',', index=False, header=True)
