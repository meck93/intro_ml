from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.externals import joblib

import numpy as np
import math
import sys

import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

# personal csv reader module
import reader

FILE_PATH_TRAIN = "train.csv"
FILE_PATH_TEST = "test.csv"
TEST_SIZE = 0.25

# read training file
training_data = reader.read_csv(FILE_PATH_TRAIN, show_info=False)

# splitting the training data set into x and y components
data_columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']

# extracting the x-values 
x_values_training = training_data[data_columns]
x_component_training = x_values_training.values

# extracting the y-values
y_component_training = training_data['y'].values

# scaling the x value components
scaler = StandardScaler()
scaler = scaler.fit(x_component_training)
x_train_scaled = scaler.transform(x_component_training)

# splitting the training set into a training & validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_scaled, y_component_training, test_size=TEST_SIZE)

# create the classification model
boost = xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=500, silent=True, 
                          objective='multi:softmax', booster='gbtree', 
                          n_jobs=4, gamma=0, min_child_weight=1, max_delta_step=0, 
                          subsample=1.0, colsample_bytree=1, colsample_bylevel=1, 
                          reg_alpha=0.0, reg_lambda=1, scale_pos_weight=1, base_score=0.5)

# evaluation metrics
evals = ['mlogloss', 'merror']

# evaluation set to be validate against
watchlist = [(x_val, y_val)]

# fit the training data
boost.fit(x_train, y_train, eval_set=watchlist, eval_metric=evals, early_stopping_rounds=50, verbose=True)

# predicting the y-values of x_val
y_pred = boost.predict(x_val)

print("The first 5 real y-values:", y_val[0:5])
print("The first 5 y-value predictions", y_pred[0:5])

# computing error metrics
print("XGBoost: Accuracy Score", accuracy_score(y_val, y_pred))
