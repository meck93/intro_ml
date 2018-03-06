import os

mingw_path = ";C:\Program Files\mingw-w64\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\mingw64\bin"
os.environ['PATH'] = os.environ['PATH'] + mingw_path

import xgboost

import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn import cross_validation, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score

# personal csv reader module
import reader

FILE_PATH_TRAIN = "train.csv"
FILE_PATH_TEST = "test.csv"

# read training file
test = reader.read_csv(FILE_PATH_TEST)
data = reader.read_csv(FILE_PATH_TRAIN)

# splitting the test data set
new_test_data = test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]

# splitting the training data set into x and y components
new_train_data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
X = new_train_data.values
Y = data['y'].values

# splitting the training set into a training & validation set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# xgboost variant
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.1, gamma=0,
                           subsample=0.8, colsample_bytree=1, max_depth=5)

xgb.fit(x_train, y_train)

predictions = xgb.predict(x_test)
print("XGBoost:", explained_variance_score(predictions, y_test))