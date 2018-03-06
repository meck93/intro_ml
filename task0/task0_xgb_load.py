from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import math
import numpy as np
import pandas as pd

import os

mingw_path = ";C:\Program Files\mingw-w64\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\mingw64\bin"
os.environ['PATH'] = os.environ['PATH'] + mingw_path

import xgboost as xgb

# personal csv reader module
import reader

FILE_PATH_TRAIN = "train.csv"
FILE_PATH_TEST = "test.csv"

# read training file
test = reader.read_csv(FILE_PATH_TEST)

# splitting the training data set into x and y components
test_data = test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
x_test = test_data.values

# how to load
loaded_model = joblib.load("trained_model.joblib.dat")

# predicting the y-values of x_val
predictions = loaded_model.predict(x_test)
print("The first 5 predictions", predictions[0:5])

means = []

for row in x_test:
    means.append(np.mean(row))

# computing error metrics
# print("XGBoost: Explained Variance Score", explained_variance_score(means, predictions))
# print("XGBoost: Mean Square Error", math.sqrt(mean_squared_error(means, predictions)))
# print("XGBoost: R^2", r2_score(means, predictions))

data = {'Id': test['Id'].values, 'y': predictions}
test_output = pd.DataFrame(data=data)

print(test_output.head(5))

test_output.to_csv("xgb_reg-linear_10-fold-cv_5-repeats.csv", sep=',', index=False)

