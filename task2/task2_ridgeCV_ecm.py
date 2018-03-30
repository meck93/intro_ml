from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
import math
import sys

# personal csv reader module
import reader

FILE_PATH_TRAIN = "train.csv"
FILE_PATH_TEST = "test.csv"
TEST_SIZE = 0.225

# read training file
training_data = reader.read_csv(FILE_PATH_TRAIN, show_info=False)

# splitting the training data set into x and y components
data_columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16']

# extracting the x-values 
x_values_training = training_data[data_columns]
x_component_training = x_values_training.values

# extracting the y-values
y_component_training = training_data['y'].values

# splitting the training set into a intermediate (training + validation) & test set
x_inter, x_test, y_inter, y_test = train_test_split(x_component_training, y_component_training, test_size=TEST_SIZE)

# alpha parameter values
alphas = [10.0e-4, 10.0e-3, 10.0e-2, 10.0e-1, 10.0e0, 10.0, 100.0]

# repeated ten-fold cross validation
repeated_ten_fold = RepeatedKFold(n_splits=10, n_repeats=5)

# create the classification model
ridgeCV = RidgeClassifierCV(alphas=alphas, fit_intercept=True, normalize=False, cv=repeated_ten_fold)

# fit the training data
ridgeCV.fit(x_inter, y_inter)
print("Alpha Value:", ridgeCV.alpha_)

# predicting the y-values of x_val
y_pred = ridgeCV.predict(x_test)

# compare real vs prediction
print("The first 5 real y-values:", y_test[0:5])
print("The first 5 y-value predictions", y_pred[0:5])

# computing error metrics
print("Accuracy Score", accuracy_score(y_test, y_pred))
