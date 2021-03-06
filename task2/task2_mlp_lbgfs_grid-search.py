from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import numpy as np
import math
import sys

# personal csv reader module
import reader

FILE_PATH_TRAIN = "./input/train.csv"
FILE_PATH_TEST = "./input/test.csv"
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

# scaling the x value components
scaler = StandardScaler()
scaler = scaler.fit(x_component_training)
x_train_scaled = scaler.transform(x_component_training)

# splitting the training set into a intermediate (training + validation) & test set
x_inter, x_test, y_inter, y_test = train_test_split(x_train_scaled, y_component_training, test_size=TEST_SIZE)

# create the classification model
mlp = MLPClassifier(activation='relu', solver='lbfgs', batch_size='auto', shuffle=True, verbose=False, warm_start=False)

# parameters to be optimzed
params = [{'hidden_layer_sizes': (10, 10), 'tol': [1e-5, 1e-6, 1e-7], 'alpha': [1e0, 1e-1, 1e-2]}]

# 10 fold cv
repeated_ten_fold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=None)

# discover the best alpha value
searcher = GridSearchCV(estimator=mlp, param_grid=params, scoring='accuracy', n_jobs=1, iid=True, refit=True, cv=repeated_ten_fold, verbose=1, pre_dispatch='2*n_jobs', error_score='raise', return_train_score='warn')

# fit the training data
searcher.fit(x_inter, y_inter)
print(sorted(searcher.cv_results_))
print(searcher.best_estimator_)

# predicting the y-values of x_val
y_pred = searcher.predict(x_test)

# compare real vs prediction
print("The first 5 real y-values:", y_test[0:5])
print("The first 5 y-value predictions", y_pred[0:5])

# computing error metrics
print("Accuracy Score", accuracy_score(y_test, y_pred))
