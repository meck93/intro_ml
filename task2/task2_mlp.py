from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

# scaling the x value components
scaler = StandardScaler()
scaler = scaler.fit(x_component_training)
x_train_scaled = scaler.transform(x_component_training)

# splitting the training set into a intermediate (training + validation) & test set
x_inter, x_test, y_inter, y_test = train_test_split(x_train_scaled, y_component_training, test_size=TEST_SIZE)

# create the classification model
mlp = MLPClassifier(activation='relu', alpha=0.1, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
                    hidden_layer_sizes=100, learning_rate='constant', learning_rate_init=0.001, max_iter=500, momentum=0.9, 
                    nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='lbfgs', tol=1e-07, 
                    validation_fraction=0.1, verbose=False, warm_start=False)

# fit the training data
mlp.fit(x_inter, y_inter)

# predicting the y-values of x_val
y_pred = mlp.predict(x_test)

# compare real vs prediction
print("The first 5 real y-values:", y_test[0:5])
print("The first 5 y-value predictions", y_pred[0:5])
# print("The probability for the first 5 x-values:", mlp.predict_proba(x_test[0:5]))

# computing error metrics
print("Accuracy Score", accuracy_score(y_test, y_pred))
