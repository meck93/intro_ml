from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import math
import sys

# personal csv reader module
import reader

FILE_PATH_TRAIN = "train.csv"
FILE_PATH_TEST = "test.csv"
TEST_SIZE = 0.225

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

# create the classification model
mlp = MLPClassifier(activation='relu', alpha=1.0, batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=100, 
                    learning_rate='constant', learning_rate_init=0.001, max_iter=250, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=None, 
                    shuffle=True, solver='lbfgs', tol=1e-07, validation_fraction=0.25, verbose=False, warm_start=False)

# initial training accuracy 
acc = 0.1

while acc < 0.895:
    # splitting the training set into a training & validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train_scaled, y_component_training, test_size=TEST_SIZE)

    # fit the training data
    mlp.fit(x_train, y_train)

    # predicting the y-values of x_val
    y_pred = mlp.predict(x_val)

    # compare real vs prediction
    print("The first 5 real y-values:", y_val[0:5])
    print("The first 5 y-value predictions", y_pred[0:5])

    # computing error metrics
    acc = accuracy_score(y_val, y_pred)
    print("Accuracy Score", acc)

# get prediction on validation set
test_pred = mlp.predict(x_test_scaled)

# preparing to write the coefficients to file
out = {"Id" : test_data['Id'], "y": test_pred}

# output data frame
out = pd.DataFrame(data=out, dtype=np.int16)

# printing test output
print()
print("Result Written To File:")
print(out.head(5))

# write to csv-file
# out.to_csv("./output/mlp_lbgfs/task2_mlp_lbgfs_[6].csv", sep=',', index=False, header=True)
