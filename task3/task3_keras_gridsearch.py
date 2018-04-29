from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.feature_selection import f_classif, SelectKBest

import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2, l1_l2

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Constants
FILE_PATH_TRAIN = "./input/train.h5"

# Validation Size
TEST_SIZE = 0.2

# read files
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

x_train, x_val, y_train, y_val = train_test_split(x_train_scaled, y_component_training, test_size=TEST_SIZE)

# define baseline model
def baseline_model(opt='adam', kernel_init='glorot_uniform', l2_lambda=None):
    # create model
    model = Sequential()
    model.add(Dense(units=100, input_dim=100, activation='relu', kernel_regularizer=l2_lambda, kernel_initializer=kernel_init))
    model.add(Dense(units=5, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# parameters
optimizers = [Adam(), SGD(lr=0.01, momentum=0.9, nesterov=False)]
inits = ['glorot_uniform', 'normal', 'uniform']
epochs = [50]
batches = [32, 64, 128]
l2_lambdas = [l2(0.1), l2(0.01), l2(0.001), None]

# parameter grid
param_grid = dict(opt=optimizers, epochs=epochs, kernel_init=inits, batch_size=batches, l2_lambda=l2_lambdas)

# build the model
model = KerasClassifier(build_fn=baseline_model, verbose=1)

# grid seracher
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=folds, return_train_score=True)
grid_result = grid.fit(x_train, y_train)

y_pred = grid.predict(x_val)
acc = accuracy_score(y_val, y_pred)

# extract the results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

with open('gs_all_result.txt', 'w+') as file:  
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    file.write("Best: %f using %s\n\n" % (grid_result.best_score_, grid_result.best_params_))

    print("Accuracy on Validation Set:", acc)
    file.write("Accuracy on Validation Set: %f \n\n" % (acc))

    for mean, stdev, param in zip(means, stds, params):
        print("mean test score: %f std test score: (%f)\nwith: %r\n\n" % (mean, stdev, param))
        file.write("mean test score: %f std test score: (%f)\nwith: %r\n\n" % (mean, stdev, param))
