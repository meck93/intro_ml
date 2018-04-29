from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, LabelBinarizer
from sklearn.feature_selection import f_classif, SelectKBest

import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2, l1_l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import save_model, load_model, model_from_json

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Constants
FILE_PATH_TRAIN = "./input/train.h5"
FILE_PATH_TEST = "./input/test.h5"
FILE_PATH_SAMPLE = "./input/sample.csv"
TEST_SIZE = 0.2

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

# one hot encoding for labels
encoded_y = np_utils.to_categorical(y_component_training, 5)

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

# define baseline model
def baseline_model(opt='sgd'):
    # create model
    model = Sequential()

    # input layer
    model.add(Dense(units=100, input_dim=100, activation='relu'))

    # hidden layers
    model.add(Dense(units=100, input_dim=100, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(units=100, input_dim=100, activation='relu', kernel_regularizer=l2(0.01)))

    # output layer
    model.add(Dense(units=5, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

# split the data into training and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_scaled, encoded_y, test_size=TEST_SIZE)

optimizers = [Adam(), SGD(lr=0.01, momentum=0.9, nesterov=False)]
names = ['Adam', 'SGD']

for optimizer in optimizers:
    # current model name
    curr_model_name = names[optimizers.index(optimizer)]

    # build the model
    estimator = KerasClassifier(build_fn=baseline_model, opt=optimizer, epochs=200, batch_size=128, verbose=1)

    # callback functions
    e_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, mode='auto')

    filepath = os.path.abspath('./output/keras/model_check/{}_check[5].hdf5'.format(curr_model_name))
    m_check = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='auto', period=1)

    # train the model
    estimator.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), shuffle=True, callbacks=[e_stop, m_check])

    # predict the test values
    best_model = load_model(filepath)
    test_pred = best_model.predict(x_test_scaled)

    # convert one hot back to categorical
    y_pred = np.argmax(test_pred, axis=1)

    # preparing to write the coefficients to file
    out = {"Id" : ids['Id'].values, "y": y_pred}

    # output data frame
    out = pd.DataFrame(data=out)

    # printing test output
    print()
    print("Result Written To File:")
    print(out.head(5))

    # write to csv-file
    out.to_csv("./output/keras/dropout/best_{}_3-layers_1-dropout[5].csv".format(curr_model_name), sep=',', index=False, header=True)

