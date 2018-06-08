from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam, RMSprop, Adamax
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import save_model, load_model, model_from_json

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# test size
TEST_SIZE = 0.2

# Constants
FILE_PATH_SAMPLE = "./input/sample.csv"
FILE_PATH_TRAIN_LABELED = "./input/train_labeled.h5"
FILE_PATH_TRAIN_UNLABELED = "./input/train_unlabeled.h5"
FILE_PATH_TEST = "./input/test.h5"

# read files
train_labeled = pd.read_hdf(FILE_PATH_TRAIN_LABELED, "train")
train_unlabeled = pd.read_hdf(FILE_PATH_TRAIN_UNLABELED, "train")
test_data = pd.read_hdf(FILE_PATH_TEST, "test")
ids = pd.read_csv(FILE_PATH_SAMPLE, sep=',', header=0, usecols=['Id'])

# labeled training data: extracting the y-values - labels
y_train_labeled = train_labeled['y'].values

# compute the class weights to balance the dataset
class_labels = np.unique(y_train_labeled)
class_weights = class_weight.compute_class_weight('balanced', class_labels, y_train_labeled)
class_weights = {label:weight for label, weight in zip(class_labels, class_weights)}

# one hot encoding for labels
encoded_y = np_utils.to_categorical(y_train_labeled, 10)

# labeled training data: extracting the x-values
x_train_labeled = train_labeled.drop(labels=['y'], axis=1)
x_train_labeled = x_train_labeled.values

# unlabeled training data: extracting the x-values
x_train_unlabeled = train_unlabeled.values

# test data: extracting the x-values
x_test = test_data.values

# training the scaler
scaler = StandardScaler(with_mean=True, with_std=True)
scaler = scaler.fit(x_train_labeled)

# scaling the training and test data
x_train_labeled_scaled = scaler.transform(x_train_labeled)
x_train_unlabeled_scaled = scaler.transform(x_train_unlabeled)

# define baseline model
def baseline_model(opt='sgd'):
    # create model
    model = Sequential()

    # input layer
    model.add(Dense(units=128, input_dim=128, activation='relu'))

    # hidden layers
    model.add(Dense(units=256, input_dim=128, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(Dense(units=256, input_dim=128, activation='relu', kernel_regularizer=l2(0.0001)))
#     model.add(Dropout(0.1))
    model.add(Dense(units=256, input_dim=128, activation='relu', kernel_regularizer=l2(0.0001)))

    # output layer
    model.add(Dense(units=10, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

# split the data into training and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_labeled_scaled, encoded_y, test_size=TEST_SIZE)

optimizers = [Adam(), SGD(lr=0.01, momentum=0.9, nesterov=False)]
names = ['Adam', 'SGD']

for optimizer in optimizers:
    # current model name
    curr_model_name = names[optimizers.index(optimizer)]

    # filepath to load the model
    in_file = './input/keep_Adam_[4].hdf5'
    
    # predict the test values
    best_model = load_model(in_file)
    test_pred = best_model.predict(x_train_unlabeled)

    # convert one hot back to categorical
    y_pred = np.argmax(test_pred, axis=1) 
    
    # extend unlabeled training data set with labeled column
    train_unlabeled.insert(loc=0, column='y', value=y_pred)
    
    # create temporary training data set
    x_train_temp = pd.concat([train_labeled, train_unlabeled])
    print(x_train_temp.shape)

    # new x and y values
    y_values = x_train_temp['y'].values
    y_train_values = np_utils.to_categorical(y_values, 10)
    x_train_values = x_train_temp.drop(labels=['y'], axis=1).values

    # scaling
    scaler = scaler.fit(x_train_values)
    x_train_values_scaled = scaler.transform(x_train_values)
    x_test_scaled = scaler.transform(x_test)

    # build the model
    estimator_2 = KerasClassifier(build_fn=baseline_model, 
                                opt=optimizer, 
                                epochs=500, 
                                batch_size=128, 
                                verbose=1)

    # early stop callback function
    e_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.0001, 
                           patience=20, 
                           verbose=0, 
                           mode='auto')

    # model checkpoint callback function
    filename = '{}_[4]'.format(curr_model_name)
    filepath = './output/model/model-checkpoint_{}.hdf5'.format(filename)
    filepath = os.path.abspath(filepath)
    m_check = ModelCheckpoint(filepath, 
                              monitor='val_loss', 
                              save_best_only=True, 
                              mode='auto', 
                              period=1)

    # split the data into training and validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train_values_scaled, y_train_values, test_size=TEST_SIZE)
    
    # train the model
    estimator_2.fit(x=x_train, 
                  y=y_train, 
                  validation_data=(x_val, y_val), 
                  shuffle=True, 
                  callbacks=[e_stop, m_check], 
                  class_weight=class_weights)
    
    # predict the test values
    best_model = load_model(filepath)
    test_pred = best_model.predict(x_test_scaled)

    # preparing to write the coefficients to file
    out = {"Id" : ids['Id'].values, "y": y_pred}
    
    # output data frame
    out = pd.DataFrame(data=out)

    # printing test output
    print()
    print("Result Written To File:")
    print(out.head(5))

    # write to csv-file
    output_filename = "./output/csv/prediction_{}.csv".format(filename)
    output_filename = os.path.abspath(output_filename)
    out.to_csv(output_filename, sep=',', index=False, header=True)
    
    