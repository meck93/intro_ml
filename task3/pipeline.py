
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Binarizer, scale
from sklearn.feature_selection import chi2, f_classif, GenericUnivariateSelect, mutual_info_classif, RFE, RFECV, SelectPercentile, SelectFpr, SelectKBest
from sklearn.pipeline import make_pipeline

import numpy as np
import pandas as pd
import os

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import xgboost as xgb

# Constants
FILE_PATH_TRAIN = "./input/train.h5"
FILE_PATH_TEST = "./input/test.h5"
TEST_SIZE = 0.25

# read training file
# test_data = pd.read_hdf(FILE_PATH_TRAIN, "test")
training_data = pd.read_hdf(FILE_PATH_TRAIN, "train")

# training data
# extracting the x-values 
x_values_training = training_data.copy()
x_values_training = x_values_training.drop(labels=['y'], axis=1)
x_component_training = x_values_training.values

# extracting the y-values
y_component_training = training_data['y'].values

# splitting the training set into a training & validation set
x_train, x_val, y_train, y_val = train_test_split(x_component_training, y_component_training, test_size=TEST_SIZE)

# feature selection
f_selector = SelectKBest(f_classif, k=30).fit(x_component_training, y_component_training)
x_train_comp_new = f_selector.transform(x_component_training)

print("f_classif - k-best", x_train_comp_new.shape)
print(sorted(f_selector.scores_, reverse=True))
print()

model = xgb.XGBClassifier(max_depth=7, learning_rate=0.3, n_estimators=250, silent=False, objective='multi:softmax', booster='gbtree', n_jobs=1, min_child_weight=1, max_delta_step=0, subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, gamma=1)

# pipline
pipeline = make_pipeline(StandardScaler(with_mean=True, with_std=True), SelectKBest(f_classif, k=30), model)

# train the model
pipeline.fit(x_train, y_train)

# prediction on validation set
pred_val = pipeline.predict(x_val)

# current validation prediction accuracy
curr_val_acc = accuracy_score(y_val, pred_val)

print("XGBoost: Accuracy Score", curr_val_acc)
print()

