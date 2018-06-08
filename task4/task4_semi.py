from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

from sklearn.model_selection import StratifiedKFold

from sklearn.decomposition import PCA
from sklearn.utils import class_weight

from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel

from sklearn.semi_supervised import LabelPropagation, LabelSpreading

import numpy as np
import pandas as pd
import os

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Constants
FILE_PATH_COMBINED = "./input/combined_training_set.csv"
FILE_PATH_TEST = "./input/test.h5"

# read files
train_data = pd.read_csv(FILE_PATH_COMBINED)
# test_data = pd.read_hdf(FILE_PATH_TEST, "test")

# labeled training data: extracting the y-values - labels
y_train_labeled = train_data['y'].values

# compute the class weights to balance the dataset
class_labels = np.unique(y_train_labeled)
class_weights = class_weight.compute_class_weight('balanced', class_labels, y_train_labeled)
class_weights = {label:weight for label, weight in zip(class_labels, class_weights)}

# labeled training data: extracting the x-values
x_train_labeled = train_data.drop(labels=['y'], axis=1)
x_train_labeled = x_train_labeled.values

# test data: extracting the x-values
# x_test = test_data.values

# training the scaler
scaler = StandardScaler(with_mean=True, with_std=True)
scaler = scaler.fit(x_train_labeled)

# scaling the training and test data
x_train_labeled_scaled = scaler.transform(x_train_labeled)
# x_test_scaled = scaler.transform(x_test)

# stratified ten fold cross validation
cv = StratifiedKFold(n_splits=10, shuffle=False, random_state=seed)

# setup the model
for train_index, val_index in cv.split(x_train_labeled_scaled, y_train_labeled):
    # create training and validation splits
    x_train, x_val = x_train_labeled_scaled[train_index], x_train_labeled_scaled[val_index]
    y_train, y_val = y_train_labeled[train_index], y_train_labeled[val_index]

    # my_kernel = polynomial_kernel(x_train, y_train, degree=5, gamma=None, coef0=1)

    # create model and fit data
    model = LabelSpreading(kernel=polynomial_kernel, gamma=20, alpha=0.2, max_iter=1, tol=0.001, n_jobs=1)
    model = model.fit(x_train, y_train)

    # evaluate model
    y_pred = model.predict(x_val)
    acc = accuracy_score(y_val, y_pred)
    print("Model Result: Split {} - Acc: {}".format(train_index, acc))


