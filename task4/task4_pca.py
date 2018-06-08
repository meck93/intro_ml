from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.utils import class_weight

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Constants
FILE_PATH_TRAIN_LABELED = "./input/train_labeled.h5"
FILE_PATH_TRAIN_UNLABELED = "./input/train_unlabeled.h5"
FILE_PATH_TEST = "./input/test.h5"

# read files
train_labeled = pd.read_hdf(FILE_PATH_TRAIN_LABELED, "train")
train_unlabeled = pd.read_hdf(FILE_PATH_TRAIN_UNLABELED, "train")
# test_data = pd.read_hdf(FILE_PATH_TEST, "test")

# labeled training data: extracting the y-values - labels
y_train_labeled = train_labeled['y'].values

# compute the class weights to balance the dataset
class_labels = np.unique(y_train_labeled)
class_weights = class_weight.compute_class_weight('balanced', class_labels, y_train_labeled)
class_weights = {label:weight for label, weight in zip(class_labels, class_weights)}

# labeled training data: extracting the x-values
x_train_labeled = train_labeled.drop(labels=['y'], axis=1)
x_train_labeled = x_train_labeled.values

# unlabeled training data: extracting the x-values
x_train_unlabeled = train_unlabeled.values

# training the scaler
scaler = StandardScaler(with_mean=True, with_std=True)
scaler = scaler.fit(x_train_labeled)

# scaling the training and test data
x_train_labeled_scaled = scaler.transform(x_train_labeled)
x_train_unlabeled_scaled = scaler.transform(x_train_unlabeled)

# model
model = KernelPCA(n_components=None, kernel='rbf', gamma=None, degree=3, coef0=1, 
                  kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', 
                  tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=1)

# training
model = model.fit(x_train_unlabeled)
predictions = model.transform(x_train_unlabeled)
print(predictions)

# Plot results
plt.figure()
plt.subplot(2, 2, 1, aspect='equal')
plt.title("Original space")
plt.hist(predictions)