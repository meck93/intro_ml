from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import class_weight

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

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

# test data: extracting the x-values
# x_test = test_data.values

# training the scaler
# scaler = StandardScaler(with_mean=True, with_std=True)
# scaler = scaler.fit(x_train_labeled)

# scaling the training and test data
# x_train_labeled_scaled = scaler.transform(x_train_labeled)
# x_test_scaled = scaler.transform(x_test)

model_a = GaussianMixture(n_components=100, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=1, verbose_interval=10)

model_b = BayesianGaussianMixture(n_components=100, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None, covariance_prior=None, random_state=None, warm_start=False, verbose=1, verbose_interval=10)

model_a = model_a.fit(x_train_unlabeled)
result = model_a.predict(x_train_unlabeled)
print("Gaussian Mixture")
print(result)

model_b = model_b.fit(x_train_unlabeled)
result = model_b.predict(x_train_unlabeled)
print("Bayesian Gaussain Mixture")
print(result)

