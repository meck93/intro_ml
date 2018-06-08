from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import class_weight

import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Constants
FILE_PATH_TRAIN_LABELED = "./input/train_labeled.h5"
FILE_PATH_TRAIN_UNLABELED = "./input/train_unlabeled.h5"
FILE_PATH_COMBINED = "./input/combined_training_set.csv"
FILE_PATH_TEST = "./input/test.h5"

# read files
# data_type = {'y':np.int32, 'x1': np.float32, 'x2': np.float32, 'x3': np.float32, 'x4': np.float32, 'x5': np.float32, 'x6': np.float32, 'x7': np.float32, 'x8': np.float32, 'x9': np.float32, 'x10': np.float32, 'x11': np.float32, 'x12': np.float32, 'x13': np.float32, 'x14': np.float32, 'x15': np.float32, 'x16': np.float32, 'x17': np.float32, 'x18': np.float32, 'x19': np.float32, 'x20': np.float32, 'x21': np.float32, 'x22': np.float32, 'x23': np.float32, 'x24': np.float32, 'x25': np.float32, 'x26': np.float32, 'x27': np.float32, 'x28': np.float32, 'x29': np.float32, 'x30': np.float32, 'x31': np.float32, 'x32': np.float32, 'x33': np.float32, 'x34': np.float32, 'x35': np.float32, 'x36': np.float32, 'x37': np.float32, 'x38': np.float32, 'x39': np.float32, 'x40': np.float32, 'x41': np.float32, 'x42': np.float32, 'x43': np.float32, 'x44': np.float32, 'x45': np.float32, 'x46': np.float32, 'x47': np.float32, 'x48': np.float32, 'x49': np.float32, 'x50': np.float32, 'x51': np.float32, 'x52': np.float32, 'x53': np.float32, 'x54': np.float32, 'x55': np.float32, 'x56': np.float32, 'x57': np.float32, 'x58': np.float32, 'x59': np.float32, 'x60': np.float32, 'x61': np.float32, 'x62': np.float32, 'x63': np.float32, 'x64': np.float32, 'x65': np.float32, 'x66': np.float32, 'x67': np.float32, 'x68': np.float32, 'x69': np.float32, 'x70': np.float32, 'x71': np.float32, 'x72': np.float32, 'x73': np.float32, 'x74': np.float32, 'x75': np.float32, 'x76': np.float32, 'x77': np.float32, 'x78': np.float32, 'x79': np.float32, 'x80': np.float32, 'x81': np.float32, 'x82': np.float32, 'x83': np.float32, 'x84': np.float32, 'x85': np.float32, 'x86': np.float32, 'x87': np.float32, 'x88': np.float32, 'x89': np.float32, 'x90': np.float32, 'x91': np.float32, 'x92': np.float32, 'x93': np.float32, 'x94': np.float32, 'x95': np.float32, 'x96': np.float32, 'x97': np.float32, 'x98': np.float32, 'x99': np.float32, 'x100': np.float32, 'x101': np.float32, 'x102': np.float32, 'x103': np.float32, 'x104': np.float32, 'x105': np.float32, 'x106': np.float32, 'x107': np.float32, 'x108': np.float32, 
# 'x109': np.float32, 'x110': np.float32, 'x111': np.float32, 'x112': np.float32, 'x113': np.float32, 'x114': np.float32, 'x115': np.float32, 'x116': np.float32, 'x117': np.float32, 
# 'x118': np.float32, 'x119': np.float32, 'x120': np.float32, 'x121': np.float32, 'x122': np.float32, 'x123': np.float32, 'x124': np.float32, 'x125': np.float32, 'x126': np.float32, 
# 'x127': np.float32, 'x128': np.float32}

# train_data = pd.read_csv(FILE_PATH_COMBINED, dtype=np.float64)
# test_data = pd.read_hdf(FILE_PATH_TEST, "test")
train_labeled = pd.read_hdf(FILE_PATH_TRAIN_LABELED, "train", dtype=np.float64)
train_unlabeled = pd.read_hdf(FILE_PATH_TRAIN_UNLABELED, "train", dtype=np.float64)
train_unlabeled.insert(loc=0, column='y', value=-1)

# labeled training data: extracting the y-values - labels
y_labeled = train_labeled['y'].values

# fake labels for unlabled set
y_unlabeled = train_unlabeled['y'].values

# compute the class weights to balance the dataset
class_labels = np.unique(y_labeled)
class_weights = class_weight.compute_class_weight('balanced', class_labels, y_labeled)
class_weights = {label:weight for label, weight in zip(class_labels, class_weights)}
print(class_weights)

# labeled training data: extracting the x-values
X_labeled = train_labeled.drop(labels=['y'], axis=1)
X_labeled = X_labeled.values

# unlabeled training data: extracting the x-values
X_unlabeled = train_unlabeled.drop(labels=['y'], axis=1)
X_unlabeled = X_unlabeled.values

# test data: extracting the x-values
# x_test = test_data.values

# training the scaler
# scaler = StandardScaler(with_mean=True, with_std=True)
# scaler = scaler.fit(X)

# scaling the training and test data
# X = scaler.transform(X)
# x_test_scaled = scaler.transform(x_test)

# split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(X_labeled, y_labeled, test_size=0.2, shuffle=True)

# combine the labeled und unlabeled training data
x_train = np.append(x_train, X_unlabeled, axis=0)
y_train = np.append(y_train, y_unlabeled, axis=0)

from pomegranate import NaiveBayes, NormalDistribution, BayesClassifier
from pomegranate import GeneralMixtureModel, MultivariateGaussianDistribution

# model_a = NaiveBayes.from_samples(NormalDistribution, x_train[y_train != -1], y_train[y_train != -1], verbose=True, stop_threshold=0.001)
# print("Naive Bayes - Supervised Learning Accuracy: {}".format((model_a.predict(x_val) == y_val).mean()))

# model_b = NaiveBayes.from_samples(NormalDistribution, x_train, y_train, verbose=True, weights=None, stop_threshold=0.1, max_iterations=100)
# print("Naive Bayes - Semisupervised Learning Accuracy: {}".format((model_b.predict(x_val) == y_val).mean()))

model_c = BayesClassifier.from_samples(MultivariateGaussianDistribution, x_train, y_train, inertia=0.0, pseudocount=0.0, stop_threshold=0.1, max_iterations=100, verbose=True, n_jobs=1)
print("Bayes Classifier - Semisupervised Learning Accuracy: {}".format((model_c.predict(x_val) == y_val).mean()))

# general mixture model
d0 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 0])
d1 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 1])
d2 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 2])
d3 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 3])
d4 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 4])
d5 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 5])
d6 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 6])
d7 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 7])
d8 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 8])
d9 = GeneralMixtureModel.from_samples(NormalDistribution, 2, x_train[y_train == 9])

# bayes classifier
model_b = BayesClassifier([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])

# supervised learning
model_b.fit(x_train[y_train != -1], y_train[y_train != -1], inertia=0.0, pseudocount=0.0, stop_threshold=0.1, max_iterations=30, verbose=True, n_jobs=1)
print("GMM - NormalDistribution - Supervised Learning Accuracy: {}".format((model_b.predict(x_val) == y_val).mean()))

# semisupervised learning
model_b.fit(x_train, y_train, inertia=0.0, pseudocount=0.0, stop_threshold=0.1, max_iterations=30, verbose=True, n_jobs=1)
print("GMM - NormalDistribution - Semisupervised Learning Accuracy: {}".format((model_b.predict(x_val) == y_val).mean()))

# general mixture model
d0 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 0])
d1 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 1])
d2 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 2])
d3 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 3])
d4 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 4])
d5 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 5])
d6 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 6])
d7 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 7])
d8 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 8])
d9 = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, 2, x_train[y_train == 9])

# bayes classifier
model_b = BayesClassifier([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9])

# supervised learning
model_b.fit(x_train[y_train != -1], y_train[y_train != -1], inertia=0.0, pseudocount=0.0, stop_threshold=0.1, max_iterations=30, verbose=True, n_jobs=1)
print("GMM - Multivariate Gaussian - Supervised Learning Accuracy: {}".format((model_b.predict(x_val) == y_val).mean()))

# semisupervised learning
model_b.fit(x_train, y_train, inertia=0.0, pseudocount=0.0, stop_threshold=0.1, max_iterations=30, verbose=True, n_jobs=1)
print("GMM - Multivariate Gaussian - Semisupervised Learning Accuracy: {}".format((model_b.predict(x_val) == y_val).mean()))

# # predict the result on the validation dataset
# y_pred = model_b.predict(x_val)

# # calculate the accuracy of the prediction
# acc = accuracy_score(y_val, y_pred)
# print("Validation accuracy:", acc)

