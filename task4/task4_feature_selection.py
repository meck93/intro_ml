from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler, QuantileTransformer, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2

from sklearn.decomposition import PCA
from sklearn.utils import class_weight

from sklearn.ensemble import ExtraTreesClassifier

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

# 2nd scaler option 
scale = RobustScaler()
scale = scale.fit(x_train_labeled)

#scale the training data
x_train_labeled_robust_scaled = scale.transform(x_train_labeled)

# normalization 
norm = Normalizer()
norm = norm.fit(x_train_labeled)

# normalizing the training data set
x_train_labeled_normalized = norm.transform(x_train_labeled)

# feature selection - robust scaling
print("-------------------------------------")
print("Robust Scaler")

# select k-best - f classif
k_best_f = SelectKBest(score_func=f_classif, k='all')
k_best_f = k_best_f.fit(x_train_labeled_robust_scaled, y_train_labeled)
print("K-Best F-Classif Scores:", sorted(k_best_f.scores_, reverse=True))
print()

# select k-best - f mutual info classif
k_best_m = SelectKBest(score_func=mutual_info_classif, k='all')
k_best_m = k_best_m.fit(x_train_labeled_robust_scaled, y_train_labeled)
print("K-Best F-Classif Scores:", sorted(k_best_m.scores_, reverse=True))
print()

# select fpr - f classif
fpr_f = SelectFpr(score_func=f_classif)
fpr_f = fpr_f.fit(x_train_labeled_robust_scaled, y_train_labeled)
print("Select FPR: F-Classif Scores:", sorted(fpr_f.scores_, reverse=True))
print()

# select fpr - f mutual info classif
fpr_m = SelectFpr(score_func=mutual_info_classif)
fpr_m = fpr_m.fit(x_train_labeled_robust_scaled, y_train_labeled)
print("Select FPR: F-Classif Scores:", sorted(fpr_m.scores_, reverse=True))
print()

# tree feature selection
model = ExtraTreesClassifier()
model.fit(x_train_labeled_robust_scaled, y_train_labeled)
print("Tree Features")
print(sorted(model.feature_importances_, reverse=True))
print()
print("Tree Feature Importance: ", model.feature_importances_.shape)
print()

pca = PCA(n_components=0.99)
pca = pca.fit(x_train_labeled_robust_scaled)
print("PCA")
print("Explained Variance: {}".format(pca.explained_variance_ratio_))
# print(pca.components_)
print("PCA - Transformed Shape:", pca.components_.shape)
print()

# feature selection - normalized
print("-------------------------------------")
print("Normalizer")

# select k-best - f classif
k_best_f = SelectKBest(score_func=f_classif, k='all')
k_best_f = k_best_f.fit(x_train_labeled_normalized, y_train_labeled)
print("K-Best F-Classif Scores:", sorted(k_best_f.scores_, reverse=True))
print()

# select k-best - f mutual info classif
k_best_m = SelectKBest(score_func=mutual_info_classif, k='all')
k_best_m = k_best_m.fit(x_train_labeled_normalized, y_train_labeled)
print("K-Best F-Classif Scores:", sorted(k_best_m.scores_, reverse=True))
print()

# select fpr - f classif
fpr_f = SelectFpr(score_func=f_classif)
fpr_f = fpr_f.fit(x_train_labeled_normalized, y_train_labeled)
print("Select FPR: F-Classif Scores:", sorted(fpr_f.scores_, reverse=True))
print()

# select fpr - f mutual info classif
fpr_m = SelectFpr(score_func=mutual_info_classif)
fpr_m = fpr_m.fit(x_train_labeled_normalized, y_train_labeled)
print("Select FPR: F-Classif Scores:", sorted(fpr_m.scores_, reverse=True))
print()

# tree feature selection
model = ExtraTreesClassifier()
model.fit(x_train_labeled_normalized, y_train_labeled)
print("Tree Features")
print(sorted(model.feature_importances_, reverse=True))
print()
print("Tree Feature Importance: ", model.feature_importances_.shape)
print()

pca = PCA(n_components=0.99)
pca = pca.fit(x_train_labeled_normalized)
print("PCA")
print("Explained Variance: {}".format(pca.explained_variance_ratio_))
# print(pca.components_)
print("PCA - Transformed Shape:", pca.components_.shape)
print()