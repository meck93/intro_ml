from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

import reader

# file paths
XGB_CMP_FILEPATH = "../output/xgb/average/task2_xgb_nativ_av[1-6].csv"
MLP_CMP_FILEPATH = "../output/mlp_lbgfs/average/task2_mlp_lbgfs_av[1-6].csv"

# read comparison files
xgb_cmp_file = reader.read_csv(XGB_CMP_FILEPATH, False)['y'].values
mlp_cmp_file = reader.read_csv(MLP_CMP_FILEPATH, False)['y'].values

# filename prefixes
FILE_PREFIX_XGB = "../output/xgb/task2_xgb_nativ_["
FILE_PREFIX_MLP = "../output/mlp_lbgfs/task2_mlp_lbgfs_["

# filename suffix
FILE_SUFFIX = "].csv"

# read training file
xgb_files = []
mlp_files = []

# read all existing xgb files
for i in range(1, 9):
    xgb_files.append(reader.read_csv(FILE_PREFIX_XGB + str(i) + FILE_SUFFIX, False)['y'].values)

# read all existing mlp files
for i in range(1, 7):
    mlp_files.append(reader.read_csv(FILE_PREFIX_MLP + str(i) + FILE_SUFFIX, False)['y'].values)

print()
print("XGBoost Error Metrics")
print()

for i in range(0, 8):
    # computing error metrics
    acc = accuracy_score(xgb_cmp_file, xgb_files[i])
    print("File Nr:", i+1, "Accuracy Score:", acc)

print()
print("Sklearn MLP Error Metrics")
print()

for i in range(0, 6):
    # computing error metrics
    acc = accuracy_score(mlp_cmp_file, mlp_files[i])
    print("File Nr:", i+1, "Accuracy Score:", acc)

print()
acc = accuracy_score(mlp_cmp_file, xgb_cmp_file)
print("MLP vs XGB", "Accuracy Score:", acc)
