import numpy as np
import math
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

# import sys
# sys.path.append('C:\\Users\\Moritz Eck\\code\\fs18\\intro_ml\\shared')
import reader

# FILE PATH: TRAINING FILE
FILE_PATH_TRAIN = "train1a.csv"
TEST_SIZE = 0.225

# alpha parameter values
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

# training data
data = reader.read_csv(FILE_PATH_TRAIN, show_info=False)

# drop the first column
data = pd.DataFrame.drop(data, columns='Id')

# x-values
X = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10'], copy=True)
X = X.values

# y-values
Y = pd.DataFrame(data, columns=['y'], copy=True)
Y = Y['y'].values

# split into 80% intermediate (training + validation) and 20% test set
x_intermediate, x_test, y_intermediate, y_test = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=False)

# storing the evaluation scores
eval = []

for entry in alphas:
    # creating the 10-fold split
    ten_fold = KFold(n_splits=10, shuffle=False)
    # fold counter
    i = 1
    # create model
    ridge = Ridge(alpha=entry, fit_intercept=True, normalize=False,
                  copy_X=True, max_iter=None, tol=1e-5, solver='auto')
    # errors
    errors = []

    for train_ind, val_ind in ten_fold.split(X, Y):
    # for train_ind, val_ind in ten_fold.split(x_intermediate, y_intermediate):
        # x & y training set (no separated test set)
        x_train = X[train_ind, : ]
        y_train = Y[train_ind]

        # # x & y training set with separated test set
        # x_train = x_intermediate[train_ind, : ]
        # y_train = y_intermediate[train_ind]

        # x & y validation set (no separated test set)
        x_val = X[val_ind, : ]
        y_val = Y[val_ind]

        # # x & y validation set with separated test set
        # x_val = x_intermediate[val_ind, : ]
        # y_val = y_intermediate[val_ind]

        # train the model
        ridge.fit(x_train, y_train)

        # evaluate with validation set
        y_val_predictions = ridge.predict(x_val)

        # compute root mean square error
        rmse = math.sqrt(mean_squared_error(y_val, y_val_predictions))
        errors.append(rmse)

        # printing alpha, fold number and root mean squared error
        print("Alpha:\t", str(entry)+"\t", "Fold:", str(i)+"\t", "RMSE:\t", rmse)

        # increment fold counter
        i += 1

    # compute mean errors for rmse of each alpha
    eval.append(np.mean(errors))
    print()

# print out the rmse averages for all 5 lambdas
for ind in range(0,5):
    print("Alpha:\t", alphas[ind], "\t\t", "Average RMSE:\t", eval[ind])

# writing the result to file
# data = {"Result":eval}
# alpha_rmnse_output = pd.DataFrame(data=eval)
# print(alpha_rmnse_output.head(5))
# alpha_rmnse_output.to_csv("alpha_mean_rmse_ecm_2.csv", sep=',', index=False, header=False)
