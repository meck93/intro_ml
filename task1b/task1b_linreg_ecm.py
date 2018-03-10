import math

import numpy as np
import pandas as pd
import reader
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# file path constants
FILE_PATH_TRAIN = "train1b.csv"

# feature transformation functions
functions = [lambda x: float(x), lambda x: math.pow(x, 2), lambda x: math.exp(x), lambda x: math.cos(x), lambda x: float(1)]

# read input data
data = reader.read_csv(FILE_PATH_TRAIN, False)

# drop the first column
data = pd.DataFrame.drop(data, columns='Id')

# splitting the data set into x & y values
# y-values
Y = pd.DataFrame(data, columns=['y'], copy=True)
Y = Y['y'].values

# x-values
X = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'x5'], copy=True)

# new data frames for feature transformations
quad = pd.DataFrame(data, columns=['x6', 'x7', 'x8', 'x9', 'x10'])
exp = pd.DataFrame(data, columns=['x11', 'x12', 'x13', 'x14', 'x15'])
cos = pd.DataFrame(data, columns=['x16', 'x17', 'x18', 'x19', 'x20'])
const = pd.DataFrame(data, columns=['x21'])

# performing the feature transformations
for index, row in X.iterrows():
    # 1. transformation: quadratic
    quad['x6'][index] = functions[1](X['x1'][index])
    quad['x7'][index] = functions[1](X['x2'][index])
    quad['x8'][index] = functions[1](X['x3'][index])
    quad['x9'][index] = functions[1](X['x4'][index])
    quad['x10'][index] = functions[1](X['x5'][index])

    # 2. transformation: exponential
    exp['x11'][index] = functions[2](X['x1'][index])
    exp['x12'][index] = functions[2](X['x2'][index])
    exp['x13'][index] = functions[2](X['x3'][index])
    exp['x14'][index] = functions[2](X['x4'][index])
    exp['x15'][index] = functions[2](X['x5'][index])

    # 3. transformation: cosine
    cos['x16'][index] = functions[3](X['x1'][index])
    cos['x17'][index] = functions[3](X['x2'][index])
    cos['x18'][index] = functions[3](X['x3'][index])
    cos['x19'][index] = functions[3](X['x4'][index])
    cos['x20'][index] = functions[3](X['x5'][index])

    # 4. transformation: constant
    const['x21'][index] = functions[4](1)

# concatenating all data frames
X_transformed = pd.concat([X, quad, exp, cos, const], axis=1)
X_transformed = X_transformed.values

# storing the evaluation scores
eval = []

# storing the coefficients
av_coeffs = []

# alpha parameter values
alphas = [10.0e-4, 10.0e-3, 10.0e-2, 10.0e-1, 10.0e0, 10.0, 20.0, 100.0]

for entry in alphas:
    # creating the 10-fold split
    ten_fold = KFold(n_splits=10, shuffle=False)
    # fold counter
    i = 1
    # create model
    ridge = Ridge(alpha=entry, fit_intercept=True, normalize=False,
                  copy_X=True, max_iter=None, tol=1e-10, solver='auto')
    # errors per alpha value
    errors = []
    # coefficients per alpha value
    coeffs = []

    # training the ridge cv model
    for train_ind, val_ind in ten_fold.split(X_transformed, Y):
        # x & y training set (no separated test set)
        x_train = X_transformed[train_ind, :]
        y_train = Y[train_ind]

        # x & y validation set (no separated test set)
        x_val = X_transformed[val_ind, :]
        y_val = Y[val_ind]

        # train the model
        ridge.fit(x_train, y_train)

        # remember the coefficients
        coeffs.append(ridge.coef_)

        # evaluate with validation set
        y_val_predictions = ridge.predict(x_val)

        # compute root mean square error
        rmse = math.sqrt(mean_squared_error(y_val, y_val_predictions))
        errors.append(rmse)

        # printing alpha, fold number and root mean squared error
        # print("Alpha:\t", str(entry)+"\t", "Fold:", str(i)+"\t", "RMSE:\t", rmse)
        # print("Coefficients:\t", ridge.coef_)

        # increment fold counter
        i += 1
    # print()
    # compute mean errors for rmse of each alpha
    eval.append(np.mean(errors))

    # compute the averaged coefficients
    av_coeffs.append(np.mean(coeffs, axis=0, dtype=np.float64))

# print out the rmse averages for all lambdas
min_rmse = 10000000
min_rmse_ind = None

# printing a summary
print("Summary of all alpha values tested:")

for ind in range(0, len(alphas)):
    print("Alpha:\t", alphas[ind], "\t\t", "Avg RMSE:\t", eval[ind])

    if eval[ind] < min_rmse:
        min_rmse = eval[ind]
        min_rmse_ind = ind

# preparing to write the coefficients to file
out = {"Avg Coeffs": av_coeffs[min_rmse_ind]}

# output data frame
best_coeffs = pd.DataFrame(data=out)

# printing test output
print()
print("Result Written To File:")
print("Alpha:\t", alphas[min_rmse_ind], "\tRMSE:\t", min_rmse)
print()
print(best_coeffs.head(5))

# # write to csv-file
# best_coeffs.to_csv("task1b_ridge_10-fold-cv_Nr4.csv", sep=',', index=False, header=False)
