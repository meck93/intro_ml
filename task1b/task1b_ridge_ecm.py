import math

import numpy as np
import pandas as pd
import reader
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split

# file path constants
FILE_PATH_TRAIN = "train1b.csv"
TEST_SIZE = 0.2

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

# intermediate / test split
x_intermediate, x_test, y_intermediate, y_test = train_test_split(X_transformed, Y, test_size=TEST_SIZE, shuffle=False)

# storing the evaluation scores
eval = []

# storing the coefficients
av_coeffs = []

# alpha parameter values
alphas = [10.0e-4, 10.0e-3, 10.0e-2, 10.0e-1, 10.0e0, 10.0, 20.0, 100.0]

# creating the 10-fold split
ten_fold = KFold(n_splits=10, shuffle=False)
repeated_ten_fold = RepeatedKFold(n_splits=10, n_repeats=5)

# create ridge cross validation model
ridge = RidgeCV(alphas=alphas, fit_intercept=True, normalize=False,
                cv=repeated_ten_fold, gcv_mode='auto')

# training the model 
ridge.fit(x_intermediate, y_intermediate)

coeffs = []

# remember the coefficients (weight vector in the cost function)
coeffs.append(ridge.coef_)

# evaluate with validation set
y_val_predictions = ridge.predict(x_test)

# compute root mean square error
rmse = math.sqrt(mean_squared_error(y_test, y_val_predictions))

# Print Out
print()
print("Error Estimate on Validation Set")
print("RMSE:", rmse)
print()
print("Model Info:", "Ridge Cross Validation")
print("Alpha:", ridge.alpha_)
print()
print("Coefficients:", ridge.coef_)

# preparing to write the coefficients to file
out = {"Coefficients": coeffs[0]}

# output data frame
best_coeffs = pd.DataFrame(data=out)

# printing test output
print()
print("Result Written To File:")
print(best_coeffs.head(5))

# write to csv-file
# best_coeffs.to_csv("task1b_elastic-net_10-fold-cv_Nr3.csv", sep=',', index=False, header=False)
