from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import math

# personal csv reader module
import reader

FILE_PATH_TRAIN = "train.csv"
FILE_PATH_TEST = "test.csv"

# read training file
data = reader.read_csv(FILE_PATH_TRAIN)
test = reader.read_csv(FILE_PATH_TEST)

# splitting the training data set into x and y components
new_train_data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
X = new_train_data.values
Y = data['y'].values

# splitting the test data set
new_test_data = test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
X_TEST = new_test_data.values

means = []

for row in X_TEST:
    means.append(np.mean(row))

best_model = None
best_rmse = 1

for i in range(0, 20):
    # splitting the training set into a training & validation set
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)

    # ridge regression model
    alphas = [1e-30, 1e-25, 1e-20, 1e-15, 1e-10, 1e-5, 0.1, 1, 10, 100, 500, 1000]
    # alphas = [1e-22, 1e-20, 1e-17, 1e-15, 1e-12, 1e-10, 1e-8, 1e-5, 1e-3, 1e-1, 1, 10, 100, 1000]

    # cross validation - 10 fold
    kfold = KFold(n_splits=10)
    ridge = RidgeCV(alphas=alphas, fit_intercept=True, normalize=True, cv=kfold,
                    gcv_mode='svd', scoring='neg_mean_squared_error')

    # fit the training data
    ridge.fit(x_train, y_train)

    # print("RidgeCV-intercept:", ridge.intercept_, "\n", "\nRidgeCV-Coefficients:", ridge.coef_)

    # computing the predictions
    y_value_predictions = ridge.predict(x_val)
    # print("Y_Value Predictions:\n", y_value_predictions)
    # print("The true y-values:\n", y_val)

    # prediction & computing error metric
    # print(ridge.score(x_val, y_val))

    # root mean squared error
    current_rmse = math.sqrt(mean_squared_error(y_val, y_value_predictions))

    # computing root mean square error metric
    print("Root Mean Square Error", current_rmse)

    if (current_rmse < best_rmse):
        best_rmse = current_rmse
        best_model = ridge

print("Best Training Model RMSE:\n", best_rmse)

print("best model alpha:", best_model.alpha_)

x_test_predictions = best_model.predict(X_TEST)
rmse = math.sqrt(mean_squared_error(means, x_test_predictions))
print("RMSE of Test Data Set:", rmse)

# Writing the results to a .csv file
data = {'Id': test['Id'].values, 'y': x_test_predictions}
test_output = pd.DataFrame(data=data)
print(test_output.head(5))
test_output.to_csv("ridgecv_out_v2_2.csv", sep=',', index=False)













