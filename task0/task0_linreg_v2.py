from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
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

for i in range(0, 1000):
    # splitting the training set into a training & validation set
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, shuffle=True)

    lin_reg = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)

    # fit the training data
    lin_reg.fit(x_train, y_train)

    # print("intercept:", lin_reg.intercept_, "\n", "\nCoefficients:", lin_reg.coef_)

    # computing the predictions
    y_value_predictions = lin_reg.predict(x_val)
    # print("Y_Value Predictions:\n", y_value_predictions)
    # print("The true y-values:\n", y_val)

    # prediction & computing error metric
    # print(lin_reg.score(x_val, y_val))

    # root mean squared error
    current_rmse = math.sqrt(mean_squared_error(y_val, y_value_predictions))

    # computing root mean square error metric
    print("Root Mean Square Error", current_rmse)

    if (current_rmse < best_rmse):
        best_rmse = current_rmse
        best_model = lin_reg

print("Best Training Model RMSE:\n", best_rmse)

x_test_predictions = best_model.predict(X_TEST)
rmse = math.sqrt(mean_squared_error(means, x_test_predictions))
print("RMSE of Test Data Set:", rmse)

# Writing the results to a .csv file
data = {'Id': test['Id'].values, 'y': x_test_predictions}
test_output = pd.DataFrame(data=data)
print(test_output.head(5))
test_output.to_csv("linreg_out_v2_5.csv", sep=',', index=False)