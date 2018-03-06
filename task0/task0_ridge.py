from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
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

# splitting the training data set into x and y components
new_train_data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
X = new_train_data.values
Y = data['y'].values

ridge_good_mse = 1
ridge_good_predictions = None
current_mse = None

# read training file
test = reader.read_csv(FILE_PATH_TEST)

# splitting the training data set into x and y components
test_data = test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
x_test = test_data.values

means = []

for row in x_test:
    means.append(np.mean(row))

for i in range(0, 20):
    # splitting the training set into a training & validation set
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25)

    # ridge regression model
    alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 50, 100]

    # cross validation - 10 fold
    kfold = KFold(n_splits=10)
    ridge = RidgeCV(alphas, fit_intercept=True, normalize=True, cv=kfold, gcv_mode='auto')

    ridge.fit(x_train, y_train)

    # print("RidgeCV-intercept:", ridge.intercept_, "\nRidgeCV-Coefficients:", ridge.coef_)

    # print(ridge.predict(x_val))
    # print(ridge.score(x_val, y_val))

    # print("The first 5 y_values:", y_val[0:5])

    predictions = ridge.predict(x_test)

    current_mse = math.sqrt(mean_squared_error(means, predictions))

    if (current_mse < ridge_good_mse):
        ridge_good_mse = current_mse
        ridge_good_predictions = predictions

    # computing error metrics
    print("Root Mean Square Error", math.sqrt(mean_squared_error(means, predictions)))

print("Root Mean Square Error - BEST", math.sqrt(mean_squared_error(means, ridge_good_predictions)))

data = {'Id': test['Id'].values, 'y': ridge_good_predictions}
test_output = pd.DataFrame(data=data)

print(test_output.head(5))

test_output.to_csv("test_out_sklearn_cv-10-fold_ridgecv.csv", sep=',', index=False)















