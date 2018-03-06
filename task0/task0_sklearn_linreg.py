import numpy as np
import math
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import cross_validation, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import statsmodels.api as sm

# personal csv reader module
import reader

FILE_PATH_TRAIN = "train.csv"
FILE_PATH_TEST = "test.csv"

# read training file
test = reader.read_csv(FILE_PATH_TEST)
data = reader.read_csv(FILE_PATH_TRAIN)

# splitting the test data set
new_test_data = test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
X_TEST = new_test_data.values

# splitting the training data set into x and y components
new_train_data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
X = new_train_data.values
Y = data['y'].values

# splitting the training set into a training & validation set
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# simple regression model
lin_reg = linear_model.LinearRegression(copy_X=True, normalize=True)
lin_reg.fit(x_train, y_train)

print(lin_reg.intercept_, lin_reg.coef_)

print(lin_reg.predict(x_test))
print(lin_reg.score(x_test, y_test))
print("Root Mean Square Error: %.2f" % math.sqrt(np.mean((lin_reg.predict(x_test) - y_test) ** 2)))

# data = {'Id': test['Id'].values, 'y': lin_reg.predict(X_TEST)}
# test_output = pd.DataFrame(data=data)
# print(test_output.head(5))
# test_output.to_csv("test_output.csv", sep=',', index=False)














