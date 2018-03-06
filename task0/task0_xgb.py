from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import math

import os

mingw_path = ";C:\Program Files\mingw-w64\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\mingw64\bin"
os.environ['PATH'] = os.environ['PATH'] + mingw_path

import xgboost as xgb

# personal csv reader module
import reader


def cross_validate(model, x, y, folds=10, repeats=5):
    '''
    Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
    model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
    x = training data, numpy array
    y = training labels, numpy array
    folds = K, the number of folds to divide the data into
    repeats = Number of times to repeat validation process for more confidence
    '''
    from sklearn.utils import shuffle
    from sklearn.model_selection import KFold
    import numpy as np

    ypred = np.zeros((len(y), repeats))
    score = np.zeros(repeats)
    other = np.zeros(repeats)


    x = np.array(x)

    for r in range(repeats):
        i = 0
        print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))

        x, y = shuffle(x, y, random_state=r)
        kf = KFold(n_splits=folds, random_state=i + 1000)

        for train_ind, test_ind in kf.split(x):
            print('Fold', i + 1, 'out of', folds)

            xtrain, ytrain = x[train_ind, :], y[train_ind]
            xtest, ytest = x[test_ind, :], y[test_ind]

            model.fit(xtrain, ytrain)

            ypred[test_ind, r] = model.predict(xtest)

            i += 1

        score[r] = R2(ypred[:, r], y)
        other[r] = math.sqrt(mean_squared_error(y, ypred[:, r]))

    print('\nOverall R2:', str(score))
    print("XGBoost: Root Mean Square Error", str(other))
    print('Mean:', str(np.mean(score)))
    print('Deviation:', str(np.std(score)))

    return model


def R2(ypred, ytrue):
    import numpy as np

    y_avg = np.mean(ytrue)
    SS_tot = np.sum((ytrue - y_avg) ** 2)
    SS_res = np.sum((ytrue - ypred) ** 2)
    r2 = 1 - (SS_res / SS_tot)

    return r2

FILE_PATH_TRAIN = "train.csv"
FILE_PATH_TEST = "test.csv"

# read training file
data = reader.read_csv(FILE_PATH_TRAIN)

# splitting the training data set into x and y components
new_train_data = data[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
X = new_train_data.values
Y = data['y'].values

# splitting the training set into a training & validation set
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25)

bst = xgb.XGBRegressor(max_depth=5, learning_rate=0.175, n_estimators=200, silent=True,
                       objective='reg:linear', booster='gbtree',
                       n_jobs=1, gamma=0, min_child_weight=1,
                       max_delta_step=0, subsample=0.5,
                       colsample_bytree=1, colsample_bylevel=1,
                       reg_alpha=0.2, reg_lambda=1, scale_pos_weight=1,
                       base_score=0.5, random_state=0)

# bst.fit(x_train, y_train)
#
# print("The first 5 Y-Values:", y_val[0:5])
#
# # predicting the y-values of x_val
# predictions = bst.predict(x_val)
# print("The first 5 predictions", predictions[0:5])
#
# # computing error metrics
# print("XGBoost: Explained Variance Score", explained_variance_score(y_val, predictions))
# print("XGBoost: Mean Square Error", math.sqrt(mean_squared_error(y_val, predictions)))
# print("XGBoost: R^2", r2_score(y_val, predictions))

trained_model = cross_validate(bst, X, Y)

# save model to file
joblib.dump(trained_model, "trained_model.joblib.dat")

# how to load
# loaded_model = joblib.load("filename.joblib.dat")