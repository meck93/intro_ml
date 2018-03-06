import numpy as np
import math
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

import os

mingw_path = ";C:\Program Files\mingw-w64\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\mingw64\bin"
os.environ['PATH'] = os.environ['PATH'] + mingw_path

import xgboost as xgb

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

# read the test file
test = reader.read_csv(FILE_PATH_TEST)

# splitting the test data set
new_test_data = test[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']]
X_TEST = new_test_data.values

def R2(ypred, ytrue):
    y_avg = np.mean(ytrue)
    SS_tot = np.sum((ytrue - y_avg)**2)
    SS_res = np.sum((ytrue - ypred)**2)
    r2 = 1 - (SS_res/SS_tot)
    return r2

def cross_validate(model, x, y, folds=10, repeats=5):
    '''
    Function to do the cross validation - using stacked Out of Bag method instead of averaging across folds.
    model = algorithm to validate. Must be scikit learn or scikit-learn like API (Example xgboost XGBRegressor)
    x = training data, numpy array
    y = training labels, numpy array
    folds = K, the number of folds to divide the data into
    repeats = Number of times to repeat validation process for more confidence
    '''
    ypred = np.zeros((len(y),repeats))
    score = np.zeros(repeats)
    x = np.array(x)

    for r in range(repeats):
        i=0
        print('Cross Validating - Run', str(r + 1), 'out of', str(repeats))

        x,y = shuffle(x,y,random_state=r) #shuffle data before each repeat
        kf = KFold(n_splits=folds,random_state=i+1000) #random split, different each time

        for train_ind, test_ind in kf.split(x):
            print('Fold', i+1, 'out of', folds)
            xtrain, ytrain = x[train_ind,:], y[train_ind]
            xtest, ytest = x[test_ind,:], y[test_ind]
            model.fit(xtrain, ytrain)
            ypred[test_ind, r] = model.predict(xtest)
            i+=1

        score[r] = R2(ypred[:,r],y)

    print('\nOverall R2:',str(score))
    print('Mean:',str(np.mean(score)))
    print('Deviation:',str(np.std(score)))

    return model

def run():
    train = pd.read_csv('train.csv')
    y = np.array(train['y'])
    train = train.drop(columns=['Id', 'y'])
    ridge_model = Ridge(alpha=1)
    xgb_model = xgb.XGBRegressor(max_depth=2, learning_rate=0.01, n_estimators=10, silent=True,
                                objective='reg:linear', nthread=-1, base_score=100, seed=4635,
                                missing=None)
    # test_model = cross_validate(ridge_model, np.array(train), y, folds=10, repeats=5) #validate ridge regression
    test_model = cross_validate(xgb_model, np.array(train), y, folds=10, repeats=5) #validate xgboost

run()

