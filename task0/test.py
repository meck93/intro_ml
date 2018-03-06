import os

mingw_path = ";C:\Program Files\mingw-w64\x86_64-7.2.0-posix-sjlj-rt_v5-rev1\mingw64\bin"
os.environ['PATH'] = os.environ['PATH'] + mingw_path

import xgboost as xgb
import numpy as np

data = np.random.rand(5,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix( data, label=label)

dtest = dtrain

param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 4
param['eval_metric'] = 'auc'

evallist  = [(dtest,'eval'), (dtrain,'train')]

num_round = 10
bst = xgb.train( param, dtrain, num_round, evallist )

bst.dump_model('dump.raw.txt')