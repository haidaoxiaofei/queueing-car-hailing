# coding: utf-8
# pylint: disable = invalid-name, C0111
import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

try:
    import cPickle as pickle
except BaseException:
    import pickle

class GBDTData(object):
    def __init__(self, numeric_data, categorical_data, label):
        assert len(numeric_data) + len(categorical_data) > 0
        for data in numeric_data:
            assert len(data) == len(label)
        for data in categorical_data:
            assert len(data) == len(label)

        # merge data together and record category index
        self.feat = []
        self.category_index = []
        featlen = 0
        for data in numeric_data:
            featlen += data.shape[1]
            self.feat.append(data)

        for data in categorical_data:
            self.category_index += range(featlen, featlen + data.shape[1])
            featlen += data.shape[1]
            self.feat.append(data)

        self.feat = np.hstack(self.feat)
        self.label = label

    def getAllFeats(self):
        return self.feat, self.category_index

    def getAllLabel(self):
        return self.label



# load or create your dataset
#print('Load data...')

def train(X_train, y_train, X_test, y_test, ckpt_path, category_index, mmn):
    num_train, num_feature = X_train.shape

    lgb_train = lgb.Dataset(X_train, y_train,
        free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, y_test,
        free_raw_data=False)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    feature_name =['feature_' + str(col) for col in range(num_feature)]

    print('start training...')
    gbm = lgb.train(params, lgb_train, 
        num_boost_round=50, 
        valid_sets=lgb_eval,
        feature_name=feature_name,
        categorical_feature=category_index
        #,init_model=''
        )

    # check feature name
    print('Finish first 10 rounds...')
    print('7th feature name is:',repr(lgb_train.feature_name[6]))

    # save model to file
    gbm.save_model(ckpt_path)

    # dump model to JSON (and save to file)
    # print('Dump model to JSON')
    # model_json = gbm.dump_model()

    # with open(ckpt_path, 'w+') as f:
    #     json.dump(model_json, f, indent=4)

    # feature names
    print('Feature names:', gbm.feature_name())

    # feature importances
    print('Feature importances:', list(gbm.feature_importance()))
    print('Load model to predict')
    bst = lgb.Booster(model_file=ckpt_path)
    y_pred = bst.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    real_rmse = rmse * (mmn._max - mmn._min) / 2
    print('rmse={}, real rmse={}'.format(rmse, real_rmse))

def inference(ckpt_path, X_test):
    print('Load model to predict')
    bst = lgb.Booster(model_file=ckpt_path)

    y_pred = bst.predict(X_test)
    return y_pred
