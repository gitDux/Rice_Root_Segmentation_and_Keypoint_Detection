# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 13:59:10 2018

@author: sheld
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
import skimage.io
import os
from sklearn.metrics import mean_squared_error


# load or create your dataset
print('Load data...')
# read data
train_path = './data/train'
test_path = './data/test'

train_data = []
train_label = []
train_data = []
train_label = []
for x in os.listdir(os.path.join(train_path, 'img')):
    img_path = os.path.join(train_path, 'img', x)
    ano_path = os.path.join(train_path, 'imgAno', x)
    img = skimage.io.imread(img_path)
    img_ano = skimage.io.imread(ano_path, 1)
    # 归一化
    img = img.astype(np.float)/255
    img_ano = img_ano.astype(np.float)/255
    img_ano[img_ano > 0] = 1
    for i in range(0, 512):
        for j in range(0, 512):
            train_data.append(img[i, j, :])
            train_label.append(img_ano[i, j])

train_data = np.array(train_data)
train_label = np.array(train_label)

# shuffle data
train = np.hstack((train_data, train_label.reshape(-1, 1)))
np.random.shuffle(train)
train_data = train[:, 0:3]
train_label = train[:, 3]

# test data
test_data = []
test_label = []
for x in os.listdir(os.path.join(test_path, 'img')):
    img_path = os.path.join(test_path, 'img', x)
    ano_path = os.path.join(test_path, 'imgAno', x)
    img = skimage.io.imread(img_path)
    img_ano = skimage.io.imread(ano_path, 1)
    # 归一化
    img = img.astype(np.float)/255
    img_ano = img_ano.astype(np.float)/255
    img_ano[img_ano > 0] = 1
    for i in range(0, 512):
        for j in range(0, 512):
            test_data.append(img[i, j, :])
            test_label.append(img_ano[i, j])

test_data = np.array(test_data)
test_label = np.array(test_label)

# create dataset for lightgbm
lgb_train = lgb.Dataset(train_data, train_label)
lgb_test = lgb.Dataset(test_data, test_label, reference=lgb_train)

#%%
# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 2,
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_test,
                early_stopping_rounds=5)


print('Start predicting...')
# predict
y_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
# eval
#%%
y_pred_c = np.argmax(y_pred, axis = 1)
acc = np.sum((y_pred_c == test_label))/test_label.size
print(acc)