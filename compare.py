'''
    Compare the proposed method to random forest and support vector machine
'''
import numpy as np
import skimage.io
from sklearn import svm
import os

# read data
train_path = './data/train'
test_path = './data/test'
#%% train data 
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


#%% test data
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

# shuffle data
test = np.hstack((test_data, test_label.reshape(-1, 1)))
np.random.shuffle(test)
test_data = test[:, 0:3]
test_label = test[:, 3]


#%% segmentation using SVM
clf = svm.SVC(C = 1.0, kernel = 'rbf', gamma='auto', verbose = True)
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)


#%% segmentation using random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, verbose = True)
clf.fit(train_data, train_label)
#%%
y_pred = clf.predict(test_data)
#%%
acc = np.sum((y_pred == test_label))/test_label.size
print(acc)











im








