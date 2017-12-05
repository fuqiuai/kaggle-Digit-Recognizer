# encoding=utf-8

import time

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA

if __name__ == '__main__':
    
    time_1= time.time()

    print('Prepare datasets...')
    # 读取训练集
    raw_data = pd.read_csv('./data/train.csv', header=0)
    data = raw_data.values
    train_features = data[::, 1::]
    train_labels = data[::, 0]
    # 读取测试集
    raw_data2 = pd.read_csv('./data/test.csv', header=0)
    test_features = raw_data2.values

    # pca降维
    print("Start reduction...")
    pca = PCA(n_components = 50, whiten=True)
    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)
    
    # 训练
    print('Training SVM...')
    clf = svm.SVC()
    clf.fit(train_features, train_labels)  # training the svc model
	
    # 预测
    print('Start predicting...')
    test_predict=clf.predict(test_features)
    
    print('Saving...')
    # 将测试结果保存到csv
    df = pd.DataFrame(test_predict)
    df.index += 1
    df.columns = ['Label']
    df.to_csv('./data/SVM_PCA_results.csv', header=True)

    time_2 = time.time()
    print('cost %f seconds' % (time_2 - time_1))
