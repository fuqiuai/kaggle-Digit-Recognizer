# encoding=utf-8

import pandas as pd
import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':

    print("Start read data...")

    time_1 = time.time()

    # 读取训练集
    raw_data = pd.read_csv('./data/train.csv', header=0)
    data = raw_data.values
    train_features = data[::, 1::]
    train_labels = data[::, 0]
    # 读取测试集
    raw_data2 = pd.read_csv('./data/test.csv', header=0)
    test_features = raw_data2.values
    
    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training...')
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(train_features, train_labels) 
    time_3 = time.time()
    print('training cost %f seconds...' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = neigh.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))
    print('End!')

    # 将测试结果保存到csv
    df = pd.DataFrame(test_predict)
    df.index += 1
    df.columns = ['Label']
    df.to_csv('./data/KNN_results.csv', header=True)
