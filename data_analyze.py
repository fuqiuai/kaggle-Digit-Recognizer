# encoding=utf-8

import pandas as pd
import numpy as np
import time



if __name__ == '__main__':

    # 读取训练集
    raw_data = pd.read_csv('./data/train.csv', header=0)
    data = raw_data.values
    train_features = data[::, 1::]
    train_labels = data[::, 0]
    # 读取测试集
    raw_data2 = pd.read_csv('./data/test.csv', header=0)
    test_features = raw_data2.values
    
    print(pd.Series(train_labels).value_counts())
    # print(raw_data2.isnull().any())
    print(raw_data2.isnull().any().describe())
