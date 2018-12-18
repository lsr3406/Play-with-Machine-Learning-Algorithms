#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-17 10:19:21
# @update on: 2018-12-17 10:48:47
import numpy as np
from collections import Counter

class kNNClassifier():
    """ kNN 算法的实现 """
    def __init__(self, k):
        """ 初始化 KNN 分类器 """
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """ 根据训练数据集 X_train 和 y_train 训练 KNN 分类器 """
        assert X_train.shape[0] == y_train.shape[0], \
            "The size of X_train must be equal to the size of y_train"
        assert self.k <= y_train.shape[0], \
            "The size of y_train must be at least k"
        self._X_train = X_train
        self._y_train = y_train

    def predict(self, X_predict):
        """ 给定预测数据集 X_predict, 返回表示 X_predict 的结果向量 """
        assert self._X_train is not None and self._y_train is not None, \
            "Must fit before predict"
        assert X_predict.shape[1] == X_predict.shape[1], \
            "The feature number of X_predict must be equal to X_train"
        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """ 给定单个预测数据 x, 返回 x 的结果向量 """
        distance = [np.linalg.norm(x_train - x) for x_train in self._X_train]
        distance = np.sqrt(distance)
        
        nearest = np.argsort(distance)
        topK_y = self._y_train[nearest[:self.k]]
        
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]

    def __repr__(self):
        return "kNNClassifier(k=%d)" % self.k
