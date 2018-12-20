#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-17 20:05:39
# @update on: 2018-12-17 22:02:46
import numpy as np
from .metrics import r2_score

class SimpleLinearRegression():
    """ 简单的线性回归 """
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """ 根据给定的训练数据集 x_train, y_train 训练 SimpleLinearRegression """
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert x_train.shape[0] == y_train.shape[0], \
            "The size of x_train must be equal to the size of y_train"
        
        x_mean = x_train.mean()
        y_mean = y_train.mean()
        # self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot((x_train - x_mean))
        self.b_ = y_mean - self.a_ * x_mean

    def predict(self, x_predict):
        """ 给定待预测数据集 x_predict, 返回表示 x_predict 的结果向量 """
        assert self.a_ is not None and self.b_ is not None, \
            "Must fit before predict"
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        return np.array([self._predict(x_item) for x_item in x_predict])

    def _predict(self, x_item):
        """ 给定单个预测数据 x, 返回表示 x 的预测结果值 """
        return self.a_ * x_item + self.b_

    def score(self, x_test, y_test):
        """ 根据测试数据集 x_test, y_test 确定当前模型的准确度 """
        assert x_test.shape[0] == y_test.shape[0], \
            "The size of x_test must be equal to the size of y_test"
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression(a=%4.2f, b=%4.2f)" % (self.a_, self.b_)