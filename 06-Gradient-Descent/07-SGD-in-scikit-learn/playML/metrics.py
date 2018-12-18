#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-17 11:59:53
# @update on: 2018-12-17 21:54:10
import numpy as np

def accuracy_score(y_test, y_predict):
    """ 计算分类算法的准确率 """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return np.sum(y_test == y_predict) / y_test.size

def mean_squared_error(y_test, y_predict):
    """ 计算回归算法评价指标 MSE """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return np.mean((y_test - y_predict) ** 2)

def root_mean_squared_error(y_test, y_predict):
    """ 计算回归算法评价指标 RMSE """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return np.sqrt(np.mean((y_test - y_predict) ** 2))

def mean_absolute_error(y_test, y_predict):
    """ 计算回归算法评价指标 MAE """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return np.mean(np.absolute(y_test - y_predict))

def r2_score(y_test, y_predict):
    """ 计算回归算法评价指标 R^2 """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)