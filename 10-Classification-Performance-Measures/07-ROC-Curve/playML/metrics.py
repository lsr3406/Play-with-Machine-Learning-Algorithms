#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-17 11:59:53
# @update on: 2018-12-20 19:35:18
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

def _tn(y_test, y_predict):
    """ 计算混淆矩阵第一项 """
    return np.sum((y_test == 0) & (y_predict == 0))

def _fp(y_test, y_predict):
    """ 计算混淆矩阵第二项 """
    return np.sum((y_test == 0) & (y_predict == 1))

def _fn(y_test, y_predict):
    """ 计算混淆矩阵第三项 """
    return np.sum((y_test == 1) & (y_predict == 0))

def _tp(y_test, y_predict):
    """ 计算混淆矩阵第四项 """
    return np.sum((y_test == 1) & (y_predict == 1))

def TPR(y_test, y_predict):
    """ 计算 tpr (召回率) """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return _tp(y_test, y_predict) / (_tp(y_test, y_predict) + _fn(y_test, y_predict))

def FPR(y_test, y_predict):
    """ 计算 fpr """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return _fp(y_test, y_predict) / (_fp(y_test, y_predict) + _tn(y_test, y_predict))

def precision_score(y_test, y_predict):
    """ 计算精准率 """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return _tp(y_test, y_predict) / (_tp(y_test, y_predict) + _fp(y_test, y_predict))

def recall_score(y_test, y_predict):
    """ 计算召回率 """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return _tp(y_test, y_predict) / (_tp(y_test, y_predict) + _fn(y_test, y_predict))

def confusion_matrix(y_test, y_predict):
    """ 计算混淆矩阵 """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return np.array([
        [_tn(y_test, y_predict), fp(y_test, y_predict)],
        [_fn(y_test, y_predict), tp(y_test, y_predict)]
    ])

def f1_score(y_test, y_predict):
    """ 计算 f1_score """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    return 2 * precision * recall / (precision + recall)


