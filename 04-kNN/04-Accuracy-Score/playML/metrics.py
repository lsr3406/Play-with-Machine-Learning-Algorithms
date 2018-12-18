#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-17 11:59:53
# @update on: 2018-12-17 12:01:52
import numpy as np

def accuracy_score(y_test, y_predict):
    """ 计算分类算法的准确率 """
    assert y_test.shape[0] == y_predict.shape[0], \
        "The size of y_predict must be equal to the size of y_test"
    return np.sum(y_test == y_predict) / y_test.size