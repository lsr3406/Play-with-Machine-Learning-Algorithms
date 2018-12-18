#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-17 18:16:48
# @update on: 2018-12-17 18:26:47
import numpy as np

class StandardScaler():
    """ 均值方差归一化 """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """ 根据训练数据集 X 获得数据的均值与方差 """
        assert X.ndim == 2, "The ndim of X must be equal to 2"
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def transform(self, X):
        """ 将 X 根据这个 StandardScaler 进行均值方差归一化 """
        assert X.ndim == 2, "The ndim of X must be equal to 2"
        assert self.mean_ is not None and self.std_ is not None, "Must fit before transform"
        assert X.shape[1] == self.mean_.size, "The feature number of X must be equal to the size of mean_ and std_"
        return (X - self.mean_) / self.std_



        