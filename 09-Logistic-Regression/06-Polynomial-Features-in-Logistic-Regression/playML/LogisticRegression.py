#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-19 18:34:38
# @update on: 2018-12-19 20:19:21
import numpy as np
from .metrics import accuracy_score

class LogisticRegression():
    """ 逻辑回归 """
    def __init__(self):
        self._theta = None
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, t):
        return 1 / (1 + np.exp(-t))
        
    def fit(self, X_train, y_train, eta=0.01, epsilon=1e-8, n_iters_max=10000):
        """ 根据训练数据 X_train, y_train, 使用梯度下降法训练 LogisticRegression 模型 """
        assert X_train.shape[0] == y_train.shape[0], \
            "The size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            """ 分类算法损失函数 """
            y_hat = self._sigmoid(X_b.dot(theta))
            return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        def dJ(theta, X_b, y):
            """ 损失函数的梯度 """
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / y.shape[0]

        def gradient_descend(X_b, y, initial_theta, eta=0.01, epsilon=1e-8, n_iters_max=10000):
            """ 梯度下降法 """
            assert initial_theta.ndim == 1, \
                "The ndim of initial_theta must be 1"
            assert initial_theta.shape[0] == X_b.shape[1], \
                "The size of initial_theta must be equal to the features number of X_b"
            theta = initial_theta
            previous_J = J(theta, X_b, y)
            n_iters = 0
            while n_iters < n_iters_max:
                gradient = dJ(theta, X_b, y)
                theta -= eta * gradient
                current_J = J(theta, X_b, y)
                if np.abs(current_J - previous_J) <= epsilon:
                    break
                previous_J = current_J
                n_iters += 1

            return theta

        X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        initial_theta = np.zeros((X_b.shape[1]))

        self._theta = gradient_descend(X_b, y_train, initial_theta)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]
        return self

    def predict_probability(self, X_test):
        assert self._theta is not None, \
            "Must fit before predict"
        assert X_test.shape[1] + 1 == self._theta.shape[0], \
            "The features number of X_test must be equal to the size of theta"

        X_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_test):
        return np.array(self.predict_probability(X_test) > 0.5, dtype=int)


    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"