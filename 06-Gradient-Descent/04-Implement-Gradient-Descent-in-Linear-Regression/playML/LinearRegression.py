#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-17 22:10:30
# @update on: 2018-12-18 13:29:51
import numpy as np
from sklearn.metrics import r2_score

class LinearRegression():
    """ 多元线性回归 """
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def fit_normal(self, X_train, y_train):
        """ 使用正规方程计算非线性回归的参数 """
        assert X_train.ndim == 2 and y_train.ndim == 1, \
            "Invalid ndim of training sets"
        assert X_train.shape[0] == y_train.shape[0], \
            "The size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.coef_ = self._theta[1:]
        self.intercept_ = self._theta[0]

    def fit_gd(self, X_train, y_train, eta=0.01, epsilon=1e-8, n_iters_max=1e4):
        """ 使用正规方程计算非线性回归的参数 """
        assert X_train.ndim == 2 and y_train.ndim == 1, \
            "Invalid ndim of training sets"
        assert X_train.shape[0] == y_train.shape[0], \
            "The size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((y_train.shape[0], 1)), X_train])
        initial_theta = np.zeros((X_train.shape[1]+1))

        def J(theta, X_b, y):
            try:
                return np.linalg.norm(y - X_b.dot(theta))
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            try:
                return X_b.T.dot(X_b.dot(theta) - y) * 2 / y.size
            except :
                return float('inf')

        def gradient_descent(X_b, y, initial_theta, eta=0.01, epsilon=1e-8, n_iters_max=1e4):
            assert X_b.shape[0] == y.shape[0], \
                "The number of cloumns of X_b must be equal to y"
            assert X_b.shape[1] == initial_theta.shape[0], \
                "The number of features of X_b must be equal to initial_theta"

            n_iters = 0
            theta = initial_theta
            previous_J = J(theta, X_b, y)
            while n_iters < n_iters_max:
                gradient = dJ(theta, X_b, y)
                theta -= eta * gradient
                current_J = J(theta, X_b, y)
                if np.abs(current_J - previous_J) < epsilon:
                    break

                previous_J = current_J
                n_iters += 1

            return theta

        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, epsilon, n_iters_max)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_test):
        assert self._theta is not None, \
            "Must fit before predict"
        assert X_test.shape[1] == self.coef_.size, \
            "The features number of X_test must be equal to X_train"

        X_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)
        
    def __repr__(self):
        return "LinearRegression()"

