#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: siru
# @create on: 2018-12-18 19:33:59
# @update on: 2018-12-18 20:14:13
import numpy as np

class PCA():
    """ 主成分分析类 """
    def __init__(self, n_components):
        """ PCA 初始化, 确定要多少个主成分 """
        assert n_components >= 1, \
            "n_components must be valid"
        self.n_components = n_components
        self.components_ = None

    def fit(self, X):
        assert X.ndim == 2, \
            "The dimension of X must be 2"

        def demean(X):
            return X - X.mean(axis=0)

        def f(w, X):
            return np.mean(X.dot(w) ** 2)

        def df(w, X):
            return X.T.dot(X.dot(w)) * 2 / X.shape[0]

        def direction(w):
            return w / np.linalg.norm(w)

        def gradient_ascent(X, initial_w, eta=0.1, epsilon=1e-8, n_iters_max=10000):

            w = initial_w
            previous_f = f(w, X)
            n_iters = 0

            while n_iters < n_iters_max:
                gradient = df(w, X)
                w = direction(w + eta * gradient)
                current_f = f(w, X)

                if np.abs(current_f - previous_f) < epsilon:
                    break

                previous_f = current_f
                n_iters += 1

            return w

        def first_component(X):
            initial_w = np.random.normal(0, 1, (X.shape[1],))
            return gradient_ascent(X, initial_w)

        self.components_ = np.empty((self.n_components, X.shape[1]))
        X_pca = demean(X)

        for k in range(self.n_components):

            w = first_component(X_pca)
            self.components_[k] = w
            # X_pca -= X_pca.dot(w).reshape(-1, 1).dot(w.reshape(1, -1))
            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    def transform(self, X):
        """ 对 X 进行一次主成分的正变换 """
        assert X.shape[1] == self.components_.shape[1]
        return X.dot(self.components_.T)

    def inverse_transform(self, X):
        """ 对 X 进行一次主成分的反变换 """
        assert X.shape[1] == self.components_.shape[0]
        return X.dot(self.components_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components

