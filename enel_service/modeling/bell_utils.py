from typing import Union, List

import numpy as np
import scipy as sp
import scipy.optimize
import scipy.spatial
from numpy.linalg import LinAlgError


class InterpolationSplits:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.xu = np.unique(x)[1:-1]
        self.n_iter = self.xu.size
        self.i_iter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i_iter == self.n_iter:
            raise StopIteration

        m = self.x == self.xu[self.i_iter]
        xtrain, ytrain = self.x[~m], self.y[~m]
        xtest, ytest = self.x[m], self.y[m]

        self.i_iter = self.i_iter + 1

        return (xtrain, ytrain), (xtest, ytest)

    def __len__(self):
        return self.n_iter


class MeanRelativeError(object):
    def __call__(self, y_pred, y):
        return np.mean(np.abs((y_pred - y) / y))


def cv_score(models, splits):
    scores = np.zeros((len(models), len(splits)))
    loss_func = MeanRelativeError()

    for i, ((xtrain, ytrain), (xtest, ytest)) in enumerate(splits):
        for j, model in enumerate(models):
            model.fit((xtrain, ytrain))
            ypred = model.predict((xtest, ytest))
            scores[j, i] = loss_func(ypred, ytest)

    return scores


class Ernest(object):
    def _fmap(self, x):
        return np.c_[np.ones_like(x), 1. / x, np.log(x), x]

    def fit(self, *args, **kwargs):
        x, y = args[0]
        x, y = x.flatten(), y.flatten()
        X = self._fmap(x)
        coeff, res = sp.optimize.nnls(X, y)
        self.coeff = coeff
        return self

    def predict(self, x, y=None):
        x, _ = x
        x = x.flatten()
        X = self._fmap(x)
        return np.dot(X, self.coeff)


class KernelReg(object):
    def __init__(self, bw=None, degree=1, tol=np.finfo(np.float).eps):
        self.bw = bw
        self.degree = degree
        self.tol = tol

    def _fmap(self, x):
        return np.vstack([x ** i for i in range(self.degree + 1)]).T

    def _predict_single(self, X, y, x, w):
        XTW = X.T * w

        matrix_x = np.dot(XTW, X) + self.tol * np.eye(X.shape[1])
        matrix_b = np.dot(XTW, y)

        try:
            c = np.linalg.solve(matrix_x, matrix_b)
        except LinAlgError as err:
            print("Could not compute solution, so either matrix_x is singular or not square.")
            print("Will try now with the pseudo-inverse of matrix_x...")
            c = np.linalg.pinv(matrix_x).dot(matrix_b)
            print("Success with pseudo-inverse of matrix_x!")

        ypred = np.dot(x, c)
        return ypred

    def _predict(self, X, y, Xpred, W):
        n, d = Xpred.shape
        ypred = np.zeros(n)
        for i in range(n):
            ypred[i] = self._predict_single(X, y, Xpred[i], W[i])
        return ypred

    def fit(self, *args, **kwargs):
        x, y = args[0]
        x, y = x.flatten(), y.flatten()
        if self.bw is None:
            models = [KernelReg(bw=bw) for bw in np.linspace(1, 100, 100)]
            scores = cv_score(models, InterpolationSplits(x, y))
            scores = np.mean(scores, axis=1)
            idx = np.argmin(scores)
            self._bw = models[idx].bw
        else:
            self._bw = self.bw

        self.x = x
        self.y = y
        return self

    def predict(self, xs, y=None):
        xs, _ = xs

        xs = xs.flatten()

        xs = np.atleast_1d(xs)
        x, y, h = self.x, self.y, self._bw

        D = sp.spatial.distance.cdist(np.atleast_2d(xs).T, np.atleast_2d(x).T, metric='sqeuclidean')
        W = np.exp(-D / (2 * h ** 2))

        X = self._fmap(x)
        X0 = self._fmap(xs)

        ypred = self._predict(X, y, X0, W)
        return ypred


class AllocationAssistant(object):

    def fit(self, *args, **kwargs):
        x, y = args[0]
        x, y = x.flatten(), y.flatten()
        models = [KernelReg(bw=bw) for bw in np.linspace(1, 100, 100)] + [Ernest()]
        scores = cv_score(models, InterpolationSplits(x, y))
        scores = np.mean(scores, axis=1)
        idx = np.argmin(scores)
        self.model = models[idx]
        return self

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
