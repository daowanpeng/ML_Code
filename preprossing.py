import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """根据X获得均值和方差"""
        assert X.ndim == 2, 'The dimension of X must be 2'
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):

        """将X进行均值方差归一化"""
        assert X.dim == 2, "The dimension of X must be 2"

        assert self.mean_ is not None and self.scale_ is not None, \
        "must fit before transform"

        assert X.shape[1] == len(self.mean_), \
        "the number of feature of X must equal to mean_ and scale_"

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]
        return resX
        # transformed_X = np.array([(X[:, i] - self.mean_[i]) / self.scale_[i] for i in range(X.shape[1])])
        # return transformed_X

class MaxMinScaler:
    def __init__(self):
        self.max_ = None
        self.min_ = None

    def fit(self, X):
        """根据X获得最大值和最小值"""
        assert X.ndim == 2, 'The dimension of X must be 2'
        self.max_ = np.array([np.max(X[:, i]) for i in range(X.shape[1])])
        self.min_ = np.array([np.min(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):

        """将X进行均值方差归一化"""
        assert X.dim == 2, "The dimension of X must be 2"

        assert self.max_ is not None and self.min_ is not None, \
        "must fit before transform"

        assert X.shape[1] == len(self.max_), \
        "the number of feature of X must equal to mean_ and scale_"

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            try:
                resX[:, col] = (X[:, col] - self.min_[col]) / (self.max_[col] - self.min_[col])
            except:
                resX[:, col] = resX[:, col]
        return resX

