import numpy as np

class LinearRegression:
    
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        
    def fit_normal(self,X_train,y_train):
        assert X_train.shape[0] == y_train.shape[0], \
        "the size of X-train must be equal to  y_train"
        X_b = np.hstack([np.ones((len(X_train),1)),X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)   # np.linalg.inv 是矩阵求逆
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self
    
    def predict(self, X_predict):
        
        assert self.interception_ is not None and self.coef_ is not None, \
        "must fit before"
        assert X_predict.shape[1] == len(self.coef_), \
        "维度要相等"
        
        X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
        
        return X_b.dot(self._theta)
