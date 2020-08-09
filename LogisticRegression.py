import numpy as np
from .metrics import accuracy_score

class LogisticRegression:
    
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        
    def _sigmoid(self, t):
        return 1 / (1 + np.exp(-t))
        

    def fit(self, X_train, y_train, eta=0.01, n_iters = 1e4):
		

        assert X_train.shape[0] == y_train.shape[0], \
        "the size of X-train must be equal to  y_train"
        
        # 新的损失函数
        def J(theta, X_b, y):
                y_hat = self._sigmoid(X_b.dot(theta))
                try:
                    return -np.sum(y*np.log(y_hat) + (1 - y)*np.log(1 - y_hat)) / len(y)
                except:
                    return float('inf')
            
        # 新的梯度函数
        def dJ(theta, X_b, y):
                return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y)  / len(y)
            

        def gradien_des(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon = 1e-8):

                 theta = initial_theta
                 i_iter = 0

                while i_iter < n_iters:
                    gradient = dJ(theta, X_b, y)
                    last_theta = theta
                    theta = theta - eta * gradient

                    if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                        break
                    i_iter += 1 

                return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        eta = 0.01
        self._theta = gradien_des(X_b, y_train, initial_theta, eta, n_iters) 
        self.coef_ = self._theta[1:]
        self.interception_ = self._theta[0]
        
        return self
       
    
    def predict_proba(self, X_predict):
        """返回X_predict的结果概率向量"""
        assert self.interception_ is not None and self.coef_ is not None, \
        "must fit before"
        assert X_predict.shape[1] == len(self.coef_), \
        "维度要相等"
        
        X_b = np.hstack([np.ones((len(X_predict),1)), X_predict])
        
        return self._sigmiod(X_b.dot(self._theta))
    
    def predict(self, X_predict):
        
        assert self.interception_ is not None and self.coef_ is not None, \
        "must fit before"
        assert X_predict.shape[1] == len(self.coef_), \
        "维度要相等"
        
        proba = self.predict_proba(X_predict)
        
        return np.array(proba >= 0.5, dtype='int') # 本来返回的是一个布尔向量，但是用一个dtype=int，将其转化成int类型
        
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
    
    def __repr__(self):
        return "LogisticRegression()"