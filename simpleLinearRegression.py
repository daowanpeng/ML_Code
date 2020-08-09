import numpy as np 
import matplotlib.pyplot as plt


class SimpleLinearRegression1:
    def __ini__(self):
        self.a_ = None
        self.b_ = None
        
    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
        "维度为1，即数据特征数量为1"
        assert len(x_train) == len(y_train), \
        "长度要求相等"
        
        
		x_means = np.mean(x_train)
        y_means = np.mean(y_train) 
		
		"""使用向量化方式运算取代for循环"""
		self.a_ = (x_train - x_means).dot(y_train - y_means) / (x_train - x_means).dot(x_train - x_means)
		self.b_ = y_means - self.a_ * x_means
		
		"""
        fenzi = 0
        fenmu = 0
        for x_i,y_i in zip(x_train,y_train):
            fenzi += (x_i - x_means)*(y_i - y_means)
            fenmu += (x_i - x_means)**2
        self.a_ = fenzi / fenmu
        self.b_ = y_means - self.a_ * x_means
        """
		
        return self
    
    def predict(self, x_predict):
        """给定的预测数据返回的是一个向量"""
        assert x_predict.ndim == 1, \
        "维度为1"
        assert self.a_ is not None and self.b_ is not None, \
        "must fit before predict!"
        
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
    
    def __repr__(self):
        return "SimpleLinearRegression1"
    