import numpy as np
from collections import Counter
from math import sqrt 
from metrics import accuracy_score

class KnnClassifier:

	def __init__(self, k):
		assert 1<= k, "k value must valid"
		self.k = k 
		self._X_train = None
		self._Y_train = None
		
	def fit(self, X_train, Y_train):
		
		assert X_train.shape[0] == Y_train.shape[0],"the number must be equal"
		self._X_train = X_train
		self._Y_train = Y_train
		return self
		
	def predict(self, X_predict):
		"""对于给定的多个的预测数据X，返回预测结果"""
		assert self._X_train is not None and self._Y_train is not None, \
		"must fit before predict"
		assert X_predict.shape[1] == self._X_train.shape[1], \
		"the feature number must be equal "
		
		y_predict = [self._predict(x) for x in X_predict]
		
		return np.array(y_predict)
		
	def _predict(self, x):
		"""对于给定的单个的预测数据x，返回预测结果"""
		assert x.shape[0] == self._X_train.shape[1], \
		"the number must be equal "
		
		distances = [sqrt(np.sum((x_train - x)**2)) for x_train in self._X_train]
		nearest = np.argsort(distances)
		
		topK_y = [self._Y_train[i] for i in nearest[:self.k]]
		votes = Counter(topK_y)
		
		return votes.most_common(1)[0][0]
		
	def score(self, X_test, y_test):
		y_predict = self.predict(X_test)
		return accuracy_score(y_test, y_predict)
		
	
	def accuracy_score(y_true, y_predict):
		"""计算准确率"""
		assert y_predict.shape[0] == y_true.shape[0], \
		"the size must be equal"
		
		return sum(y_true == y_predict)/len(y_true)
		
	def __repr__(self):
		return "KNN(k=%d)" %self.k
	
	