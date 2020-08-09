import numpy as np

def accuracy_score(y_true, y_predict):
	"""计算准确率"""
	assert y_predict.shape[0] == y_true.shape[0], \
	"the size must be equal"
	
	return sum(y_true == y_predict)/len(y_true)