from collections import Counter
from math import log

class treeBasedEntropy:

	def split(X, y, d, value):
		index_a = (X[: ,d] <= value)
		index_b = (X[:,d] > value)
		return X[index_a], X[index_b], y[index_a], y[index_b]

	def entropy1(y):
		counter = Counter(y)
		res = 0.0
		for num in counter.values():
			p = num / len(y)
			res += -p * log(p)
		return  res

	def try_split(X, y):
		best_entropy = float('inf')
		best_d, best_v = -1, -1
		for d in range(X.shape[1]):
			sorted_index = np.argsort(X[:,d])
			for i in range(1, len(X)):
				if X[sorted_index[i-1],d] != X[sorted_index[i],d]: 
					v = (X[sorted_index[i-1], d] + X[sorted_index[i], d]) / 2
					x_l, x_r, y_l, y_r = split(x, y, d, v)
					e = entropy1(y_l) + entropy1(y_r)
					if e < best_entropy:
						best_entropy, best_d, best_v = e, d, v
						
		return best_entropy, best_d, best_v


class treeBasedGini:

	def split(X, y, d, value):
		index_a = (X[:, d] <= value)
		index_b = (X[:, d] > value)
		return X[index_a], X[index_b], y[index_a], y[index_b]

	def gini(y):
		counter = Counter(y)
		res = 1
		for num in counter.values():
			p = num / len(y)
			res -= p**2
		return  res

	def try_split(X, y):
		best_g = float('inf')
		best_d, best_v = -1, -1
		for d in range(X.shape[1]):
			sorted_index = np.argsort(X[:,d])
			for i in range(1, len(X)):
				if X[sorted_index[i-1],d] != X[sorted_index[i],d]:  # 相邻的样本在d维度上面不等的话
					v = (X[sorted_index[i-1], d] + X[sorted_index[i], d]) / 2
					x_l, x_r, y_l, y_r = split(x, y, d, v)
					g = gini(y_l) + gini(y_r)
					if g < best_g:
						best_g, best_d, best_v = g, d, v
						
		return best_g, best_d, best_v