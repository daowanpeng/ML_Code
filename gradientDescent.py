

"""使用梯度下降的方法来处理线性回归"""

# 损失函数
def J(theta, X_b, y):
    try:
        return np.sum((y - X_b.dot(theta))** 2) / len(X_b)
    except:
        return float('inf')
    
    
# 求导
def dJ(theta, X_b, y):
    res = np.empty(len(theta))
    res[0] = np.sum(X_b.dot(theta) - y)
    for i in range(1,len(theta)):
        res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
    return res * 2 / len(X_b)

# 求导的向量化表达形式
def dJ1(theta, X_b, y):
    return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(y)


# 梯度函数
def gradien_des(X_b, y, initial_theta, eta, n_iters = 1e4, epsilon = 1e-8):
    
    theta = initial_theta
    i_iter = 0
#     theta_his.append(theta)
    
    while i_iter < n_iters:
        gradient = dJ1(theta, X_b, y)
        last_theta = theta
        theta = theta - eta * gradient  #更新梯度
#         theta_his.append(theta)
        
        if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
            break
            
        i_iter += 1 
        
    return theta