import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
import random

def ini_centriods(dataset, k):
    n, m = dataset.shape
    the_indexes = random.sample(range(n), k)
    centriod = dataset[the_indexes]
    return centriod, the_indexes

def k_means(dataset, k, centriods):
    n, m = dataset.shape
    old_centriods = centriods
    new_centriods = np.zeros((k, m))
    clusterChanged = True
    count = 0
    t1 = time.time()
    while clusterChanged:
        clusterChanged = False
        count += 1
        dis_mat = cdist(old_centriods, dataset)
        indexes = np.argmin(dis_mat, axis=0)   # 返回每列的最小值的索引
      
        # 更新聚类中心
        for j in range(k):  # np.nonzero()返回的是非零元素的索引值
            pointsInCluster = dataset[np.nonzero(indexes == j)]  # 当其是二维数组及其以上时，用np.nonzero()[0]返回的是非零元素所在的行索引
            new_centriods[j] = np.mean(pointsInCluster, axis=0)  # axis = 0：压缩行，对各列求均值，返回 1* m 矩阵
            # axis = 1：压缩列，对各行求均值，返回 n *1 矩阵
        if not (new_centriods == old_centriods).all():
            old_centriods = new_centriods.copy()  # 一定要注意这一步  不能直接将新的中心点用等号的形式赋值给旧的中心点！！！！
            clusterChanged = True

    t2 = time.time()
    print('次数', count)
    print("时间：", t2-t1)
    return new_centriods, indexes

def showCluster(dataset, clusterAssment):
    plt.scatter(dataset[:, 0], dataset[:, 1], c=[clusterAssment])  # 尽量使用矩阵操作
    plt.show()

def readtxt_to_np(txt, n, m):

    A = np.zeros((n, m), dtype=float)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型
    with open(txt) as f:  # 打开数据文件文件
        lines = f.readlines()  # 把全部数据文件读到一个列表lines中
        A_row = 0  # 表示矩阵的行，从0行开始
        for line in lines:  # 把lines中的数据逐行读取出来
            list = line.strip('\n').split(' ')  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
            A[A_row, :] = list[0:m]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
            A_row += 1
    return A


if __name__ == '__main__':
    # fp = sio.loadmat("dataset16.mat")
    # data = fp['fourclass']
    # dataset = data[:, 1:]
    fp = np.load('susy.npy')
    dataset = fp[:500000, :]

    # data = np.loadtxt('kegg_shuffled_normal.txt')
    # dataset = data[:, :]

    # np.set_printoptions(threshold=np.nan)  # 由于pycharm在输出数据量较大的时候，会省略一些值输出，这一句的作用是将输出所有的值包括被省略的值
    # np.seterr(divide='ignore', invalid='ignore')  # 忽略无效值在计算中的报错
    n, m = dataset.shape
    k = 30
    new_centriods = np.zeros((k, m))

    # centriods,the_indes = ini_centriods(dataset, k)  # 初始化得到的中心点 ===》 矩阵的形式
    # print(the_indes)

    center_indexs = [374734, 323084, 205473, 244685, 155006, 274126, 267758, 114821, 73095, 321509, 429066, 202318, 312931, 36423, 303841,
           190750, 374713, 92800, 171648, 248439, 110826, 309055, 410903, 368877, 211488, 183198, 70535, 11979, 165176, 293411]
    centriods = dataset[center_indexs]
    now_centriods, indexes = k_means(dataset, k, centriods)  # 通过k_means函数得到的新的聚类中心
    # print("中心点：")
    print(now_centriods)
    # print('-------------------')
    # print(now_centriods, indexes)
    # showCluster(dataset, indexes)


