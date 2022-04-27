import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

def load_file(file_path):
    """
    load the needed file of data
    args:
        file_path: the path of needed file
    
    returns:
        data saved as a ndarray
    """      
   
    df = pd.read_excel(file_path, sheet_name=0, header=0) # 表头为第0行
    # 聚类使用无标记样本，所以这里去除掉"是否好瓜"这一列
    data = df.values[:, 1:-1] # 所有的数据
    labels = df.values[:, -1] # 所有的标签

    return data, labels


def normalize_data(data):
    """
    normalize the given data,
    which means transforming their values into the same range

    means:
        min-max method
        z-score method
    we use z-score method here

    args:
        the loaded data

    returns:
        the normalized data
    """
    cols = data.shape[1]
    normalized_data = np.zeros(data.shape) # 初始化normalized_data array
    for i in range(cols):
        mean = np.mean(data[:, i]) # 均值
        std = np.std(data[:, i]) # 标准差
        normalized_data[:, i] = (data[:, i] - mean) / std # 标准化过程
    
    return normalized_data


def cal_centers(Xarray, groups: dict):
    """
    calculate the new centers of all clusters

    args:
        Xarray: data matrix
        group_situations: the corresponding cluster of each sample point

    returns:
        centers
    """
    N, feature_size = Xarray.shape
    k = len(groups.keys())
    centers = np.zeros([k, feature_size])
    values = list(groups.values()) # 以列表返回可遍历的(key, val)元组数组
    for i in range(k):
        samples = Xarray[values[i]]
        centers[i] = samples.mean(axis=0)
    
    return centers
    

def k_means(Xarray, k, iter=100, epsilon=1e-8):
    """
    kmeans algorithm process

    args:
        Xarray: input data
        k: the number of clusters
        iter: max iteration times
        epsilon: the threshold for the change of all mean vectors, measured by norm2 of variance

    returns:
        clusters_dict: clusters dict
        col_idx: the cluster label of each sample point
    """
    N, feature_size = Xarray.shape
    # 随机选取k个样本作为初始均值向量
    center_idx = random.sample(range(N), k)
    centers = [Xarray[ri] for ri in center_idx]
    centers = np.array(centers)

    for ii in range(iter):
        clusters_dict = {f"c{j}": [] for j in range(k)}
        print("正在迭代{}/{}".format(ii + 1, iter))

        dist = np.zeros([N, k]) # 距离矩阵
        for n in range(Xarray.shape[0]):
            dist[n] = np.sqrt(((Xarray[n].reshape(1, -1) - centers) ** 2).sum(axis=1))     
        # 求出dist矩阵中每行的最小值
        # 并把该行数据点，划分到最小值的列所对应的类
        row_idx, col_idx = np.where(dist == np.min(dist, axis=1).reshape(-1, 1))
        for i in range(N):
            clusters_dict[f"c{col_idx[i]}"].append(row_idx[i])
        
        # 更新所有均值向量
        new_centers = cal_centers(Xarray, clusters_dict)
        variance = np.linalg.norm(new_centers - centers)
        if variance <= epsilon:
            print(f"均值向量的更新已经小于: {epsilon}, 停止时迭代{ii + 1}/{iter}")
            break
        # print(variance)
        # break
        centers = new_centers

    return clusters_dict, col_idx

        
if __name__ == "__main__":
    data, labels = load_file("/root/python_codes/machine_learning_codes/kmeans/Watermelon_Database_4.0.xlsx")
    normalized_data = normalize_data(data)
    groups, cor = k_means(Xarray=normalized_data, k=3)
    print("分类结果如下: \n\t", groups)
    # matplotlib绘图
    fig = plt.figure()
    ax1 = fig.add_subplot(111) # 不加画纸，默认一张画纸
    # 设置标题
    ax1.set_title("cluster result")
    # 设置标签
    plt.xlabel("attribute1")
    plt.ylabel("attribute2")
    plt.scatter(data[:, 0], data[:, 1], c=cor, cmap=plt.cm.Spectral)
    plt.show()