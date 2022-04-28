import numpy as np
from scipy.stats import multivariate_normal


class GMM(object):
    def __init__(self, data, k, weights=None, means=None, covars=None, max_iter=2000, tol=1e-3) -> None:
        """
        GMM(多元高斯混合模型)类的构造函数
        :param data: 训练数据 (n, feature_size) feature_size元(维)高斯分布
        :param k: 高斯分布的个数
        :param weights: 各高斯分布的初始概率向量
        :param means: 高斯分布的均值向量
        :param covars: 高斯分布的协方差矩阵
        :param max_iter: 最大迭代次数
        :param tol: EM迭代停止阈值
        """
        self.data = data
        self.n, self.feature_size = np.shape(self.data)
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

        # 给定weights, means, covars的初始化值
        self.params_initialize(weights, means, covars)

    def params_initialize(self, weights, means, covars):
        """
        初始化weights, means, covars
        """
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(self.k)
            self.weights /= np.sum(self.weights)

        if means is not None:
            self.means = means
        else:
            self.means = np.random.rand(self.k, self.feature_size)

        if covars is not None:
            self.covars = covars
        else:
            self.covars = np.zeros([self.k, self.feature_size, self.feature_size])
            for k_idx in range(self.k):
                self.covars[k_idx] = np.eye(self.feature_size) * np.random.rand(1) * self.k

    def GMM_EM_iteration(self):
        """
        利用EM算法迭代优化GMM参数
        """
        # 定义Q函数初始值
        Q_function_old = 0
        for it in range(self.max_iter):
            print(f"==========正在迭代{it + 1}/{self.max_iter}==========")
            # E步
            # self.data.shape = (数据个数self.n, 数据维度self.feature_size)
            # 对于第k个模型来说，数据个数为self.n个，将产生self.n个值
            density = np.zeros([self.n, self.k])  # 所有self.k个模型对于数据的概率密度值
            self.posterior = np.zeros([self.n, self.k])  # 用以记录第n个样本来自第k个高斯模型的概率

            for k_idx in range(self.k):
                # 产生第k个多维正态分布随机变量
                rv = multivariate_normal(mean=self.means[k_idx], cov=self.covars[k_idx])
                density[:, k_idx] = rv.pdf(self.data)  # self.data的每一行作为一个数据点坐标带入pdf函数，得到一个概率密度函数值
            # 计算所有样本属于每一类别的后验概率
            self.posterior = density * self.weights  # (self.n, self.k) * (self.k)
            self.posterior /= self.posterior.sum(axis=1, keepdims=True)  # keepdims是的求和后结果仍然保持二维属性

            # 计算Q函数的值，也即完全数据的对数似然函数的期望值
            weights_hat = self.posterior.sum(axis=0)  # shape = (1, self.k)
            Q_function_new = weights_hat * np.log(self.weights) + (self.posterior * np.log(density)).sum(axis=0)
            Q_function_new = Q_function_new.sum()

            # M步
            means_hat = np.tensordot(self.posterior, self.data, axes=[0, 0])
            # 计算协方差
            covars_hat = np.zeros(self.covars.shape)  # shape = (self.k, self.feature_size, self.feature_size)
            for k_idx in range(self.k):
                tmp = self.data - self.means[k_idx]
                covars_hat[k_idx] = np.dot(tmp.T * self.posterior[:, k_idx], tmp) / weights_hat[k_idx]
            # 更新参数
            self.covars = covars_hat
            self.means = means_hat / weights_hat.reshape(-1, 1)
            self.weights = weights_hat / self.n

            # 判断Q函数的变化是否足够小，是否满足停止条件
            if abs(Q_function_new - Q_function_old) > self.tol or Q_function_old == 0:
                Q_function_old = Q_function_new
            else:
                print(f"Q函数的变化值{abs(Q_function_new - Q_function_old)} < {self.tol}, 满足停止条件")
                print(f"停止时迭代次数{it + 1}/{self.max_iter}")
                break


if __name__ == '__main__':
    # 随机生成测试数据2000个
    # 使用3个二元高斯分布
    true_weights = np.array([0.3, 0.6, 0.1])  # 600, 1200, 200
    true_means = np.array([
        [2.5, 8],
        [8, 2.7],
        [9, 10]
    ])
    true_covars = np.array([
        [[2, 1], [1, 2]],
        [[3, 2], [2, 3]],
        [[2, 0], [0, 2]]
    ])
    # print(multivariate_normal(mean=true_means[0], cov=true_covars[0]))
    # <scipy.stats._multivariate.multivariate_normal_frozen object at 0x0000020978E1B9D0>
    X = np.concatenate([
        np.random.multivariate_normal(mean=true_means[0], cov=true_covars[0], size=(600,)),
        np.random.multivariate_normal(mean=true_means[1], cov=true_covars[1], size=(1200,)),
        np.random.multivariate_normal(mean=true_means[2], cov=true_covars[2], size=(200,))
    ])
    # print(X)
    np.random.shuffle(X)
    gmm = GMM(X, 3, tol=1e-9)
    gmm.GMM_EM_iteration()
    print('各模型权重:', gmm.weights, \
    '各模型均值:', gmm.means, \
    '各模型协方差矩阵:' ,gmm.covars, sep="\n")
