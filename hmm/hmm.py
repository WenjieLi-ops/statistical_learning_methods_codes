import numpy as np


def ForwardAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行、列数 可能的状态数
    M = B.shape[1]  # 数组B的列数 可能的观测数
    T = O.shape[0]  # 向量O的列数 观测序列的长度
    # 向量采用横向输入

    # 注：标准
    # 为了将观测序列用数字表达，此处使用观测概率矩阵中的列索引代表相应的状态取值
    # 此时，观测序列采用状态对应的索引值表达
    alpha = np.zeros((N, T))  # 所有的前向概率
    state = np.multiply(Pi, B[:, O[0]])  # 结果为(N,)
    # 将alpha_1加入alpha矩阵
    alpha[:, 0] = state

    for t in range(1, T):
        state = np.multiply(state.reshape(N, 1), A)
        state = np.sum(state, axis=0).reshape(N, 1)
        state = np.multiply(state, B[:, O[t]].reshape(N, 1))
        alpha[:, t] = state.reshape(N, )

    # print(np.sum(alpha, axis=0).shape) -> (3,)
    probability_O = np.sum(alpha, axis=0)[-1]
    return probability_O, alpha


def BackwardAlgo(A, B, Pi, O):
    N = A.shape[0]  # 数组A的行、列数 可能的状态数
    M = B.shape[1]  # 数组B的列数 可能的观测数
    T = O.shape[0]  # 向量O的列数 观测序列的长度
    # 向量采用横向输入
    beta = np.zeros((N, T))  # 所有的后向概率
    # initial state
    state = np.ones(N, )
    # 将beta_1加入beta矩阵
    beta[:, T - 1] = state

    for t in range(T - 1, 0, -1):
        tmp = np.multiply(B[:, O[t]].reshape(1, N), A)
        tmp = np.multiply(state.reshape(1, N), tmp)
        state = np.sum(tmp, axis=1)
        beta[:, t - 1] = state.reshape(N, )

    tmp = np.multiply(Pi.reshape(N, 1), B[:, O[0]].reshape(N, 1))
    probability_O = np.sum(np.multiply(tmp, beta[:, 0].reshape(N, 1)))
    return probability_O, beta


def getGamma(A, B, Pi, O):
    """
    获取给定模型lambda和观测O下,在时刻t处于状态qi的概率
    """
    N = A.shape[0]  # 所有可能的状态数
    T = O.shape[0]  # 观测序列的长度
    gamma = np.zeros([N, T])
    p1, alpha = ForwardAlgo(A, B, Pi, O)
    p2, beta = BackwardAlgo(A, B, Pi, O)
    gamma = np.multiply(alpha, beta) / np.sum(np.multiply(alpha, beta), axis=0)

    return gamma


def getXi(A, B, Pi, O):
    """
    获取给定模型lambda和观测O下,在时刻t处于状态qi,在时刻t+1处于状态qj的概率
    """
    N = A.shape[0]  # 所有可能的状态数
    M = B.shape[1]  # 数组B的列数 可能的观测数
    T = O.shape[0]  # 观测序列的长度
    Xi = np.zeros([T - 1, N, N])
    p1, alpha = ForwardAlgo(A, B, Pi, O)
    p2, beta = BackwardAlgo(A, B, Pi, O)

    for t in range(T - 1):
        for ni in range(N):
            for nj in range(N):
                Xi[t, ni, nj] = alpha[ni, t] * A[ni, nj] * B[nj, O[t + 1]] * beta[nj, t + 1] / p1

    return Xi


def BaumWelchTrain(A_initial, B_initial, Pi_initial, observation_sequence, iteration_n=1):
    """
    EM算法学习HMM(Baum-Welch)
    """
    N = A.shape[0]  # 所有可能的状态数
    M = B.shape[1]  # 数组B的列数 可能的观测数
    T = O.shape[0]  # 观测序列的长度

    # 初始化模型参数学习值
    A_hat = np.zeros_like(A_initial)
    B_hat = np.zeros_like(B_initial)
    Pi_hat = np.zeros_like(Pi_initial)

    for iter in range(iteration_n):
        if (not iter):
            gamma = getGamma(A_initial, B_initial, Pi_initial, observation_sequence)
            xi = getXi(A_initial, B_initial, Pi_initial, observation_sequence)
        else:
            gamma = getGamma(A_hat, B_hat, Pi_hat, observation_sequence)
            xi = getXi(A_hat, B_hat, Pi_hat, observation_sequence)
        xi_sum = xi.sum(axis=0)
        gamma_sum = np.subtract(gamma.sum(axis=1), gamma[:, T - 1])

        for ni in range(N):
            for nj in range(N):
                A_hat[ni, nj] = xi_sum[ni, nj] / gamma_sum[ni]

        tmp = 0
        for m in range(M):
            for n in range(N):
                for t in range(T):
                    if observation_sequence[t] == m:
                        tmp = tmp + gamma[n, t]
                B_hat[n, m] = tmp / gamma.sum(axis=1)[n]
                tmp = 0

        Pi_hat = gamma[:, 0]


def viterbiAlgo(state_trans, observation_probs, state_initial, observation_sequence):
    """
    给定模型和观测，求取最佳路径及其概率
    """
    state_num = state_trans.shape[0]  # 所有可能的状态数
    sequence_len = observation_sequence.shape[0]  # 序列的长度
    # 初始化相关矩阵
    delta = np.zeros([state_num, sequence_len])
    psi = np.zeros([state_num, sequence_len]).astype(int)
    # state_delta的shape为(state_num,),为一维
    state_delta = np.multiply(state_initial, observation_probs[:, observation_sequence[0]])
    delta[:, 0] = state_delta

    for t in range(1, sequence_len):
        tmp = np.multiply(state_delta.reshape([state_num, 1]), state_trans)
        tmp_max = np.max(tmp, axis=0)
        max_location = np.where(tmp == tmp_max)[0]
        state_delta = np.multiply(tmp_max, observation_probs[:, observation_sequence[t]]).reshape([state_num, ])
        # delta和psi矩阵更新
        delta[:, t] = state_delta
        psi[:, t] = max_location

    # 最佳路径的概率
    probability_optimal = np.max(delta[:, -1])
    # 最佳路径概率在delta矩阵中的行列位置
    row_loc, col_loc = np.where(delta == probability_optimal)
    path_optimal = np.append(psi[row_loc, 1:], row_loc)

    # 注意：此处的最佳路径表示为状态转移矩阵中相应状态值的索引
    return path_optimal, probability_optimal


if __name__ == '__main__':
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]
    ])
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]
    ])
    Pi = np.array(
        [0.2, 0.4, 0.4]
    )
    O = np.array(
        [0, 1, 0, 1]
    )
    path, prob = viterbiAlgo(A, B, Pi, O)
    print(path, prob, sep="\n")
    # gamma = getGamma(A, B, Pi, O)
    # print(gamma)
    # p_F = ForwardAlgo(A, B, Pi, O)
    # p_B = BackwardAlgo(A, B, Pi, O)
    # print(f'观测到序列: {O}的概率为(p_F): ', p_F)
    # print(f'观测到序列: {O}的概率为(p_B): ', p_B)
