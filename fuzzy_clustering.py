import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import torch.nn as nn
class FWCW_FCM:
    def __init__(self, q, p, beta_memory, fuzzy_degree, beta):
        self.q = q
        self.p = p
        self.beta_memory = beta_memory
        self.fuzzy_degree = fuzzy_degree
        self.beta = beta
    def init_cluster_c(self,init_c):
        self.init_c = init_c

    def calculate_distance(self,X, C, k, beta, distance_in, distance_inter):
        # 求参数γm
        N, d = X.shape
        x_average = np.average(X, axis=0)  # 按行求均值 1 * 4
        alpha_tem_ = []
        for j in range(d):
            alpha_tem = []
            for i in range(N):
                alpha_ = np.linalg.norm(np.array(X[i][j]) - np.array(x_average[j])) ** 2
                alpha_tem.append(alpha_)

            alpha_tem_.append(sum(alpha_tem))

        landa_m = N / np.array(alpha_tem_)
        alpha_tem1 = []
        for i in range(N):
            alpha_ = np.linalg.norm(np.array(X[i]) - np.array(x_average)) ** 2
            alpha_tem1.append(alpha_)
        landa = sum(alpha_tem1)

        # 求参数gi
        #g_i = beta / np.sum(np.power(np.linalg.norm(C - x_average,axis=1), 2))

        # 求参数gi
        par_a = []
        for i in range(k):
            tem_min = []
            for j in range(k):
                if i != j:
                    tem = np.linalg.norm(np.array(C[i]) - np.array(C[j])) ** 2
                    tem_min.append(tem)
            min_par_a = min(1 - np.exp(-1 * landa * np.array(tem_min)))  # 最小值
            par_a.append(min_par_a)  # 2个最小值

        par_b = max([1 - np.exp(-1 * landa * (np.linalg.norm(C[i, :] - x_average) ** 2)) for i in range(k)])
        g_i = (beta / 4) * (par_a / par_b)
        # 计算目标函数
        for j in range(0, k):

            distance_in[j, :, :] = (1 - np.exp((-landa_m * (X - np.tile(C[j, :], (N, 1))) ** 2)))
            # 求聚类中心c和数据点的平均值之间的距离
            distance_inter[j, :] = (1 - np.exp((-landa_m * (C[j, :] - x_average) ** 2)))

        return distance_in, distance_inter, landa_m, g_i


    def kl_loss(self, q):
        weight = (q ** 2) / np.sum(q, 0)
        p = (weight.T / np.sum(weight, 1)).T
        kl = nn.KLDivLoss(size_average=False)(torch.Tensor(q).log(), torch.Tensor(p)) #np.sum(p * (np.log(p) - q))
        return kl


    def objective_function(self, distance_in, distance_inter,  U, W, Z, p, q, g_i, fuzzy_degree):

        w_q = np.power(W, q)  # w的q次方,k*d
        z_p = np.power(Z, p)  # z的p次方,1*k
        z_p[z_p == np.inf] = 0

        dNK_in = np.squeeze(np.dot(distance_in, np.dot(w_q.T, z_p.T)))

        dNK_inter = g_i * np.squeeze(np.dot(distance_inter, np.dot(w_q.T, z_p.T)))

        OF = np.sum(np.power(U.T, fuzzy_degree) * dNK_in) - np.sum(np.power(U, fuzzy_degree) * dNK_inter)  #[k,N] *

        return OF

    def cluster_membership(self,X, W, Z, k, distance_in, distance_inter, fuzzy_degree, p, q, g_i):
        N, d = X.shape
        w_q = np.power(W, q)  # w的q次方,k*d
        z_p = np.power(Z, p)  # z的p次方,1*k
        z_p[z_p == np.inf] = 0

        dNK_in = np.squeeze(np.dot(distance_in, np.dot(w_q.T,z_p.T)))

        dNK_inter = np.squeeze(np.dot(distance_inter, np.dot(w_q.T, z_p.T)))

        tmp1 = np.zeros((N, k))  # 定义模糊隶属度矩阵
        epsilon = 1e-6

        pos_neg = dNK_in.T - np.tile(g_i * dNK_inter, (N, 1))
        row, col = np.where(pos_neg < 0)
        for i in range(0,k):
            tmp2 = np.power((dNK_in.T - np.tile(g_i * dNK_inter,(N,1)))/ ((np.tile(dNK_in[i,:],(k,1)).T - g_i[i] * dNK_inter[i]) + epsilon), 1 / (fuzzy_degree - 1))
            tmp2[tmp2 == np.inf] = 0
            tmp2[np.isnan(tmp2)] = 0
            tmp1 = tmp1 + tmp2
        Cluster_elem = 1 / (tmp1 + epsilon)  # 隶属度矩阵,【N*k】

        #特殊情况处理
        Cluster_elem[row, :] = 0.001
        Cluster_elem[row, col] = 1  #如果为负则为1

        return Cluster_elem


    def cluster_centers(self,X, Cluster_elem, C, fuzzy_degree, k, N, landa_m, g_i):

        x_average = np.average(X, axis=0)  # 按行求均值 1 * 4
        u_a = np.power(Cluster_elem, fuzzy_degree).T  # 模糊隶属度的α次方,【N*k】->(转置变成)【k*N】
        epsilon = 1e-6

        for j in range(0, k):
            #[1,N] dot [N,d] * [N,d]
            exp_in = np.dot(u_a[j,:], (np.exp(
                -1 * (np.tile(landa_m, (N, 1)) * np.power((X - np.tile(C[j, :], (N, 1))),
                                                        2))) * X))
            #【k,d】
            exp_inter = g_i[j] * np.dot(u_a[j,:], (np.exp(
                -1 * (np.tile(landa_m, (N, 1)) * np.power((np.tile(C[j, :], (N, 1)) - np.tile(x_average, (N, 1))),
                                                        2))) * x_average))

            fen_z = exp_in - exp_inter
            fen_m = np.dot(u_a[j,:], (np.exp(
                -1 * (np.tile(landa_m, (N, 1)) * np.power((X - np.tile(C[j, :], (N, 1))),
                                                        2))))) - \
                    g_i[j] * np.dot(u_a[j, :], (np.exp(
                -1 * (np.tile(landa_m, (N, 1)) * np.power((np.tile(C[j, :], (N, 1)) - np.tile(x_average, (N, 1))),
                                                        2)))))
            # new center
            C[j, :] = fen_z / (fen_m + epsilon)  # C原为k*d矩阵

        return C

    def feature_w(self, distance_in, distance_inter, Dwkm, Cluster_elem, N, fuzzy_degree, g_i, d, k, q):
        epsilon = 1e-6
        # Update the feature weights. Dwkm为k*d
        Cluster_elem = Cluster_elem.T  # 将模糊隶属度矩阵【N*K】转置成【k*N】
        for j in range(0, k):

            Dwkm[j, :] = np.dot(np.power(Cluster_elem[j, :], fuzzy_degree), np.reshape(distance_in[j, :, :], [N, d])) - g_i[j] * np.dot(np.power(Cluster_elem[j, :], fuzzy_degree), np.tile(distance_inter[j,:], (N, 1)))

        tmp1 = np.zeros((k, d))
        for j in range(0, d):

            tmp2 = np.power((Dwkm / (np.tile(Dwkm[:, j], (d, 1)).T + epsilon)), 1 / (q - 1))  # k*1扩展成k*d,【k*d】/【k*d】
            tmp2[tmp2 == np.inf] = 0
            tmp2[np.isnan(tmp2)] = 0
            tmp1 = tmp1 + tmp2

        W = 1 / (tmp1 + epsilon)
        W[np.isnan(W)] = 1
        W[W == np.inf] = 1
        if np.sum(Dwkm == 0) > 0:
            for j in range(k):
                if np.sum(Dwkm[j, :] == 0) > 0:
                    W[j, Dwkm[j, :] == 0] = 1 / np.sum(Dwkm[j, :] == 0)
                    W[j, Dwkm[j, :] != 0] = 0

        return W


    def cluster_w(self, distance_in, distance_inter, Dz, Cluster_elem, N, fuzzy_degree, g_i, W, d, k, p, q):
        Dz_ = np.ones((k, d))
        epsilon = 1e-6
        Cluster_elem = Cluster_elem.T

        # Update the cluster weights.
        for j in range(0, k):

            w_q = np.power(W[j, :], q)  # w（【k*d】）的q次方,【1*d】
            # 【1*d】【N*d】 【N*1】-> 【1*d】矩阵乘积【d*N】矩阵乘积【N*1】-》【1*1】
            Dz_[j,:] = w_q * np.dot(np.power(Cluster_elem[j, :], fuzzy_degree), np.reshape(distance_in[j, :, :], [N, d])) - g_i[j] * w_q * np.dot(np.power(Cluster_elem[j, :], fuzzy_degree), np.tile(distance_inter[j, :], (N, 1)))
            Dz = np.sum(Dz_, axis=1)  #k矩阵

        tmp1 = np.zeros(k)
        for j in range(0, k):
            tmp2 = np.power((Dz / (np.squeeze(np.tile(Dz[j], (k, 1)).T) + epsilon)), (1 / (p - 1)))  # k*1扩展成k*d,【k*d】/【k*d】
            tmp2[tmp2 == np.inf] = 0
            tmp2[np.isnan(tmp2)] = 0
            tmp1 = tmp1 + tmp2


        Z = 1 / (tmp1 + epsilon)
        Z[np.isnan(Z)] = 1
        Z[Z == np.inf] = 1
        if np.sum(Dz == 0) > 0:
            Z[Dz == 0] = 1 / np.sum(Dz == 0)
            Z[Dz != 0] = 0

        return Z

    def run(self,X):

        # 求数据的形状,N为样本个数，d为维数
        N, d = X.shape
        X = np.array(X)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        k = 15  # 去重，看分为几类
        #q = 4  # q初始化值
        #p = 0.01  # Initial p value.
        p_init = 0  # p初始化值
        p_max = 0.9 #0.5  # p最大值
        p_step = 0.05  # 每次迭代p累加的值
        p_flag = 1  #p增加与否
        iter = 1
        t_max = 300  # 最大迭代次数
        #beta_memory = 0.2  # 记忆力因子
        #fuzzy_degree = 2  # 模糊系数
        #beta = 0.08  #0.3# random.uniform(0, 1)
        tmp = np.random.permutation(N)
        #C = X[tmp[0:k], :]  # C为k*d矩阵，为初始化聚类中心
        C = self.init_c
        # kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        # y_pred = kmeans.fit_predict(X)
        # C = kmeans.cluster_centers_
        # Weights are uniformly initialized.
        W = np.ones((k, d)) / d  # initial faeture weights k*d即K*M
        Z = np.ones((1, k)) / k  # initial cluster weights 1*k
        empty = 0  # the number of an empty or singleton cluster is detected.
        O_F_old = float("inf")  # Previous iteration objective (used to check convergence).
        kl_old = float("inf")
        # 定义U,W,Z存储历史数据
        Cluster_elem_history = []  #np.empty((0, k), float)  # 存储模糊隶属度矩阵,N*K
        W_history = []  #np.empty((0, d), float)  # 存储特征权重矩阵,k*d
        Z_history = []   #np.empty((0, k), float)  # 存储聚类权重矩阵,1*k
        # 定义距离矩阵，目标函数部分矩阵，特征权重更新参考变量，聚类权重更新参考变量
        distance_in = np.ones((k, N, d))  # 定义类内紧凑性距离
        distance_inter = np.ones((k, d))  # 定义类间分离性距离
        dNK_in = np.ones((N, k))
        dNK_inter = np.ones((N, k))
        Dwkm = np.ones((k, d))
        Dz = np.ones((1,k))
        # 算法迭代
        for ss in range(1):

            distance_in, distance_inter, landa_m, g_i = self.calculate_distance(X, C, k, self.beta, distance_in, distance_inter)

            # 求模糊隶属度
            U = self.cluster_membership(X, W, Z, k, distance_in, distance_inter, self.fuzzy_degree, self.p, self.q, g_i)

            for i in range(k):
                I = np.where(U[:, i] <= 0.05)
                if len(I) == N - 1 or len(I) == N:
                    print('Empty or singleton clusters detected for p=%g.' %self.p)
                    print('Reverting to previous p value.\n')
                    if self.p < p_init or p_step == 0:
                        C = np.full((k, X.shape[1]), np.nan)
                        return U, C, W, Z
                    else:
                        empty += 1
                        self.p -= p_step
                        p_flag = 0  # Never increase p again
                        U = Cluster_elem_history[-1, :]  # 存储模糊隶属度矩阵,N*K
                        W = W_history[-1, :]  # 存储特征权重矩阵,k*d
                        Z = Z_history[-1, :]  # 存储聚类权重矩阵,1*k
                        break

            # Update the cluster centers
            C = self.cluster_centers(X, U, C, self.fuzzy_degree, k, N, landa_m, g_i)

            # Increase the p value.
            if p_flag == 1:
                Cluster_elem_history.append(np.copy(U))
                W_history.append(np.copy(W))
                Z_history.append(np.copy(Z))
                self.p = self.p + p_step
                if self.p >= p_max:
                    self.p = p_max
                    p_flag = 0

            # W_old = np.copy(W)
            # z_old = np.copy(Z)

                W = self.feature_w(distance_in, distance_inter, Dwkm, U, N, self.fuzzy_degree, g_i, d, k, self.q)

                Z = self.cluster_w(distance_in, distance_inter, Dz, U, N, self.fuzzy_degree, g_i, W, d, k, self.p, self.q)

            # Memory effect
            # W = (1 - self.beta_memory) * W + self.beta_memory * W_old
            # Z = (1 - self.beta_memory) * Z + self.beta_memory * z_old

            # 求目标函数
            kl = self.kl_loss(U)

            # if iter >= t_max or (abs(1 - kl / kl_old) < 1e-6):
            #     break

            # kl_old = kl
            # iter = iter + 1

        return U,C,kl


    def fit(self,X):
        U,C,kl = self.run(X)
        # 通过对聚类标签的每个元素加1来对齐标签
        cluster_results = np.argmax(U, axis=1) + 1

        return cluster_results,C,kl

