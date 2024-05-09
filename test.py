"""
Test cases for clustering models.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from new import CRLI
from metrics import cal_rand_index, cal_cluster_purity
from cluster_metrics import calculate_ACC,calculate_Purity,calculate_Compactness,calculate_Separation
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
import torch
import torch.nn as nn
import scipy.signal


class TrainCrliCluster(nn.Module):
    def __init__(self):
        super(TrainCrliCluster, self).__init__()
        self.epoch = 400
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        self.batch_size = 32
        self.rnn_cell_type = "GRU"
        self.learning_rate = 0.001
        self.rnn_hidden_size = 50
        self.n_generator_layers = 2
        self.features_dims = 4
        self.d_hidden = 40
        self.m = 2
        self.beta = 0.08


    def forward(self,train,test,n_step,n_feature,class_num):
        print("Running test cases for CRLI...")
        self.crli = CRLI(
            n_steps=n_step,
            n_features=n_feature,
            n_clusters=train["n_classes"],
            n_generator_layers=self.n_generator_layers,  # 1
            rnn_hidden_size=self.rnn_hidden_size,
            features_dims=self.features_dims,
            class_nums=class_num,
            epochs=self.epoch,
            batch_size=self.batch_size,
            rnn_cell_type=self.rnn_cell_type,
            learning_rate=self.learning_rate,
            device=self.device,
            d_hidden=self.d_hidden,
            m=self.m,
            beta=self.beta,

        )
        self.crli.fit(train)  #训练
        cluster_label, true_label = self.crli.clustering(test)  # 预测
        # compactness = round(calculate_Compactness(cluster_elem, C),4)
        # seperation = round(calculate_Separation(C),4)
        ri = round(cal_rand_index(cluster_label, true_label),4)   #取值范围为[0,1]
        acc = round(calculate_ACC(true_label, cluster_label), 4)
        purity = round(cal_cluster_purity(cluster_label, true_label),4)  #取值范围为[0,1]
        #purity = round(calculate_Purity(true_label, clustering),4)
        ari = round(adjusted_rand_score(true_label, cluster_label),4)   #取值范围为[-1,1]
        nmi = round(normalized_mutual_info_score(true_label, cluster_label,average_method='arithmetic'),4)  #取值范围为[0,1]
        f1 = round(f1_score(true_label, cluster_label,average='weighted',zero_division='warn'),4)

        print("真实的标签值：",true_label)
        print("预测的标签值：", cluster_label)
        print("真实标签去重之后:", set(true_label))
        print("聚类标签去重之后:",set(cluster_label))
        print("真实的标签值形状：", true_label.shape)
        print("预测的标签值形状：", cluster_label.shape)

        print(f"RI: {ri}\nacc: {acc}\npurity:{purity}\nari: {ari}\nnmi: {nmi}\nf1: {f1}")
        metrics_clustering = pd.DataFrame({"ri": ri, "acc": acc, "purity": purity,"ari": ari, "nmi": nmi, "f1": f1}, index=[0])
        #print(f"Compactness:{compactness}\nSeperation:{seperation}\nRI: {ri}\nCP: {cp}\nacc: {acc}\npurity:{purity}\nari: {ari}\nnmi: {nmi}\nf1: {f1}")
        #metrics_clustering = pd.DataFrame({"compactness":compactness,"seperation":seperation,"ri": ri, "cp": cp, "acc": acc,"purity":purity,"ari":ari,"nmi":nmi,"f1":f1},index=[0])
        #metrics_clustering.to_excel(r'C:/Users/lyr/Desktop/202331/CRLI/metrics_clustering.xls')
        # pd.DataFrame(train["y"]).to_excel(r'C:/Users/lyr/Desktop/202331/CRLI/true_label.xls')
        # pd.DataFrame(clustering).to_excel(r'C:/Users/lyr/Desktop/202331/CRLI/pred_label.xls')

def mask(data,miss_rate):
    data = np.array(data)
    num, rows, cols = data.shape
    unif_random_matrix = np.random.uniform(0., 1., size=[num, rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < 1 - miss_rate)  # 80%为1,20%为0
    data[binary_random_matrix == 0] = torch.nan  # 20%为True,赋值为空值，即20%的缺失率
    miss_data = torch.tensor(data)
    return miss_data

def data_dims(file):
    '''dim of multivariable dataset'''
    dims = dict()
    with open('../dims.txt') as f:
        for line in f:
            tmps = line[:-1].split(',')
            dims[tmps[0]] = int(tmps[1])
    return dims[file]

def missing_ratio(file):
    '''dim of multivariable dataset'''
    ratio = dict()
    with open('../missing_ratio.txt') as f:
        for line in f:
            tmps = line[:-1].split(',')
            ratio[tmps[0]] = float(tmps[1])
    return ratio[file]


# 归一化
def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data

# 中值滤波去噪
def median_filtering(data, window_size):
    filtered_data = scipy.signal.medfilt(data, kernel_size=window_size)

    return filtered_data

# 特征缩放
def norm(x_train,x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train,x_test

def wgn(x, snr):
    Ps = np.sum(abs(x)**2)/len(x)
    Pn = Ps/(10**((snr/10)))
    noise = np.random.randn(*x.shape) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise


def load_datas(x,y,n_feature,miss_rate):
    inputs_data = {}
    # data_label = np.loadtxt(path, delimiter=",")
    # pd.DataFrame(data_label).to_csv("data_label.txt", header=0, index=0)
    # data = pd.DataFrame(data_label[:, 1:].astype(np.float32))
    # label = pd.DataFrame(data_label[:, 0].astype(np.int32))
    # model_smote = SMOTE(n_jobs=-1)
    # data, label = model_smote.fit_resample(data, label)  # 输入数据进行过抽样处理
    # smote_resampled = pd.concat([label, data], axis=1)  # 将特征和标签重新拼接
    # smote_resampled.to_csv("data.txt",header=0,index=0)
    # with open(path, "r", encoding="utf-8") as f:
    #     datas = f.readlines()
    # with open("data.txt", "r", encoding="utf-8") as f:
    #     datas = f.readlines()


    # train_ys = []
    # train_xs = []
    # for data in datas:
    #     train_x = []
    #     data = data.strip().split(",")
    #     x, y = data[1:], data[0]
    #     x = [float(i) for i in x]
    #     train_ys.append(int(float(y)))
    #     for i in range(n_step):
    #         train_x.append(x[i * n_feature:(i + 1) * n_feature])
    #     train_xs.append(train_x)
    # reshape into time series samples
    X = x.reshape(x.shape[0],-1, n_feature)
    inputs_data["x"] = torch.tensor(X, dtype=torch.float)   #原始数据
    inputs_data["miss_x"] = mask(inputs_data["x"], miss_rate)  #带有缺失值的数据
    inputs_data["y"] = torch.tensor(y, dtype=torch.int)
    inputs_data["n_classes"] = len(set(y))

    # inputs_data["x"] = torch.tensor(train_xs, dtype=torch.float)
    # inputs_data["miss_x"] = mask(inputs_data["x"], miss_rate)
    # inputs_data["y"] = torch.tensor(train_ys, dtype=torch.int)
    # inputs_data["n_classes"] = len(set(train_ys))
    # data_size = len(train_ys)
    return inputs_data

# # missing
# class Mask(nn.Module):
#     def __init__(self,config):
#         super(Mask, self).__init__()
#         self.miss_rate = config.miss_rate
# def mask(self,data):
#     data = np.array(data)
#     rows,cols = data.shape
#     unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
#     binary_random_matrix = 1 * (unif_random_matrix < 1 - self.miss_rate) #80%为1,20%为0
#     data[binary_random_matrix == 0] = np.nan #20%为True,赋值为空值，即20%的缺失率
#     miss_data = data
#     return miss_data

# def load_datas(path,config):
#     inputs_data = {}
#     batch_size = config.batch_size
#
#     data_label = np.loadtxt(path, delimiter=",")
#     x= data_label[:, 1:].astype(np.float32)
#     y = data_label[:, 0].astype(np.int32)
#     # with open(path, "r", encoding="utf-8") as f:
#     #     datas = f.readlines()
#     mask = Mask(config)
#     x = mask(x)
#     # train_xs = []
#     # train_ys = []
#     # train_mask = []
#     samples_num = x.shape[0]
#     # for data in datas:
#     #     data = data.strip().split(",")
#     #     x, y = data[1:], data[0]
#     batchs_data = []
#     batchs_label = []
#     # batchs_mask = []
#     for i in range(samples_num):
#         if samples_num-i > batch_size:
#             batch_data = x[i: i + batch_size, :]
#             batch_label = y[i: i + batch_size]
#             # batch_mask = [0 if data == "None" else 1 for seq in batch_data for data in seq]
#
#             batchs_data.append(batch_data)
#             batchs_label.append(batch_label)
#             # batchs_mask.append(np.array(batch_mask).reshape(batch_size,-1))
#         # else:
#         #     batch_data = x[i: samples_num, :]
#         #     batch_label = y[i: samples_num]
#         #     # batch_mask = [0 if data == "None" else 1 for seq in batch_data for data in seq]
#         #
#         #     batchs_data.append(batch_data)
#         #     batchs_label.append(batch_label)
#             # batchs_mask.append(np.array(batch_mask).reshape(samples_num-i, -1))
#         # train_xs.append(batchs_data)
#         # train_ys.append(batchs_label)
#         # train_mask.append(batchs_mask)
#
#     inputs_data["x"] = np.array(batchs_data, dtype=np.float32)
#     inputs_data["y"] = np.array(batchs_label, dtype=np.int32)
#     print(inputs_data["x"].shape)
#     print(inputs_data["y"].shape)
#     inputs_data["n_classes"] = len(set([i for _ in batchs_label for i in _ ]))  #len(set(i for i in batchs_label))
#
#     # inputs_data["mask"] = np.array(train_mask, dtype=np.int32)
#
#     return inputs_data


if __name__ == "__main__":

    data_path = "./dataset/MTS/Libras/"
    train_path = data_path + "Libras_TRAIN.npy"
    test_path = data_path + "Libras_TEST.npy"
    train_label = data_path + "Libras_TRAIN_label.txt"
    test_label = data_path + "Libras_TEST_label.txt"
    x_train = np.load(train_path, allow_pickle=True)
    y_train = np.loadtxt(train_label)
    x_test = np.load(test_path, allow_pickle=True)
    y_test = np.loadtxt(test_label)


    #获取数据的形状
    n_train_samples, n_step, n_feature = x_train.shape
    n_test_samples, _, _ = x_test.shape
    x_train = x_train.reshape(n_train_samples, -1)
    x_test = x_test.reshape(n_test_samples, -1)
    # x_test = wgn(x_test,60)   # snr值越大噪声越小
    # 添加20%的噪声
    # noise_percentage = 10
    # noise_amplitude = 0.2  # 调整噪声振幅以控制噪声水平
    # # 生成噪声
    # noise = np.random.normal(loc=0, scale=noise_amplitude, size=x_test.shape)
    # # 将噪声添加到时间序列数据中
    # x_test = x_test + noise_percentage / 100 * x_test * noise

    x_train = min_max_normalization(x_train)
    x_test = min_max_normalization(x_test)
    #x_train, x_test = norm(x_train, x_test)
    x_train = median_filtering(x_train, 5)
    x_test = median_filtering(x_test, 5)

    miss_ratio = 0.2
    class_num = len(list(set(y_train)))
    train_data = load_datas(x_train, y_train, n_feature, miss_ratio)
    test_data = load_datas(x_test, y_test, n_feature, miss_ratio)


    # data_path = "./dataset/UCR_TS/synthetic_control/"
    # train_path = data_path + "synthetic_control_TRAIN"
    # test_path = data_path + "synthetic_control_TEST"
    # name = data_path.split("/")[-2]
    # datas_train = np.loadtxt(train_path, delimiter=",")
    # datas_test = np.loadtxt(test_path, delimiter=",")
    # x_train = datas_train[:, 1:]
    # y_train = datas_train[:, 0]
    # x_test = datas_test[:, 1:]
    # y_test = datas_test[:, 0]
    # # 使用 vstack 合并特征
    # x_all = np.vstack((x_train, x_test))
    #
    # # 使用 vstack 合并标签
    # y_all = np.hstack((y_train, y_test))
    # # x_test = wgn(x_test, 90)  # snr值越大噪声越小
    # # 添加20%的噪声
    # # noise_percentage = 10
    # # noise_amplitude = 0.2  # 调整噪声振幅以控制噪声水平
    # # # 生成噪声
    # # noise = np.random.normal(loc=0, scale=noise_amplitude, size=x_test.shape)
    # # # 将噪声添加到时间序列数据中
    # # x_test = x_test + noise_percentage / 100 * x_test * noise
    #
    # x_train = min_max_normalization(x_train)
    # x_test = min_max_normalization(x_test)
    # x_train = median_filtering(x_train, 5)
    # x_test = median_filtering(x_test, 5)
    # n_feature = 1  #int(data_dims(train_data_path.split('/')[2]))
    # n_step = x_train.shape[1] #int(data.shape[1] // n_feature)
    # miss_ratio = 0.1  #missing_ratio(train_data_path.split('/')[2])
    # class_num = len(list(set(y_train)))
    # train_data = load_datas(x_train, y_train, n_feature, miss_ratio)
    # test_data = load_datas(x_test, y_test, n_feature, miss_ratio)

    #data = load_datas(x_all, y_all.tolist(), n_feature, miss_ratio)

    crli = TrainCrliCluster()
    crli(train_data, test_data, n_step, n_feature, class_num)
