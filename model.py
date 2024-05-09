"""
Torch implementation of CRLI (Clustering Representation Learning on Incomplete time-series data).

Please refer to :cite:``ma2021CRLI``.
"""

from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from math import sqrt
from times_dataset import TimeSeriesDataset
from metrics import cal_mse
import matplotlib.pyplot as plt
from kl_fuzzy import FWCW_FCM
from sklearn.cluster import KMeans
from find_optimal import find_parameters
from random import sample
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
from metrics import cal_rand_index, cal_cluster_purity
from cluster_metrics import calculate_ACC
from sklearn.metrics.cluster import normalized_mutual_info_score



RNN_CELL = {
    "LSTM": nn.LSTMCell,
    "GRU": nn.GRUCell,
}


def reverse_tensor(tensor_):
    if tensor_.dim() <= 1:
        return tensor_
    indices = range(tensor_.size()[1])[::-1]
    indices = torch.tensor(
        indices, dtype=torch.long, device=tensor_.device, requires_grad=False
    )
    return tensor_.index_select(1, indices)


class MultiRNNCell(nn.Module):
    def __init__(self, cell_type, n_layer, d_input, d_hidden, device):
        super(MultiRNNCell, self).__init__()
        self.cell_type = cell_type
        self.n_layer = n_layer
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.device = device

        self.model = nn.ModuleList()
        if cell_type in ["LSTM", "GRU"]:
            for i in range(n_layer):
                if i == 0:
                    self.model.append(RNN_CELL[cell_type](d_input, d_hidden))
                else:
                    self.model.append(RNN_CELL[cell_type](d_hidden, d_hidden))

        self.output_layer = nn.Linear(d_hidden, d_input)

    def forward(self, inputs):
        X, missing_mask = inputs["X"], inputs["missing_mask"]
        missing_mask = missing_mask.to(self.device)
        bz, n_steps, _ = X.shape
        hidden_state = torch.zeros((bz, self.d_hidden), device=self.device)
        hidden_state_collector = torch.empty(
            (bz, n_steps, self.d_hidden), device=self.device
        )
        pre_collector = torch.empty((bz, n_steps, self.d_input), device=self.device)
        output_collector = torch.empty((bz, n_steps, self.d_input), device=self.device)
        if self.cell_type == "LSTM":
            # TODO: cell states should have different shapes
            cell_states = torch.zeros((self.d_input, self.d_hidden), device=self.device)
            for step in range(n_steps):
                x = X[:, step, :]
                x = x.to(self.device)
                estimation = self.output_layer(hidden_state)
                pre_collector[:, step,:] = estimation
                #output_collector[:, step] = estimation
                imputed_x = (
                        missing_mask[:, step] * x + (1 - missing_mask[:, step]) * estimation
                )
                output_collector[:, step, :] = imputed_x
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state, cell_states = self.model[i](
                            imputed_x, (hidden_state, cell_states)
                        )
                    else:
                        hidden_state, cell_states = self.model[i](
                            hidden_state, (hidden_state, cell_states)
                        )
                hidden_state_collector[:, step, :] = hidden_state

        elif self.cell_type == "GRU":
            for step in range(n_steps):
                x = X[:, step, :]
                x = x.to(self.device)
                estimation = self.output_layer(hidden_state)
                pre_collector[:, step, :] = estimation
                #output_collector[:, step] = estimation
                imputed_x = (
                        missing_mask[:, step] * x + (1 - missing_mask[:, step]) * estimation
                )
                output_collector[:, step, :] = imputed_x
                for i in range(self.n_layer):
                    if i == 0:
                        hidden_state = self.model[i](imputed_x, hidden_state)
                    else:
                        hidden_state = self.model[i](hidden_state, hidden_state)

                hidden_state_collector[:, step, :] = hidden_state

        output_collector = output_collector[:, 1:]
        estimation = self.output_layer(hidden_state).unsqueeze(1)
        output_collector = torch.concat([output_collector, estimation], dim=1)
        return pre_collector,output_collector, hidden_state, hidden_state_collector


# 定义深度融合层
class DeepFusionLayer(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(DeepFusionLayer, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"
        self.fc = nn.Linear(sum(input_dims), output_dim).to(self.device)
    def forward(self, X):
        fused_features = self.fc(X)

        return fused_features


# 定义统计特征自编码器
class StatAutoencoder(nn.Module):
    def __init__(self, n_steps, input_dim, encoding_dim):
        super(StatAutoencoder, self).__init__()
        self.seq_length = n_steps
        self.encoder = nn.Linear(input_dim * self.seq_length, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim * self.seq_length)
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"

    def forward(self, inputs):
        X = inputs["X"].to(self.device)
        _, seq_length, input_dim = X.shape
        X = X.view(-1, input_dim * seq_length)
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, seq_length, input_dim)
        inputs["StatAutoencoder_X"] = encoded
        inputs["StatAutoencoder_decoded"] = decoded
        return inputs

# 定义频域特征自编码器
class FreqAutoencoder(nn.Module):
    def __init__(self, n_steps, input_dim, encoding_dim):
        super(FreqAutoencoder, self).__init__()
        self.seq_length = n_steps
        self.input_dim = input_dim
        self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * self.seq_length, encoding_dim)  # 添加全连接层
        # 反卷积层用于将编码后的特征重建为频域特征（可选）
        self.deconv1 = nn.Conv1d(encoding_dim, input_dim, kernel_size=3, padding=1)
        # self.encoder = nn.Conv1d(input_dim, encoding_dim, kernel_size=3, padding=1)
        # self.decoder = nn.Conv1d(encoding_dim, input_dim, kernel_size=3, padding=1)
        self.device = 'cuda:0' if torch.cuda.is_available() else "cpu"

    def forward(self, inputs):
        x = self.conv1(inputs["X"].permute(0, 2, 1))
        x = x.reshape(-1, 16 * self.seq_length)  # 展平特征图，转换为一维张量
        encoded = self.fc(x)
        # 添加维度以适应反卷积层
        encoded_ = encoded.reshape(-1, encoded.size(1), 1)
        # 反卷积层用于重建原始频域特征（可选）
        decoded = self.deconv1(encoded_)
        decoded = decoded.permute(0, 2, 1)

        inputs["FreqAutoencoder_X"] = encoded
        inputs["FreqAutoencoder_decoded"] = decoded
        return inputs

# 定义时域特征自编码器
class Generator(nn.Module):
    def __init__(self, n_layers, n_features, d_hidden, features_dims, class_nums, cell_type, device):
        super().__init__()
        self.f_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden, device)
        self.b_rnn = MultiRNNCell(cell_type, n_layers, n_features, d_hidden, device)
        self.linear_latent = nn.Linear(d_hidden * 2,features_dims)
        self.bn = torch.nn.BatchNorm1d(features_dims)
        self.drop = torch.nn.Dropout(0.2)
        self.device = device

    def forward(self, inputs):
        pref_outputs, f_outputs, f_final_hidden_state, f_hidden_state_collector = self.f_rnn(inputs)
        inputs["X"] = inputs["X"].to(self.device)
        inputs["missing_mask"] = inputs["missing_mask"].to(self.device)
        backward = {'X': reverse_tensor(inputs["X"]), "missing_mask": reverse_tensor(inputs["missing_mask"])}
        preb_outputs, b_outputs, b_final_hidden_state, b_hidden_state_collector = self.b_rnn(backward)  # todo


        pred = (pref_outputs + preb_outputs) / 2
        imputed_X = (f_outputs + b_outputs) / 2

        # H is the concatenation of the last hidden state of the forward and backward RNN.
        fb_final_hidden_states = torch.concat(
            [f_final_hidden_state, b_final_hidden_state], dim=-1
        )

        fb_hidden_state_collector = f_hidden_state_collector

        latent_representation = self.drop(self.bn(torch.tanh(self.linear_latent(fb_final_hidden_states))))

        inputs["imputation"] = pred
        inputs["imputed_X"] = imputed_X
        inputs["generator_fb_hidden_states"] = fb_final_hidden_states
        inputs["encoder_outputs"] = fb_hidden_state_collector
        inputs["fcn_latent"] = latent_representation

        return inputs


class Discriminator(nn.Module):
    def __init__(self, cell_type, d_input, d_hidden, device):
        super().__init__()
        self.cell_type = cell_type
        self.device = device
        self.d_input = d_input
        self.d_hidden = d_hidden
        # this setting is the same with the official implementation
        self.rnn_cell_module_list = nn.ModuleList(
            [
                RNN_CELL[cell_type](d_input, 32),
                RNN_CELL[cell_type](32, 16),
                RNN_CELL[cell_type](16, 8),
                RNN_CELL[cell_type](8, 16),
                RNN_CELL[cell_type](16, 32),
            ]
        )
        self.output_layer = nn.Linear(32, d_input)

    def forward(self, inputs):
        imputed_X = inputs["imputed_X"]
        bz, n_steps, _ = imputed_X.shape
        hidden_states = [
            torch.zeros((bz, 32), device=self.device),
            torch.zeros((bz, 16), device=self.device),
            torch.zeros((bz, 8), device=self.device),
            torch.zeros((bz, 16), device=self.device),
            torch.zeros((bz, 32), device=self.device),
        ]
        hidden_state_collector = torch.empty((bz, n_steps, 32), device=self.device)
        if self.cell_type == "LSTM":
            cell_states = torch.zeros((self.d_input, self.d_hidden), device=self.device)

            for step in range(n_steps):
                x = imputed_X[:, step, :]
                x = x.to(self.device)
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state, cell_state = rnn_cell(
                            x, (hidden_states[i], cell_states)
                        )
                    else:
                        hidden_state, cell_state = rnn_cell(
                            hidden_states[i - 1], (hidden_states[i], cell_states)
                        )
                    hidden_states[i] = hidden_state
                    # cell_states[i] = cell_state
                hidden_state_collector[:, step, :] = hidden_state

        elif self.cell_type == "GRU":
            for step in range(n_steps):
                x = imputed_X[:, step, :]
                x = x.to(self.device)
                for i, rnn_cell in enumerate(self.rnn_cell_module_list):
                    if i == 0:
                        hidden_state = rnn_cell(x, hidden_states[i])
                    else:
                        hidden_state = rnn_cell(hidden_states[i - 1], hidden_states[i])
                    hidden_states[i] = hidden_state
                hidden_state_collector[:, step, :] = hidden_state

        output_collector = self.output_layer(hidden_state_collector)
        return output_collector


class Decoder(nn.Module):
    def __init__(
            self, n_steps, d_input, d_output, device, d_hidden):#fcn_output_dims: list = None):
        super().__init__()
        self.n_steps = n_steps
        self.d_output = d_output
        self.device = device
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.rnn_cell_1 = nn.GRUCell(d_input, d_hidden)
        self.rnn_cell_2 = nn.GRUCell(d_hidden, d_hidden)
        self.attention_layer1 = nn.Linear(d_input * 2, d_input)
        self.attention_layer2 = nn.Linear(d_hidden * 2, d_hidden)
        self.output_layer = nn.Linear(d_hidden, d_output)
        self.drop_out = nn.Dropout(0.2)

    def forward(self, inputs):

        fcn_latent = inputs["fcn_latent"]
        hidden_state = fcn_latent  # 提取的中间特征,做为decoder的初始隐藏状态
        output_encoder = inputs["encoder_outputs"]
        bz, dim = fcn_latent.shape
        hidden_state_collector = torch.empty(
            (bz, self.n_steps, self.d_hidden), device=self.device
        )


        for i in range(self.n_steps):
            scores = torch.matmul(hidden_state.transpose(0, 1), output_encoder[:,i,:])   #[b,e_d] * [b,d]=>[e_d,b] * [b*d]=[e_d,d]
            weight = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(weight, output_encoder[:,i,:].transpose(0, 1)) #[e_d,d] * [d,b] = [e_d,b]
            #context_layers = torch.add(torch.matmul(weight, attention_data).squeeze(1), hidden_state)
            #hidden_state = self.rnn_cell(context_layers, hidden_state)
            if i == 0:
                attention = torch.tanh(self.attention_layer1(torch.concat((context.transpose(0, 1), hidden_state),
                                                                         dim=1)))  # [b,e_d] + [b,e_d]=[b,2*e_d]->[b,e_d]
                hidden_state = hidden_state.repeat(1, int(self.d_hidden/self.d_input))  #[b,5*e_d]
                hidden_state = self.rnn_cell_1(attention, hidden_state)  #[2 * e_d,d]
            else:
                attention = torch.tanh(self.attention_layer2(torch.concat((context.transpose(0, 1), hidden_state),
                                                                          dim=1)))  # [b,d] + [b,d]=[b,2*d]->[b,d]
                hidden_state = self.rnn_cell_2(attention, hidden_state)  # [2 * e_d,d]
            hidden_state_collector[:, i, :] = hidden_state
        reconstruction = self.output_layer(hidden_state_collector) #[d,d_out]

        return reconstruction


class _CRLI(nn.Module):
    def __init__(
            self,
            n_steps,
            n_features,
            n_clusters,
            n_generator_layers,
            rnn_hidden_size,
            features_dims,
            class_nums,
            d_hidden, #decoder_fcn_output_dims,
            lambda_kmeans,
            rnn_cell_type,
            device,
            m,
            beta,
    ):
        super(_CRLI,self).__init__()
        self.generator = Generator(
            n_generator_layers, n_features, rnn_hidden_size, features_dims, class_nums, rnn_cell_type, device
        )
        self.statautoencoder = StatAutoencoder(n_steps,n_features,features_dims)
        self.freqautoencoder = FreqAutoencoder(n_steps,n_features,features_dims)
        self.discriminator = Discriminator(rnn_cell_type, n_features, rnn_hidden_size, device)
        self.decoder = Decoder(n_steps, features_dims, n_features, device, d_hidden)
        self.kmeans = KMeans(n_clusters=class_nums)  # TODO: implement KMean with torch for gpu acceleration
        self.n_clusters = n_clusters
        self.lambda_kmeans = lambda_kmeans
        self.device = device
        self.features_dims = features_dims

    def parameter(self,concat_features,concat_labels):
        # 找最优参数
        true_label = concat_labels.to(self.device)
        optimal_parameters, C = find_parameters(concat_features.cpu().detach().numpy(),
                                            true_label.cpu().detach().numpy())
        # 用最优参数创建聚类
        self.clusterings = FWCW_FCM(**optimal_parameters)
        # class_nums = len(set(true_label.cpu().detach().numpy()))
        # init_indices = sample(range(len(concat_features.cpu().detach().numpy())), k=class_nums)
        # init_samples = np.stack([concat_features.cpu().detach().numpy()[i] for i in init_indices])
        # C = init_samples.reshape(class_nums, -1)
        # self.clusterings = FWCW_FCM(q=4, p=0.1, beta_memory=0.1, fuzzy_degree=1.5, beta=0.1)

        self.clusterings.init_cluster_c(C)

    def forward(self, inputs, training_object="generator", fea=False, recon=False):
        assert training_object in [
            "generator",
            "discriminator",
        ], 'training_object should be "generator" or "discriminator"'

        X = inputs["X"].to(self.device)
        missing_mask = inputs["missing_mask"].to(self.device)
        true_label = inputs["y"].to(self.device)
        batch_size, n_steps, n_features = X.shape
        # 损失字典
        losses = {}
        # 运行生成器
        inputs = self.generator(inputs)
        inputs = self.statautoencoder(inputs)
        inputs = self.freqautoencoder(inputs)
        input_dims = [inputs["StatAutoencoder_X"].size(1), inputs["FreqAutoencoder_X"].size(1),
                      inputs["fcn_latent"].size(1)]
        output_dim = self.features_dims  # 设定融合后的特征维度
        fusion_layer = DeepFusionLayer(input_dims, output_dim)
        # 将三种类型的特征输入到深度融合层进行融合
        fea_X = torch.cat([torch.as_tensor(inputs["StatAutoencoder_X"]), torch.as_tensor(inputs["FreqAutoencoder_X"]),
                           torch.as_tensor(inputs["fcn_latent"])], dim=1).to(self.device)
        combined_features = fusion_layer(fea_X)
        # 提取的融合特征
        inputs["combined_features"] = combined_features
        # 只需要提取特征
        if fea == True:
            return inputs

        if training_object == "discriminator":
            discrimination = self.discriminator(inputs)
            inputs["discrimination"] = discrimination.to(self.device)
            l_D = F.binary_cross_entropy_with_logits(
                inputs["discrimination"], missing_mask
            )
            losses["l_disc"] = l_D
        else:
            reconstruction = self.decoder(inputs)
            inputs["reconstruction"] = reconstruction
            if recon == True:
                return inputs
            results, c, kl = self.clusterings.fit(inputs["combined_features"].cpu().detach().numpy())
            losses["test_initc"] = c

            ri = round(cal_rand_index(results, true_label.cpu()), 4)  # 取值范围为[0,1]
            acc = round(calculate_ACC(true_label.cpu(), results), 4)
            pur = round(cal_cluster_purity(results, true_label.cpu()), 4)  # 取值范围为[0,1]
            nmi = round(normalized_mutual_info_score(true_label.cpu(), results, average_method='arithmetic'),
                        4)  # 取值范围为[0,1]
            losses["ri"] = ri
            losses["nmi"] = nmi
            losses["acc"] = acc
            losses["pur"] = pur



            inputs["discrimination"] = inputs["discrimination"].detach()

            l_G = F.binary_cross_entropy_with_logits(
                inputs["discrimination"], 1 - missing_mask, weight=1 - missing_mask
            )  # 填充缺失数据,生成数据的loss,或者缺失地方的loss
            l_pre = cal_mse(inputs["imputation"], X, missing_mask)
            l_rec = cal_mse(inputs["reconstruction"], X, missing_mask)
            l_stat = nn.MSELoss()(inputs["StatAutoencoder_decoded"], X)
            l_freq = nn.MSELoss()(inputs["FreqAutoencoder_decoded"], X)
            l_cluster = kl
            # l_cluster = nn.KLDivLoss(size_average=False)(inputs["fuzzy_membership"].log(), inputs["traget"])
            # l_classifier = nn.CrossEntropyLoss()(torch.as_tensor(class_predictions,dtype=torch.float).to(self.device),torch.as_tensor(true_label,dtype=torch.float).to(self.device))

            # HTH = torch.matmul(inputs["fcn_latent"], inputs["fcn_latent"].permute(1, 0))
            # term_F = torch.nn.init.orthogonal_(
            #     torch.randn(batch_size, self.n_clusters, device=self.device), gain=1
            # )
            # FTHTHF = torch.matmul(torch.matmul(term_F.permute(1, 0), HTH), term_F)
            # l_kmeans = torch.trace(HTH) - torch.trace(FTHTHF)  # k-means loss

            loss_gene = l_pre + l_rec + l_stat + l_freq + l_G + 100*l_cluster
            losses["l_gene"] = loss_gene

        return losses


class CRLI:
    def __init__(
            self,
            n_steps,
            n_features,
            n_clusters,
            n_generator_layers,
            rnn_hidden_size,
            features_dims,
            class_nums,
            epochs,
            batch_size,
            rnn_cell_type,
            learning_rate,
            device,
            d_hidden,
            m,
            beta,
            decoder_fcn_output_dims=[64,6],#None,#[10,1],
            lambda_kmeans=1e-3,
            G_steps=1,
            D_steps=1,
            patience=15,
            weight_decay=1e-5
    ):
        assert G_steps > 0 and D_steps > 0, "G_steps and D_steps should both >0"
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.original_patience = patience
        self.lr = learning_rate
        self.weight_decay = weight_decay
        # self.clusterer = RccCluster(measure='cosine')
        self.n_steps = n_steps
        self.n_features = n_features
        self.G_steps = G_steps
        self.D_steps = D_steps
        self.device = device
        self.class_nums = class_nums
        self.update_interval = 10
        self.tol = 0.001
        self.model = _CRLI(
            n_steps,
            n_features,
            n_clusters,
            n_generator_layers,
            rnn_hidden_size,
            features_dims,
            class_nums,
            d_hidden,#decoder_fcn_output_dims,
            lambda_kmeans,
            rnn_cell_type,
            device,
            m,
            beta,

        )
        self.model = self.model.to(self.device)
        self._print_model_size()
        self.logger = {"training_loss_generator": [], "training_loss_discriminator": []}

    def _print_model_size(self):
        """Print the number of trainable parameters in the initialized NN model."""
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(
            f"Model initialized successfully. Number of the trainable parameters: {num_params}"
        )

    def check_input(
            self, expected_n_steps, expected_n_features, x, y=None, out_dtype="tensor"
    ):
        """Check value type and shape of input X and y

        Parameters
        ----------
        expected_n_steps : int
            Number of time steps of input time series (X) that the model expects.
            This value is the same with the argument `n_steps` used to initialize the model.

        expected_n_features : int
            Number of feature dimensions of input time series (X) that the model expects.
            This value is the same with the argument `n_features` used to initialize the model.

        x : array-like,
            Time-series data that must have a shape like [n_samples, expected_n_steps, expected_n_features].

        y : array-like, default=None
            Labels of time-series samples (X) that must have a shape like [n_samples] or [n_samples, n_classes].

        out_dtype : str, in ['tensor', 'ndarray'], default='tensor'
            Data type of the output, should be np.ndarray or torch.Tensor

        Returns
        -------
        X : tensor

        y : tensor
        """
        assert out_dtype in [
            "tensor",
            "ndarray",
        ], f'out_dtype should be "tensor" or "ndarray", but got {out_dtype}'
        is_list = isinstance(x, list)
        is_array = isinstance(x, np.ndarray)
        is_tensor = isinstance(x, torch.Tensor)
        assert is_tensor or is_array or is_list, TypeError(
            "X should be an instance of list/np.ndarray/torch.Tensor, "
            f"but got {type(x)}"
        )

        # convert the data type if in need
        if out_dtype == "tensor":
            if is_list:
                x = torch.tensor(x).to(self.device)
            elif is_array:
                x = torch.from_numpy(x).to(self.device)
            else:  # is tensor
                x = x.to(self.device)
        else:  # out_dtype is ndarray
            # convert to np.ndarray first for shape check
            if is_list:
                x = np.asarray(x)
            elif is_tensor:
                x = x.numpy()
            else:  # is ndarray
                pass

        # check the shape of X here
        x_shape = x.shape
        assert len(x_shape) == 3, (
            f"input should have 3 dimensions [n_samples, seq_len, n_features],"
            f"but got shape={x.shape}"
        )
        assert (
                x_shape[1] == expected_n_steps
        ), f"expect X.shape[1] to be {expected_n_steps}, but got {x_shape[1]}"
        assert (
                x_shape[2] == expected_n_features
        ), f"expect X.shape[2] to be {expected_n_features}, but got {x_shape[2]}"

        if y is not None:
            assert len(x) == len(y), (
                f"lengths of X and y must match, " f"but got f{len(x)} and {len(y)}"
            )
            if isinstance(y, torch.Tensor):
                y = y.to(self.device) if out_dtype == "tensor" else y.numpy()
            elif isinstance(y, list):
                y = (
                    torch.tensor(y).to(self.device)
                    if out_dtype == "tensor"
                    else np.asarray(y)
                )
            elif isinstance(y, np.ndarray):
                y = torch.from_numpy(y).to(self.device) if out_dtype == "tensor" else y
            else:
                raise TypeError(
                    "y should be an instance of list/np.ndarray/torch.Tensor, "
                    f"but got {type(y)}"
                )
            return x, y
        else:
            return x

    def fit(self, train):
        train_x,train_y = self.check_input(self.n_steps, self.n_features, x=train["x"], y=train["y"])
        training_set = TimeSeriesDataset(train)
        training_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        print("dataloader============================>:", training_loader)
        #自编码器优化器
        self.G_optimizer = torch.optim.Adam(
            [
                {"params": self.model.generator.parameters()},
                {"params": self.model.statautoencoder.parameters()},
                {"params": self.model.freqautoencoder.parameters()},
                {"params": self.model.decoder.parameters()},
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # 鉴别器优化器
        self.D_optimizer = torch.optim.Adam(
            self.model.discriminator.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.best_loss = float("inf")
        self.best_model_dict = None

        #用全部的训练集寻找聚类算法最优参数
        features = []
        labels = []
        for idx, data in enumerate(training_loader):
            inputs = self.assemble_input_data(data)
            inputs = self.model.forward(inputs, training_object="discriminator",fea=True)
            features.append(inputs["combined_features"].detach().cpu())
            labels.append(inputs["y"].detach().cpu())
        concat_features = torch.cat(features)
        concat_labels = torch.cat(labels)
        self.model.parameter(concat_features, concat_labels)
        RI = []
        NMI = []
        ACC = []
        PUR = []
        LOSS = []

        try:
            for epoch in range(self.epochs):

                self.model.train()
                epoch_train_loss_G_collector = []
                epoch_train_loss_D_collector = []
                RI_collector = []
                NMI_collector = []
                ACC_collector = []
                PUR_collector = []
                LOSS_collector = []
                # 随机初始化聚类中心
                init_indices = sample(range(len(training_set)), k=self.class_nums)
                init_samples = torch.stack([concat_features[i] for i in init_indices])
                C = np.array(init_samples.view(self.class_nums, -1))
                self.model.clusterings.init_cluster_c(C)
                for idx, data in enumerate(training_loader):
                    if len(data[0]) < self.class_nums:
                        num_repeats = self.class_nums - len(data[0])
                        repeated_indices = torch.randint(0, len(training_set), (num_repeats,))
                        repeated_samples = [training_set[i][0] for i in repeated_indices]
                        repeated_labels = [training_set[i][2] for i in repeated_indices]
                        repeated_mask = [training_set[i][1] for i in repeated_indices]
                        data[0] = torch.cat((data[0], torch.stack(repeated_samples, dim=0)), dim=0)
                        data[2] = torch.cat((data[2], torch.stack(repeated_labels,dim=0)), dim=0)
                        data[1] = torch.cat((data[1], torch.stack(repeated_mask,dim=0)), dim=0)
                    inputs = self.assemble_input_data(data)
                    for _ in range(self.D_steps):
                        self.D_optimizer.zero_grad()
                        results = self.model.forward(
                            inputs, training_object="discriminator"
                        )
                        results["l_disc"].backward(retain_graph=True)
                        self.D_optimizer.step()
                        epoch_train_loss_D_collector.append(results["l_disc"].item())


                    for _ in range(self.G_steps):
                        self.G_optimizer.zero_grad()
                        # 首先随机初始化聚类中心

                        results = self.model.forward(
                            inputs, training_object="generator"
                        )
                        self.test_initc = results["test_initc"]
                        results["l_gene"].backward()
                        self.G_optimizer.step()
                        epoch_train_loss_G_collector.append(results["l_gene"].item())
                        RI_collector.append(results["ri"])
                        NMI_collector.append(results["nmi"])
                        ACC_collector.append(results["acc"])
                        PUR_collector.append(results["pur"])
                        LOSS_collector.append(results["l_gene"].item())

                RI.append(np.mean(RI_collector))
                NMI.append(np.mean(NMI_collector))
                ACC.append(np.mean(ACC_collector))
                PUR.append(np.mean(PUR_collector))
                LOSS.append(np.mean(LOSS_collector))

                mean_train_G_loss = np.mean(
                    epoch_train_loss_G_collector
                )  # mean training loss of the current epoch
                mean_train_D_loss = np.mean(
                    epoch_train_loss_D_collector
                )  # mean training loss of the current epoch

                self.logger["training_loss_generator"].append(mean_train_G_loss)
                self.logger["training_loss_discriminator"].append(mean_train_D_loss)
                print(
                    f"epoch {epoch}: "
                    f"training loss_generator {mean_train_G_loss:.4f}, "
                    f"train loss_discriminator {mean_train_D_loss:.4f}"
                )
                mean_loss = mean_train_G_loss
                # early_stopping = EarlyStopping('./save_model')
                # early_stopping(mean_loss, self.model)
                # if early_stopping.early_stop:
                #     print("Exceeded the training patience. Terminating the training procedure...")
                #     break  # 跳出迭代，结束训练
                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    # self.best_model_dict = self.model.state_dict()
                    self.best_model_dict = torch.save(self.model, './save_model/best_network.pt')
                    self.patience = self.original_patience
                else:
                    self.patience -= 1
                    print(self.patience)
                    if self.patience == 0:
                        print(
                            "Exceeded the training patience. Terminating the training procedure..."
                        )
                        break
        except Exception as e:
            print(f"Exception: {e}")
            if self.best_model_dict is None:
                raise RuntimeError(
                    "Training got interrupted. Model was not get trained. Please try fit() again."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. "
                    "Model will load the best parameters so far for testing. "
                    "If you don't want it, please try fit() again."
                )

        if np.equal(self.best_loss, float("inf")):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        visualization(LOSS, RI, NMI, ACC, PUR)

        print("Finished training.")
        # self._train_model(training_loader)  #保存模型权重
        # self.model = torch.load("./save_model/best_network.pt", map_location=self.device)
        # self.model.load_state_dict(self.best_model_dict)
        #self.model.eval()  # set the model as eval status to freeze it.


    def assemble_input_data(self, data):
        """Assemble the input data into a dictionary.

        Parameters
        ----------
        data : list
            A list containing data fetched from Dataset by Dataload.

        Returns
        -------
        inputs : dict
            A dictionary with data assembled.
        """
        # fetch data
        x, missing_mask, y = data
        inputs = {
            "X": x,
            "missing_mask": missing_mask,
            "y": y,
        }
        return inputs

    def clustering(self, test):
        test_x, test_y = self.check_input(self.n_steps, self.n_features, test["x"], test["y"])  # 检查x是不是三维的
        self.model = torch.load("./save_model/best_network.pt",map_location=self.device)   #载入模型权重
        # self.model = self.model.load_state_dict(torch.load("./save_model/best_network.pt"))
        self.model.eval()  # set the model as eval status to freeze it.
        test_set = TimeSeriesDataset(test)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        raw_data = []
        true_label = []
        latent_collector = []
        reconstruction_data = []
        imputation_data = []


        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self.assemble_input_data(data)
                outputs = self.model.forward(inputs,recon=True)

                raw_data.append(outputs["X"])
                true_label.append(outputs["y"])
                latent_collector.append(outputs["combined_features"])
                reconstruction_data.append(outputs['reconstruction'])
                imputation_data.append(outputs["imputed_X"])

        latent_collector = torch.cat(latent_collector).cpu().detach().numpy()
        raw_data = torch.cat(raw_data).cpu().detach().numpy()
        reconstruction_data = torch.cat(reconstruction_data).cpu().detach().numpy()
        true_label = torch.cat(true_label).cpu().detach().numpy()
        imputation_data = torch.cat(imputation_data).cpu().detach().numpy()

        raw_data = raw_data.reshape(raw_data.shape[0],-1)
        reconstruction_data = reconstruction_data.reshape(reconstruction_data.shape[0],-1)
        imputation_data = imputation_data.reshape(imputation_data.shape[0],-1)
        print("原始x的shape为:", raw_data.shape)
        print("重构后x的shape为:", reconstruction_data.shape)

        mse = round(mean_squared_error(raw_data,imputation_data),4)
        rmse = round(sqrt(mean_squared_error(raw_data,imputation_data)),4)
        mae = round(mean_absolute_error(raw_data,imputation_data),4)

        # 用训练过程中得到的聚类中心进行聚类初始化
        self.model.clusterings.init_cluster_c(self.test_initc)
        # 在测试数据上执行聚类
        results,c,loss_clustering = self.model.clusterings.fit(latent_collector)

        #results = self.model.kmeans.fit_predict(latent_collector)
        np.savetxt(r'Meat_0.7truelabels.txt', np.array(true_label))
        np.savetxt(r'Meat_0.7predlabels.txt', np.array(results))
        np.savetxt(r'features.txt', np.array(latent_collector))

        metrics_imputation = pd.DataFrame({"mse":mse,"rmse":rmse,"mae":mae},index=[0])
        #metrics_imputation.to_excel(r'C:/Users/lyr/Desktop/202331/CRLI/metrics_imputation.xls')
        print("mse:", mse, "rmse:", rmse, "mae:", mae)

        # draw(raw_data, results, "raw_data.png")
        # draw(imputation_data, results, "imputation_data.png")
        # draw(reconstruction_data, results, "reconstruction_data.png")
        return results,true_label


# def visual(x, centroids, labels):
#     # 提取的特征
#     n_samples, n_features = x.shape
#     print("提取的特征形状:", x.shape)
#     print("质心形状:", centroids.shape)
#     print("标签形状:", labels.shape)
#     print("标签长度:", len(labels))
#     #print("缺失值矩阵形状:", is_complete.shape)
#     # 聚类中心
#     n_clusters, n_features = centroids.shape
#
#     colors = cm.get_cmap('viridis', n_clusters)
#     #is_complete_bool = is_complete.bool()
#
#     #is_complete = np.random.choice([True, False], size=n_samples, p=[0.3, 0.7])  # 70% 数据完整，30% 数据不完整
#     # 所有数据
#     all_data = np.vstack((x, centroids))
#     is_centroid = np.zeros(all_data.shape[0], dtype=bool)
#     is_centroid[-centroids.shape[0]:] = True
#
#     # 使用t-SNE进行降维
#     tsne = TSNE(n_components=2, random_state=0)
#     all_data_2d = tsne.fit_transform(all_data)
#
#     # 分离出原始数据点和质心的低维表示
#     X_2d = all_data_2d[:-centroids.shape[0], :]
#     centroids_2d = all_data_2d[-centroids.shape[0]:, :]
#     labels = labels.squeeze()
#
#     # 绘制完整数据
#     # for i in range(n_clusters):
#     #     cluster_complete = X_2d[is_complete.cpu() & (labels == (i+1)),:]
#     #     plt.scatter(cluster_complete[:,0], cluster_complete[:,1], c=colors(i),
#     #                 marker='.')
#     #     print("完整数据的形状:",cluster_complete.shape)
#     #
#     # print("======================")
#     # # 绘制不完整数据叉号，使用不同的颜色表示不同的簇
#     # for i in range(n_clusters):
#     #     cluster_incomplete = X_2d[~is_complete.cpu() & (labels == (i+1)),:]
#     #     plt.scatter(cluster_incomplete[:,0], cluster_incomplete[:,1], c=colors(i),
#     #                  marker='x', s=10)
#     #     print("不完整数据的形状:",cluster_incomplete.shape)
#
#     for i in range(n_clusters):
#         cluster_complete = X_2d[(labels == (i+1)),:]
#         plt.scatter(cluster_complete[:,0], cluster_complete[:,1], c=colors(i),
#                     marker='.')
#
#
#     # 绘制质心星号
#     plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', marker='*')
#
#     # 不显示刻度线
#     plt.xticks([])  # X轴刻度线为空
#     plt.yticks([])  # Y轴刻度线为空
#
#     # 保存图像为PDF
#     #plt.savefig('RacketSports_0.9.pdf', format='pdf')
#
#     # 显示图像
#     plt.show()

def visualization(losses, ri_scores, nmi_scores, acc_scores, pur_scores):
    plt.figure(figsize=(10, 6))

    x_epochs = np.arange(len(losses))

    # 创建第一个y轴（左侧），用于聚类性能
    ax1 = plt.gca()
    line_ri, = ax1.plot(x_epochs, ri_scores, color='green', linestyle='-', marker='o')
    line_nmi, = ax1.plot(x_epochs, nmi_scores, color='red', linestyle='-', marker='s')
    line_acc, = ax1.plot(x_epochs, acc_scores, color='purple', linestyle='-', marker='^')
    line_pur, = ax1.plot(x_epochs, pur_scores, color='yellow', linestyle='-', marker='*')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Clustering Performance')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0.0, 1.0)
    ax1.yaxis.set_ticks(np.arange(0.0, 1.1, 0.2))

    # 创建第二个y轴（右侧），用于损失值
    ax2 = ax1.twinx()
    line_loss, = ax2.plot(x_epochs, losses, label='Loss', color='blue', linestyle='-', marker='v')
    ax2.set_ylabel('Loss Value')
    ax2.tick_params(axis='y')
    ax2.set_ylim(min(losses) * 0.9, max(losses) * 1.1)

    # 合并图例
    lines = [line_ri, line_nmi, line_acc, line_pur, line_loss]
    labels = ['RI', 'NMI', 'ACC', 'PUR', 'Loss']
    ax1.legend(lines, labels, loc='upper right', ncol=1, bbox_to_anchor=(1, 1), frameon=True)

    #plt.xticks(x_epochs)
    plt.xticks(x_epochs[::5])  # 这里的 [::2] 表示每隔一个元素取一个
    plt.grid(True)
    plt.show()


# def draw(datas, clustering, name):
#     colours = {0: 'r', 1: "b", 2: 'y', 3: 'c', 4: 'k', 5: 'g', 6: "m"}
#     plt.figure()
#     line = {}
#     for i, data in enumerate(datas):
#         if clustering[i] not in line:
#             line[clustering[i]] = plt.plot(range(len(data)), data, colours[clustering[i]])[0]
#         else:
#             plt.plot(range(len(data)), data, colours[clustering[i]])
#     plt.legend([value for item, value in line.items()], [f"class_{item}" for item, value in line.items()],
#                loc='upper left')
#     plt.savefig(name)
