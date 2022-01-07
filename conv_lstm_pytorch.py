import os
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

import pandas as pd
import data_preclean as dp
import eval_res
from datetime import datetime
from lstm_pytorch import lstm_data_dt,lstm_data

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#TODO： 调参，尝试总结出合适的超参数
from torch.utils.data import Dataset

class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.len

class ConvLSTMCell(nn.Module,):
    def __init__(self, input_dim, hidden_dim, kernel_size,bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM_Net(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM_Net, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, seq_length, nf, dense_size,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
        self.nlayer = num_layers
        self.convlstm = ConvLSTM_Net(input_dim, hidden_dim, kernel_size, num_layers, batch_first,bias,return_all_layers)
        self.batchnorm = nn.BatchNorm2d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim*seq_length*nf,dense_size)
        self.fc2 = nn.Linear(dense_size,1)

    def forward(self, x, hidden_state=None):
        layer_outputs, last_states = self.convlstm(x,hidden_state)
        hs_last = last_states[-1][0]
        output = self.batchnorm(hs_last)
        output = self.fc1(output.flatten(start_dim=1))
        output = self.fc2(output)
        return output

class Atten_ConvLSTM(nn.Module):
    def __init__(self,input_dim, hidden_dim, kernel_size, num_layers, seq_length, nf, dense_size,
                 batch_first=False, bias=True, return_all_layers=False):
        super(Atten_ConvLSTM, self).__init__()
        self.nlayer = num_layers
        self.convlstm = ConvLSTM_Net(input_dim, hidden_dim, kernel_size, num_layers, batch_first,bias,return_all_layers)
        self.atten_layer = SEAttention(hidden_dim,reduction=4)
        self.batchnorm = nn.BatchNorm2d(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim*seq_length*nf,dense_size)
        self.fc2 = nn.Linear(dense_size,1)

    def forward(self, x, hidden_state=None):
        layer_outputs, last_states = self.convlstm(x,hidden_state)
        hs_last = last_states[-1][0]
        hs_last = self.atten_layer(hs_last)
        output = self.batchnorm(hs_last)
        output = self.fc1(output.flatten(start_dim=1))
        output = self.fc2(output)
        return output

def conv_lstm(tr_x, ts_x, tr_y, ts_y, batch_num=32, epoch_num=100, k_size=(3,1), filters=16, dense_size=32,
             input_channel=1, long_length=1, num_layer=1, proid='if',model_name='ConvLSTM', store_dir=None,use_cuda=True):

    seq_length = tr_x.shape[1]
    nf = tr_x.shape[2]
    dt = datetime.now().strftime('%m%d_%H%M')

    # 输入数据构建（格式为：(batch_size, seq_length, input_channel, hight, width)
    tr_x = tr_x.reshape((tr_x.shape[0], long_length, input_channel, seq_length, nf))
    ts_x = ts_x.reshape((ts_x.shape[0], long_length, input_channel, seq_length, nf))
    dataset = timeseries(tr_x,tr_y)
    val_dataset = timeseries(ts_x,ts_y)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_num)
    val_loader = DataLoader(val_dataset,shuffle=True, batch_size=batch_num)

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU
    convlstm_model = ConvLSTM(input_channel, filters, k_size, num_layer, seq_length, nf, dense_size, batch_first=True).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(convlstm_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=3,cooldown=10) # 学习率自适应操作
    valid_loss_min = float("inf")
    global_step=0

    from torchinfo import summary
    print(summary(convlstm_model, input_size=(batch_num,long_length,input_channel, seq_length,nf)))

    for i in range(epoch_num):
        convlstm_model.train()
        train_loss_array = []
        hidden_train = None
        optimizer.zero_grad()  # 每次训练前将梯度重置为0
        for j,data in enumerate(train_loader):
            train_X, train_Y = data[0].to(device),data[1].to(device)
            y_pred = convlstm_model(train_X,hidden_train)
            loss = criterion(y_pred, train_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step+=1

        convlstm_model.eval()
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in val_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y = convlstm_model(_valid_X, hidden_valid)
            loss = criterion(pred_Y, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        print("For {}th epoch, the train loss is {:.6f}. ".format(i,train_loss_cur) +
              "For {}th epoch, the valid loss is {:.6f}.".format(i,valid_loss_cur))
        scheduler.step(valid_loss_cur)

        if store_dir is None:
            filepath = 'test_checkpoints/{}_{}_{}_model.pth'.format(dt,proid,model_name)
        else:
            if not os.path.exists(store_dir):
                os.makedirs(store_dir)
            filepath = '{}/{}_{}_{}_model.pth'.format(store_dir,dt,proid,model_name)

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            torch.save(convlstm_model.state_dict(), filepath)  # 模型保存
            print('With lower loss, store new simple_model!')
            # best_model = lstm_model(feature_size,nb_unites,nlayer, dense_size).to(device)
            # best_model.load_state_dict(torch.load(filepath))
            best_model = convlstm_model

    return best_model

def conv_lstm_attention(tr_x, ts_x, tr_y, ts_y, batch_num=32, epoch_num=100, k_size=(3,1), filters=16, dense_size=32,
             input_channel=1, long_length=1, num_layer=1, proid='if',model_name='ConvLSTM', store_dir=None,use_cuda=True):

    seq_length = tr_x.shape[1]
    nf = tr_x.shape[2]
    dt = datetime.now().strftime('%m%d_%H%M')

    # 输入数据构建（格式为：(batch_size, seq_length, input_channel, hight, width)
    tr_x = tr_x.reshape((tr_x.shape[0], long_length, input_channel, seq_length, nf))
    ts_x = ts_x.reshape((ts_x.shape[0], long_length, input_channel, seq_length, nf))
    dataset = timeseries(tr_x,tr_y)
    val_dataset = timeseries(ts_x,ts_y)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_num)
    val_loader = DataLoader(val_dataset,shuffle=True, batch_size=batch_num)

    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU
    att_convlstm_model = Atten_ConvLSTM(input_channel, filters, k_size, num_layer, seq_length, nf, dense_size, batch_first=True).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(att_convlstm_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=3,cooldown=10) # 学习率自适应操作
    valid_loss_min = float("inf")
    # vis = visdom.Visdom(env='model_pytorch')
    global_step=0

    from torchinfo import summary
    print(summary(att_convlstm_model, input_size=(batch_num,long_length,input_channel, seq_length,nf)))


    for i in range(epoch_num):
        att_convlstm_model.train()
        train_loss_array = []
        hidden_train = None
        optimizer.zero_grad()  # 每次训练前将梯度重置为0
        for j,data in enumerate(train_loader):
            train_X, train_Y = data[0].to(device),data[1].to(device)
            y_pred = att_convlstm_model(train_X,hidden_train)
            loss = criterion(y_pred, train_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step+=1

        att_convlstm_model.eval()
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in val_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y = att_convlstm_model(_valid_X, hidden_valid)
            loss = criterion(pred_Y, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        print("For {}th epoch, the train loss is {:.6f}. ".format(i,train_loss_cur) +
              "For {}th epoch, the valid loss is {:.6f}.".format(i,valid_loss_cur))
        scheduler.step(valid_loss_cur)

        if store_dir is None:
            filepath = 'test_checkpoints/{}_{}_{}_model.pth'.format(dt,proid,model_name)
        else:
            if not os.path.exists(store_dir):
                os.makedirs(store_dir)
            filepath = '{}/{}_{}_{}_model.pth'.format(store_dir,dt,proid,model_name)

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            torch.save(att_convlstm_model.state_dict(), filepath)  # 模型保存
            print('With lower loss, store new simple_model!')
            # best_model = lstm_model(feature_size,nb_unites,nlayer, dense_size).to(device)
            # best_model.load_state_dict(torch.load(filepath))
            best_model = att_convlstm_model

    return best_model

def average_output(ts_y,ts_pred):
    if ts_y.shape[1]<=1:
        print('No need for average!')
        return ts_y,ts_pred
    else:
        ydf,predf = pd.DataFrame(ts_y),pd.DataFrame(ts_pred)
        roll_num = ydf.shape[1] # 第i个col就向下平移i位
        column_list=[0]
        for i in range(1,roll_num):
            col = '{}_L1'.format(i)
            ydf[col] = ydf[i].shift(i)
            predf[col]=predf[i].shift(i)
            column_list.append(col)
        tmp_y = ydf[column_list].mean(axis=1)
        tmp_pred = predf[column_list].mean(axis=1)
        return tmp_y.to_numpy(),tmp_pred.to_numpy()

def convlstm_main(input_folder, output_folder, pid_list, yq_list, stime='20150907',etime='20210720'):
    for product_id in pid_list:
        # 获取输入数据名称及地址
        hqfile1719 = os.path.join(input_folder, '{}_to_features1521_avg.pkl'.format(product_id))
        product_index = '{}_index'.format(product_id)

        # 生成与产品对应的保存数据路径
        pid_folder = os.path.join(output_folder,product_id)
        if not os.path.exists(pid_folder):
            os.makedirs(pid_folder)
        for yq in yq_list:
            if yq == 'WithYQ':
                norm_columns = ['todaily', 'settle_basis', 'inner_volatility', 'tradingcost', 'xh_mov_vol30',
                                'xh_mov_vol05',
                                'xh_liq', product_index]

                column_list = ['todaily_L1', 'settle_basis_L1', 'inner_volatility_L1', 'delivery_pivtol',
                               'tradingcost_L1', 'inopen', 'xh_mov_vol30_L1', 'xh_mov_vol05_L1', 'xh_liq_L1', product_index]

            elif yq == 'NoYQ':
                norm_columns = ['todaily', 'settle_basis', 'inner_volatility', 'tradingcost', 'xh_mov_vol30',
                                'xh_mov_vol05', 'xh_liq']
                column_list = ['todaily_L1', 'settle_basis_L1', 'inner_volatility_L1', 'delivery_pivtol',
                               'tradingcost_L1', 'inopen', 'xh_mov_vol30_L1', 'xh_mov_vol05_L1', 'xh_liq_L1']
            else:
                return -1

            out_sub_folder = os.path.join(pid_folder, yq) # 结果输出路径
            if not os.path.exists(out_sub_folder):
                os.makedirs(out_sub_folder)

            df, mstdf = dp.norm_maindf_gen(hqfile1719, column_list=norm_columns, rolling_window=5, timestart=stime,
                                           timeend=etime)
            dt = datetime.now().strftime('%m%d_%H%M')
            dim_in = 3
            dim_out = 1
            is_cnn = True
            td = '20150910'
            long_length = 1
            input_channel = 1
            num_layer = 1
            dense_size = 32

            pred_date = df[df['tradingday'] >= td].tradingday.tolist()
            _, all_ts_x, _, _, all_ts_y, _ = lstm_data_dt(df, column_list, y_target='todaily', dim_in=dim_in,
                                                          dim_out=dim_out, test_date=td)

            reshaped_inputs_ts = all_ts_x.reshape((all_ts_x.shape[0], long_length, input_channel, all_ts_x.shape[1], all_ts_x.shape[2]))

            mn = 'ConvLSTM'

            conv_model = conv_lstm(all_ts_x, all_ts_x, all_ts_y, all_ts_y, k_size=(3, 1), epoch_num=100, filters=16,
                                  dense_size=dense_size,num_layer=num_layer,long_length=long_length,input_channel=input_channel,
                                  proid=product_id,model_name=mn,store_dir=out_sub_folder)

            y_pred = conv_model(torch.Tensor(reshaped_inputs_ts)).reshape(-1)

            eval_res.eval_regression_model(all_ts_y, y_pred, product_id, mn, mstd=mstdf, store_dir=out_sub_folder,dt=dt)

            mn='Attention_ConvLSTM'
            att_convlstm_model = conv_lstm_attention(all_ts_x, all_ts_x, all_ts_y, all_ts_y, k_size=(3,1), epoch_num=100, filters=16,
                                       dense_size=dense_size, num_layer=num_layer, long_length=long_length,
                                       input_channel=input_channel,proid=product_id, model_name=mn, store_dir=out_sub_folder)
            y_pred = att_convlstm_model(torch.Tensor(reshaped_inputs_ts)).reshape(-1)

            eval_res.eval_regression_model(all_ts_y,y_pred,product_id,mn,mstd=mstdf,store_dir=out_sub_folder,dt=dt)


if __name__=='__main__':
    input_folder = 'data_files'
    output_folder = 'for_yq_system'
    pid_list =['if']
    yq_list = ['WithYQ','NoYQ']
    stime = '20150907'
    etime = '20210915'
    convlstm_main(input_folder,output_folder,pid_list,yq_list,stime,etime)


