import os
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import data_preclean as dp
import eval_res
from datetime import datetime
from math import ceil
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def plotting(history):
    plt.plot(history.history['loss'], color = "red")
    plt.plot(history.history['val_loss'], color = "blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()

# 返回：训练集，验证集和测试集
def lstm_data_dt(df,columns,y_target='todaily',dim_in=3,dim_out=1,test_date='20150907'):
    inputs, outputs = [], []
    for i in range(len(df) - dim_in - dim_out+1):
        # inputs.append(df[columns].iloc[i:i + dim_in].to_numpy())
        inputs.append(array(df[columns].iloc[i:i + dim_in]))
        outputs.append(array(df[y_target].iloc[i + dim_in:i + dim_in + dim_out]))
    inputs = array(inputs)
    outputs = array(outputs)
    tmpdf = df.reset_index()
    min_date = max(tmpdf.tradingday.iloc[0],test_date)
    ts_pv = int(tmpdf[tmpdf['index'] == datetime.strptime(min_date, '%Y%m%d')].index.to_numpy())
    inputs_tr = inputs[:ts_pv]
    inputs_ts = inputs[ts_pv:]
    outputs_tr = outputs[:ts_pv]
    outputs_ts = outputs[ts_pv:]
    return (inputs_tr, inputs_ts, inputs_ts, outputs_tr, outputs_ts, outputs_ts)

# 返回：训练集，验证集和测试集
def lstm_data(df,columns,y_target='todaily',dim_in=3,dim_out=1,test_size=0.2,shuffle=False):
    inputs, outputs = [], []
    if y_target=='todaily_cate':
        cate_all_list = LabelBinarizer().fit_transform(df[y_target])
        for i in range(len(df) - dim_in - dim_out+1):
            # inputs.append(df[columns].iloc[i:i + dim_in].to_numpy())
            inputs.append(array(df[columns].iloc[i:i + dim_in]))
            outputs.append(cate_all_list[i + dim_in:i + dim_in + dim_out])
        inputs = array(inputs)
        outputs = array(outputs)
    else:
        for i in range(len(df) - dim_in - dim_out+1):
            # inputs.append(df[columns].iloc[i:i + dim_in].to_numpy())
            inputs.append(array(df[columns].iloc[i:i + dim_in]))
            outputs.append(array(df[y_target].iloc[i + dim_in:i + dim_in + dim_out]))
        inputs = array(inputs)
        outputs = array(outputs)
    if shuffle == True:
        X_train, X_val, y_train, y_val = train_test_split(inputs, outputs, test_size =test_size, random_state = 1)
        ts_pv = ceil(len(inputs) * (1 - test_size))
        X_test = inputs[ts_pv:]
        y_test = outputs[ts_pv:]
        return (X_train, X_val, X_val, y_train, y_val, y_val)
        # return (X_train, X_test, X_val, y_train, y_test, y_val)
    else:
        ts_pv = ceil(len(inputs)*(1-test_size))
        inputs_tr = inputs[:ts_pv]
        inputs_ts = inputs[ts_pv:]
        outputs_tr = outputs[:ts_pv]
        outputs_ts = outputs[ts_pv:]
        return (inputs_tr, inputs_ts, inputs_ts, outputs_tr, outputs_ts, outputs_ts)


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

class lstm_model(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dense_num):
        super(lstm_model,self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc1 = nn.Linear(in_features=hidden_size,out_features=dense_num)
        self.fc2 = nn.Linear(in_features=dense_num,out_features=1)

    def forward(self,x,hidden=None):
        output,hidden = self.lstm(x,hidden)
        output = output[:,-1,:]
        output = self.fc1(output)
        output = self.fc2(output)
        return output,hidden

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

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

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

class Attention_LSTM(nn.Module):
    def __init__(self,dim_in, input_size, hidden_size, num_layers, dense_num, atten):
        super(Attention_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # self.fc1 = nn.Linear(in_features=dim_in*hidden_size, out_features=dense_num)
        self.fc1 = nn.Linear(in_features=hidden_size, out_features=dense_num)
        self.fc2 = nn.Linear(in_features=dense_num, out_features=1)
        self.atten = atten

    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        # output = hidden[0].permute(1, 0, 2)
        output = self.atten(output, output, output)
        # output = torch.flatten(output,start_dim=1)
        output = output[:,-1,:]
        output = self.fc1(output)
        output = self.fc2(output)
        return output, hidden

def simple_lstm(tr_x,ts_x,tr_y,ts_y,nb_unites=50,batch_num=32,vb=0,epoch_num=100,lstmact='tanh',proid='if',model_name ='Simple_LSTM',store_dir=None,use_cuda=True):
    dim_in, feature_size = tr_x.shape[1],tr_x.shape[2]
    nlayer = 1
    dense_size = 32
    dt = datetime.now().strftime('%m%d_%H%M')

    dataset = timeseries(tr_x,tr_y)
    val_dataset = timeseries(ts_x,ts_y)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_num)
    val_loader = DataLoader(val_dataset,shuffle=True, batch_size=batch_num)
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU

    simple_model = lstm_model(feature_size,nb_unites,nlayer, dense_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(simple_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=3,cooldown=10) # 学习率自适应操作
    valid_loss_min = float("inf")
    # vis = visdom.Visdom(env='model_pytorch')
    global_step=0

    from torchinfo import summary
    print(summary(simple_model, input_size=(batch_num,dim_in, feature_size)))

    # 前馈网络
    for i in range(epoch_num):
        simple_model.train()
        train_loss_array = []
        hidden_train = None
        optimizer.zero_grad()  # 每次训练前将梯度重置为0
        for j,data in enumerate(train_loader):
            train_X, train_Y = data[0].to(device),data[1].to(device)
            y_pred,_ = simple_model(train_X,hidden_train)
            loss = criterion(y_pred, train_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step+=1

        simple_model.eval()
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in val_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, _ = simple_model(_valid_X, hidden_valid)
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
            torch.save(simple_model.state_dict(), filepath)  # 模型保存
            print('With lower loss, store new simple_model!')
            # best_model = lstm_model(feature_size,nb_unites,nlayer, dense_size).to(device)
            # best_model.load_state_dict(torch.load(filepath))
            best_model = simple_model

    return best_model

def attention_lstm(tr_x,ts_x,tr_y,ts_y,nb_unites=50,batch_num=32,vb=0,epoch_num=100,lstmact='tanh',proid='if',model_name ='Simple_LSTM',store_dir=None,use_cuda=True):
    dim_in, feature_size = tr_x.shape[1],tr_x.shape[2]
    nlayer = 1
    dense_size = 32
    dt = datetime.now().strftime('%m%d_%H%M')

    dataset = timeseries(tr_x,tr_y)
    val_dataset = timeseries(ts_x,ts_y)
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_num)
    val_loader = DataLoader(val_dataset,shuffle=True, batch_size=batch_num)
    device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")  # CPU训练还是GPU

    att = ScaledDotProductAttention(d_model=nb_unites, d_k=nb_unites, d_v=nb_unites, h=2)
    att_model = Attention_LSTM(dim_in,feature_size,nb_unites,nlayer, dense_size,att).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(att_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.5,patience=3,cooldown=10) # 学习率自适应操作
    valid_loss_min = float("inf")
    # vis = visdom.Visdom(env='model_pytorch')
    global_step=0

    from torchinfo import summary
    print(summary(att_model, input_size=(batch_num,dim_in, feature_size)))

    # 前馈网络
    for i in range(epoch_num):
        att_model.train()
        train_loss_array = []
        hidden_train = None
        optimizer.zero_grad()  # 每次训练前将梯度重置为0
        for j,data in enumerate(train_loader):
            train_X, train_Y = data[0].to(device),data[1].to(device)
            y_pred, hidden_tr = att_model(train_X,hidden_train)
            loss = criterion(y_pred, train_Y)
            loss.backward()
            optimizer.step()
            train_loss_array.append(loss.item())
            global_step+=1

        att_model.eval()
        valid_loss_array = []
        hidden_valid = None
        for _valid_X, _valid_Y in val_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y, hidden_val = att_model(_valid_X, hidden_valid)
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
            torch.save(att_model.state_dict(), filepath)  # 模型保存
            print('With lower loss, store new attention_model!')
            best_model = att_model

    return best_model

# 只有当dim_out的值大于1时才需要在eval_LSTM前增加这个步骤
# 具体功能：当输出序列长度大于1时，对被重复预测的时间点进行平均
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

def lstm_main(input_folder,output_folder,pid_list,yq_list,start_time,end_time,use_cuda):
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
                                'xh_mov_vol05',
                                'xh_liq']
                column_list = ['todaily_L1', 'settle_basis_L1', 'inner_volatility_L1', 'delivery_pivtol',
                               'tradingcost_L1', 'inopen', 'xh_mov_vol30_L1', 'xh_mov_vol05_L1', 'xh_liq_L1']
            else:
                return -1

            out_sub_folder = os.path.join(pid_folder, yq) # 结果输出路径
            if not os.path.exists(out_sub_folder):
                os.makedirs(out_sub_folder)

            df, mstdf = dp.norm_maindf_gen(hqfile1719, column_list=norm_columns, rolling_window=5, timestart='20150907',
                                           timeend='20210720')

            dt = datetime.now().strftime('%m%d_%H%M')
            dim_in = 4
            dim_out = 1
            is_cnn = False
            td = '20150909'
            pred_date = df[df['tradingday'] >= td].tradingday.tolist()

            _, all_ts_x, _, _, all_ts_y, _ = lstm_data_dt(df, column_list, y_target='todaily', dim_in=dim_in,
                                                          dim_out=dim_out, test_date=td)

            test_x = torch.Tensor(all_ts_x)

            mn='SimpleLSTM'

            simple_model = simple_lstm(all_ts_x, all_ts_x, all_ts_y, all_ts_y,proid=product_id,model_name=mn,store_dir=out_sub_folder,use_cuda=True)

            y_pred = simple_model(test_x)[0].reshape(-1) # predict,hidden_state

            eval_res.eval_regression_model(all_ts_y, y_pred, product_id, mn, mstd=mstdf, store_dir=out_sub_folder,
                                           dt=dt)

            mn='AttenLSTM'
            att_model = attention_lstm(all_ts_x, all_ts_x, all_ts_y, all_ts_y,proid=product_id,model_name=mn,store_dir=out_sub_folder)
            y_pred = att_model(test_x)[0].reshape(-1)

            eval_res.eval_regression_model(all_ts_y,y_pred,product_id,mn,mstd=mstdf,store_dir=out_sub_folder,dt=dt)

if __name__=='__main__':
    use_cuda = True
    input_folder = 'data_files'
    output_folder = 'for_yq_system'
    pid_list = ['if']
    yq_list = ['WithYQ', 'NoYQ']
    start_time ='20150907'
    end_time = '20210720'
    lstm_main(input_folder,output_folder,pid_list,yq_list,start_time,end_time,use_cuda)