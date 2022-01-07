import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import os
import pickle as pkl

def eval_regression_model(ts_y,ts_pred,productid='if',model_name='Simple LSTM',sigmul=3,mstd=None,store_dir=None,dt=None):
    true_seq = ts_y.reshape(len(ts_y))
    pred_seq = ts_pred.reshape(len(ts_pred))
    # 将标准化的结果还原为原始值
    if mstd is not None:
        meanstd=mstd.todaily.tolist()
        ts_y_th,ts_pred_th=[],[]
        mean= meanstd[0]
        std = meanstd[1]
        for idx,ts in enumerate(true_seq):
            ts_conv = std*ts+mean
            prd_conv = std*pred_seq[idx]+mean
            ts_y_th.append(ts_conv)
            ts_pred_th.append(prd_conv)
        true_seq,pred_seq=np.array(ts_y_th),np.array(ts_pred_th)
    df = pd.DataFrame({'true':true_seq,'predict':pred_seq})
    sigma = df.true.std()
    df['upper_margin'] = df.predict.apply(lambda x: x + sigmul * sigma)
    df['lower_margin'] = df.predict.apply(lambda x: x - sigmul * sigma)

    print(model_name)
    mse = round(mean_squared_error(true_seq,pred_seq),4)
    print('MSE: ', mse)

    print('=' * 50)

    # df.plot(y=['predict','true'],style='-o',figsize=(20,10))
    plt.figure(figsize=(20, 10), dpi=100, facecolor="white")
    plt.plot(range(len(true_seq)), true_seq, color='seagreen', linewidth=2, linestyle="-.", marker='o', markersize=6)
    plt.plot(range(len(pred_seq)), pred_seq, color='navy', linewidth=2, linestyle="-", marker='s', markersize=4)
    # plt.fill_between(x=df.index, y1=df.upper_margin, y2=df.lower_margin, color='silver')
    plt.title('predict by *{}* MSE={}'.format(model_name,mse))
    plt.legend(['ground truth', model_name])

    if not dt:
        dt = datetime.now().strftime('%m%d_%H%M')

    if store_dir is None:
        plt.savefig('my_results/{}_{}_{}_mse.png'.format(dt,productid,model_name))
        print('Regression mse plot saved to my_results/{}_{}_{}_mse.png! '.format(dt,productid,model_name))
    else:
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        plt.savefig('{}/{}_{}_{}_mse.png'.format(store_dir, dt, productid,model_name))
        print('Regression mse plot saved to {}/{}_{}_{}_mse.png!'.format(store_dir,dt,productid,model_name))

    plt.show()
    plt.close()
    return mse

def error_distribute_regression(ts_y,ts_pred,model_name='Simple LSTM',store_dir=None,dt=None):
    error_list = ts_y-ts_pred
    ax = sns.distplot(error_list,label='error')
    ax.set_title("*{}* Error distirbution!".format(model_name))

    if not dt:
        dt = datetime.now().strftime('%m%d_%H%M')

    if store_dir is None:
        plt.savefig('my_results/{}_{}_edist.png'.format(dt,model_name))
        print('Error distribute plot save {} to my_results/'.format(model_name))
    else:
        if not os.path.exists(store_dir):
            os.makedirs(store_dir)
        plt.savefig('{}/{}_{}_edist.png'.format(store_dir, dt, model_name))
        print('Error distribute plot save {}_{} to {}/'.format(dt, model_name,store_dir))
    plt.show()
    plt.close()

