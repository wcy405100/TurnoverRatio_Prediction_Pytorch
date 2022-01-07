import pickle as pkl
import pandas as pd
from datetime import datetime

def read_pkl(file):
    with open(file,'rb') as fid:
        ot = pkl.load(fid)
        return ot

def norm_maindf_gen(hq_pkl, column_list=None,rolling_columns=None, rolling_window=5, timestart='20100420', timeend='20191231'):
    maindf = read_pkl(hq_pkl)
    meanstd = pd.DataFrame()
    # 行情数据部分
    maindf['index'] = maindf.tradingday.apply(lambda x: datetime.strptime(x, '%Y%m%d'))
    maindf.set_index('index', inplace=True)
    maindf = maindf.loc[datetime.strptime(timestart, '%Y%m%d'):datetime.strptime(timeend, '%Y%m%d')]

    if rolling_columns:
        maindf = rollmean_fucntion(maindf,columns =rolling_columns,window=rolling_window)

    if column_list:
        for cnorm in column_list:
            meanstd[cnorm], maindf[cnorm] = normalize_single(maindf[cnorm])

    maindf['todaily_L1'] = maindf.todaily.shift(1)
    maindf['todaily_L2'] = maindf.todaily.shift(2)
    maindf['inner_volatility_L1'] = maindf.inner_volatility.shift(1)
    maindf['settle_basis_L1'] = maindf.settle_basis.shift(1)
    maindf['xh_mov_vol30_L1'] = maindf.xh_mov_vol30.shift(1)
    maindf['xh_mov_vol05_L1'] = maindf.xh_mov_vol05.shift(1)
    maindf['xh_liq_L1'] = maindf.xh_liq.shift(1)
    maindf['volume_L1'] = maindf.volume.shift(1)
    maindf['tradingcost_L1'] = maindf.tradingcost.shift(1)
    maindf['position_L1'] = maindf.position.shift(1)
    df = maindf.dropna()
    return df,meanstd

def emodf_gen(yq_pkl, column_list=None,rolling_columns=None, rolling_window=5, timestart='20190101', timeend='20191231'):
    maindf = read_pkl(yq_pkl)
    meanstd = pd.DataFrame()

    if rolling_columns:
        maindf = rollmean_fucntion(maindf,columns = rolling_columns,window=rolling_window)

    if column_list:
        for cnorm in column_list:
            meanstd[cnorm],maindf[cnorm] = normalize_single(maindf[cnorm])
            maindf[cnorm+"_L1"] = maindf[cnorm].shift(1)
    df = maindf.dropna()
    return df,meanstd

def normalize_single(nds):
    mean = nds.mean()
    std = nds.std()
    output = nds.apply(lambda x: (x-mean)/std)
    mstd = pd.Series([mean, std])
    return mstd,output

def rollmean_fucntion(df,columns=None,window=10):
    if not columns:
        columns=df.columns.to_list()
    for feature in columns:
        df[feature] = df[feature].rolling(window=window,center=False).mean()
    df.dropna(inplace=True)
    return df

# 该方法将dataframe中的特定diff_column列做差分操作，并与未做差分的raw_column合并并返回新dataframe
# # dateframe: 原始dataframe
# # raw_column_list: 原始无需修改的column
# # diff_column_list: 需要做差分的column
# ndiff： 差分次数
def merge_df(dataframe,raw_column_list,diff_column_list,ndiff=1):
    newdf=pd.DataFrame()
    for diff_col in diff_column_list:
        diff_df=dataframe[diff_col].diff(ndiff).dropna()
        newdf[diff_col]=diff_df
    for col in raw_column_list:
        newdf[col]=dataframe[col]
    return newdf
