import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np

#%%
# cleaning column and index ni1910
# df = pd.read_csv('ni.csv')
# df = pd.DataFrame(df)
# col = ['index','name', 'tradingprice', 'tradingvolume', 'date',
#                                  'time','sellprice','sellvolume',
#                         'buyprice','buyvolume','holdvolume','exchange','none']
# df.columns = col
# df=df.loc[df['name']=='ni1910']
# data = df.drop(['index','name','date','none'], axis=1)

#%%
# figure out whether have the Nah or not and fill with median
# data = data.fillna(data['sellprice'].mean())
# print(data.isnull().any())
# print(data.info())
#%%
import os

path = r'D:\luojie\ni'  # 获取文件目录，下面是所有的要合并的csv文件

# 新建列表存放每个文件数据(依次读取多个相同结构的Excel文件并创建DataFrame)
DFs = []   #存放多个DataFrame

for root, dirs, files in os.walk(path):  # 第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
    for file in files: #files是包含所有文件名的一个列表
        # print(root)
        file_path = os.path.join(root, file)  #路径拼接;os.path.join()函数：连接两个或更多的路径名组件
        # print(file_path)
        df = pd.read_csv(file_path, encoding="gbk")  # 将excel转换成DataFrame
        DFs.append(df)  # 多个df的list
# 将多个DataFrame合并为一个
df = pd.concat(DFs) #concat:合并

# 写入excel或者csv文件，不包含索引数据
# df.to_excel(r'D:\python脚本\csv合并结果1.xlsx', index=False)  #
df.to_csv(r'D:\luojie\ni_concat.csv', index=False),#写入csv文件中比较好，单元格格式不会有其他格式；
                                                            #如果合并的文件中文是乱码，可以指定编码格式；例如：encoding="gbk"


#%%
data_1905 = pd.read_csv('ni1905.csv')
data_1905.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_1905['diff_holdvolume'] = data_1905['close_oi'] - data_1905['open_oi']
data_1905 = data_1905.drop(['timestamp','open_oi'], axis=1)
data_1905 = data_1905[(data_1905.time>='2019-01-02 0:00:00')& (data_1905.time<='2019-04-08 0:00:00')]

data_1906 = pd.read_csv('ni1906.csv')
data_1906.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_1906['diff_holdvolume'] = data_1906['close_oi'] - data_1906['open_oi']
data_1906 = data_1906.drop(['timestamp','open_oi'], axis=1)
data_1906 = data_1906[(data_1906.time>='2019-04-09 0:00:00')& (data_1906.time<='2019-04-30 0:00:00')]

data_1907 = pd.read_csv('ni1907.csv')
data_1907.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_1907['diff_holdvolume'] = data_1907['close_oi'] - data_1907['open_oi']
data_1907 = data_1907.drop(['timestamp','open_oi'], axis=1)
data_1907 = data_1907[(data_1907.time>='2019-05-06 0:00:00')& (data_1907.time<='2019-06-03 0:00:00')]

data_1908 = pd.read_csv('ni1908.csv')
data_1908.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_1908['diff_holdvolume'] = data_1908['close_oi'] - data_1908['open_oi']
data_1908 = data_1908.drop(['timestamp','open_oi'], axis=1)
data_1908 = data_1908[(data_1908.time>='2019-06-04 0:00:00')& (data_1908.time<='2019-07-04 0:00:00')]

data_1909 = pd.read_csv('ni1909.csv')
data_1909.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_1909['diff_holdvolume'] = data_1909['close_oi'] - data_1909['open_oi']
data_1909 = data_1909.drop(['timestamp','open_oi'], axis=1)
data_1909 = data_1909[(data_1909.time>='2019-07-05 0:00:00')& (data_1909.time<='2019-07-11 0:00:00')]

data_1910 = pd.read_csv('ni1910.csv')
data_1910.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_1910['diff_holdvolume'] = data_1910['close_oi'] - data_1910['open_oi']
data_1910 = data_1910.drop(['timestamp','open_oi'], axis=1)
data_1910 = data_1910[(data_1910.time>='2019-07-12 0:00:00')& (data_1910.time<='2019-08-28 0:00:00')]

data_1911 = pd.read_csv('ni1911.csv')
data_1911.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_1911['diff_holdvolume'] = data_1911['close_oi'] - data_1911['open_oi']
data_1911= data_1911.drop(['timestamp', 'open_oi'], axis=1)
data_1911 = data_1911[(data_1911.time>='2019-08-29 0:00:00')& (data_1911.time<='2019-09-27 0:00:00')]

data_1912 = pd.read_csv('ni1912.csv')
data_1912.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_1912['diff_holdvolume'] = data_1912['close_oi'] - data_1912['open_oi']
data_1912= data_1912.drop(['timestamp', 'open_oi'], axis=1)
data_1912 = data_1912[(data_1912.time>='2019-09-30 0:00:00')& (data_1912.time<='2019-10-30 0:00:00')]

data_2001 = pd.read_csv('ni2001.csv')
data_2001.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2001['diff_holdvolume'] = data_2001['close_oi'] - data_2001['open_oi']
data_2001= data_2001.drop(['timestamp','open_oi'], axis=1)
data_2001 = data_2001[(data_2001.time>='2019-10-31 0:00:00')& (data_2001.time<='2019-11-13 0:00:00')]

data_2002 = pd.read_csv('ni2002.csv')
data_2002.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2002['diff_holdvolume'] = data_2002['close_oi'] - data_2002['open_oi']
data_2002= data_2002.drop(['timestamp', 'open_oi'], axis=1)
data_2002 = data_2002[(data_2002.time>='2019-11-14 0:00:00')& (data_2002.time<='2019-12-31 0:00:00')]

frame = [data_1905,data_1906,data_1907,data_1908,data_1909,data_1910,data_1911,data_1912,data_2001,data_2002]
data = pd.concat(frame)
#%%
data.to_csv('ni_reconstruct.csv')
#%%
data = pd.read_csv('ni_reconstruct.csv')
#%%
# resample the data by time period
data['time'] = pd.to_datetime(data['time'])
start_time = datetime.datetime.strptime('09:00:00', '%H:%M:%S').time()
end_time = datetime.datetime.strptime('10:15:00','%H:%M:%S').time()

start_time1 = datetime.datetime.strptime('13:30:00', '%H:%M:%S').time()
end_time1 = datetime.datetime.strptime('15:00:00','%H:%M:%S').time()

start_time2 = datetime.datetime.strptime('21:00:00', '%H:%M:%S').time()
end_time2 = datetime.datetime.strptime('23:59:29','%H:%M:%S').time()

start_time3 = datetime.datetime.strptime('10:30:00', '%H:%M:%S').time()
end_time3 = datetime.datetime.strptime('11:30:00','%H:%M:%S').time()

start_time4 = datetime.datetime.strptime('00:00:00', '%H:%M:%S').time()
end_time4 = datetime.datetime.strptime('02:00:00','%H:%M:%S').time()

data_time = data[(data.time.dt.time >= start_time) & (data.time.dt.time <= end_time)|
                 (data.time.dt.time >= start_time1) & (data.time.dt.time <= end_time1)|
                 (data.time.dt.time >= start_time2) & (data.time.dt.time <= end_time2)|
                (data.time.dt.time >= start_time3) & (data.time.dt.time <= end_time3)|
                 (data.time.dt.time >= start_time4) & (data.time.dt.time <= end_time4)]
#%%
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
#
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# fig = plt.figure(figsize=(20, 5))
# ax = fig.add_subplot(1, 1, 1)
#
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  #設置x軸主刻度顯示格式（日期）
# plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=10))  #設置x軸主刻度間距
#
# plt.xlabel('日期')
# plt.ylabel('持仓量')
# plt.title('data_1911持仓量折线图')
# plt.plot(data_1911['time'], data_1911['holdvolume'])
# plt.show()


#%%
import ta
data_time = ta.utils.dropna(data_time)
#%%
# window = 12
# data_time[f"roc_{window}"] = ta.momentum.ROCIndicator(close=data_time["close"], window=window).roc()
#%%
data_time = ta.add_all_ta_features(data_time, "open", "high", "low", "close", "volume", fillna=True)

#%%
data_time = data_time.drop(['time'], axis=1)
#%%
corr = data_time.corr()
#%%
import seaborn as sns
import warnings
from pyforest import *
ax = plt.subplots(figsize=(100,100))
ax = sns.heatmap(corr, vmax=1.0, square=True, annot=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.show()
plt.savefig('corr_1.0.png')

#%%
# from statsmodels.tsa.stattools import adfuller
# adf = adfuller(data_time['log_return'])
# print(adf)
#%%
# def get_index(data):
#     sp = data_time['sellprice']
#     bp = data_time['buyprice']
#     sv = data_time['sellvolume']
#     bv = data_time['buyvolume']
#     data_time['pricediff'] = sp-bp
#     data_time['volumedeep'] = sv+bv
#     data_time['volumeratio'] = np.log(sv/bv)
#     data_time['pricemean'] = (sp*bv+bp*sv)/(sv+bv)
#     return data
# %%
# data_index = data_time.apply(get_index)
#
#
# #%%
# freq = '2min'
# a = data_index.set_index(['time']).groupby(pd.Grouper(freq=freq)).get_group('new')
# #%%
# #%%
# time_group = data_index.set_index(['time'])
# #%%
# # grouped by 15min freq
# freq = '2min'
# new_by_group = time_group.groupby(pd.Grouper(freq=freq)).agg(np.mean)
#
# #%%
# data_time_index = new_by_group.dropna(axis=0, how='all')
#%%
data_time.to_csv('data_ni_2019.csv')
#%%
# time = np.array(data['time'])
# # print(time)
# a=[]
# for i in range(len(time)):
#     timeArray = datetime.datetime.strptime(time[i],'%Y/%m/%d %H:%M:%S.%f')
#     timeStamp = datetime.datetime.timestamp(timeArray)
#     a.append(timeStamp)
# data['time'] = a
# print(data)
#%%
data_time = pd.read_csv('data_ni_2019.csv')
#%%
data_time = data_time.iloc[:,1:]
#%%
data_time['target'] = np.log(data_time['close']/data_time['close'].shift(1))
#%%
data_time['target'] = data_time['target'].shift(1)
#%%
data_test = data_time.dropna(axis=0, how='any')
#%%
drop_list = ['open', 'high','low','volume_vwap','volatility_bbm','volatility_bbh',
             'volatility_kcc','volatility_kch','volatility_dcl','volatility_dch','trend_sma_fast',
             'trend_ema_fast','trend_ema_slow','trend_ichimoku_conv','trend_ichimoku_base',
             'trend_ichimoku_a','trend_visual_ichimoku_a','trend_visual_ichimoku_b','trend_psar_up','others_cr']
data = data.drop(drop_list, axis=1)
#%%
data_time['target'] = data_time['close']
#%%
data_test= data_time.drop(['close'],axis=1)
#%% 计算IV
import woe.feature_process as fp
import woe.eval as eval

data_woe = data_test  # 用于存储所有数据的woe值
civ_list = []
n_positive = sum(data_test['target'])
n_negtive = len(data_test) - n_positive
for column in list(data_test.columns[0:90]):
    if data_time[column].dtypes == 'object':
        civ = fp.proc_woe_discrete(data_test, column, n_positive, n_negtive, 0.05 * len(data_test), alpha=0.05)
    else:
        civ = fp.proc_woe_continuous(data_test, column, n_positive, n_negtive, 0.05 * len(data_test), alpha=0.05)
    civ_list.append(civ)
    data_woe[column] = fp.woe_trans(data_test[column], civ)

civ_df = eval.eval_feature_detail(civ_list, 'output_feature_detail_log_return.csv')
#%%
civ_df = pd.read_csv('output_feature_detail_log_return.csv')
# 删除iv值过小的变量
iv_thre = 0.01
iv = civ_df[['var_name', 'iv']].drop_duplicates()
#%%
x_columns = iv.var_name[iv.iv > iv_thre]
y_columns = x_columns.tolist()
#%%
data = data_time.reindex(columns=y_columns)
#%%
corr = data.corr()
#%%
import seaborn as sns
import warnings
from pyforest import *

ax = plt.subplots(figsize=(100,100))
ax = sns.heatmap(corr, vmin=0, vmax=1, square=True, annot=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.show()
plt.savefig('new_corr.png')
#%%
drop_list = ['open','high','low','volume_vwap','volatility_bbm','volatility_bbh',
             'volatility_kcc','volatility_kch','volatility_dcl','volatility_dch','trend_sma_fast',
             'trend_ema_fast','trend_ema_slow','trend_ichimoku_conv','trend_ichimoku_base',
             'trend_ichimoku_a','trend_visual_ichimoku_a','trend_visual_ichimoku_b','trend_psar_up']
data = data.drop(drop_list, axis=1)
#%%
corr_34var=data.corr()
#%%
ax = plt.subplots(figsize=(100,100))
ax = sns.heatmap(corr_34var, vmin=0, vmax=1, square=True, annot=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('corr_34var.png')
#%%
data['target'] = data_time['close']
data.to_csv('test.csv')



