#%%
import matplotlib.pyplot as plt
import pandas as pd


data_frame = pd.read_csv('ni_2019-2020.csv',
                         names=['name', 'tradingprice', 'tradingvolume',
        'time','sellprice','sellvolume',
                        'buyprice','buyvolume','holdvolume','exchange'],
                         index_col=0, parse_dates=True)

#%%
data_frame['time'] = pd.to_datetime(data_frame['time'])
data_frame.set_index(['time'], inplace=True)

#%%
data_ask = data_frame['tradingprice'].resample('1Min').ohlc()

def get_1min_diff(data):
    high = data_ask['high']
    low = data_ask['low']
    data_ask['diff'] = high - low
    return data

data_diff = data_ask.apply(get_1min_diff)

data_ask = data_ask.dropna(axis=0, how='all')

diff_1 = data_ask['diff']
t_time = data_ask.index
plt.figure(figsize=(20,8), dpi=72)
plt.plot(t_time,diff_1,label='1min')
plt.legend(loc=0, frameon=True)
plt.ylabel('diff')
plt.show()

#%%
data_ask_2 = data_frame['tradingprice'].resample('2Min').ohlc()

def get_2min_diff(data):
    high = data_ask_2['high']
    low = data_ask_2['low']
    data_ask_2['diff'] = high - low
    return data

data_diff = data_ask_2.apply(get_2min_diff)
data_ask_2 = data_ask_2.dropna(axis=0, how='all')

diff_2 = data_ask_2['diff']
t_time = data_ask_2.index
plt.figure(figsize=(20,8), dpi=72)
plt.plot(t_time,diff_2,label='2min')
plt.legend(loc=0, frameon=True)
plt.ylabel('diff')
plt.show()
#%%
data_ask_0_5 = data_frame['tradingprice'].resample('0.5Min').ohlc()

def get_2min_diff(data):
    high = data_ask_0_5['high']
    low = data_ask_0_5['low']
    data_ask_0_5['diff'] = high - low
    return data

data_diff = data_ask_0_5.apply(get_2min_diff)

data_ask_0_5 = data_ask_0_5.dropna(axis=0, how='all')

diff_0_5 = data_ask_0_5['diff']
t_time = data_ask_0_5.index
plt.figure(figsize=(20,8), dpi=72)
plt.plot(t_time,diff_0_5,label='0.5min')
plt.legend(loc=0, frameon=True)
plt.ylabel('diff')
plt.show()

#%%
data_ask_5 = data_frame['tradingprice'].resample('5Min').ohlc()

def get_2min_diff(data):
    high = data_ask_5['high']
    low = data_ask_5['low']
    data_ask_5['diff'] = high - low
    return data

data_diff = data_ask_5.apply(get_2min_diff)

data_ask_5 = data_ask_5.dropna(axis=0, how='all')

diff_5 = data_ask_5['diff']
t_time = data_ask_5.index
plt.figure(figsize=(20,8), dpi=72)
plt.plot(t_time,diff_5,label='5min')
plt.legend(loc=0, frameon=True)
plt.ylabel('diff')
plt.show()

#%%
from statsmodels.tsa import stattools
# data = data_ask_2['tradingprice']

plt.stem(stattools.acf(data_frame['tradingprice'],nlags=100))
print('自相关系数: ', stattools.acf(data_frame['tradingprice'], nlags=100))
# print('自相关系数: ', stattools.pacf(df.pct_chg, nlags=10))