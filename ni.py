import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np


#%%
# cleaning column and index ni1910
df = pd.read_csv('ni_2019-2020.csv')
df = pd.DataFrame(df)
col = ['name', 'tradingprice', 'tradingvolume',
        'time','sellprice','sellvolume',
                        'buyprice','buyvolume','holdvolume','exchange']
df.columns = col
# df=df.loc[df['name']=='ni1910']
df.to_csv('ni1903.csv')
#%%
data = df.drop(['name'], axis=1)
data = data.fillna(data['sellprice'].mean())
data['time'] = pd.to_datetime(data['time'])
#%%
freq = '2min'
data_1 = data.set_index(['time']).groupby(pd.Grouper(freq=freq))

#%%
data_2 = data_1.agg(np.mean)

#%%
a = pd.DataFrame(data, columns=['time','sellprice', 'buyprice'])
#%%
data_ask = a['sellprice'].resample('1Min').ohlc().set_index(['time'])