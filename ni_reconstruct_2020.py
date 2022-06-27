import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np

#%%
data_2003 = pd.read_csv('ni2003.csv')
data_2003.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2003['diff_holdvolume'] = data_2003['close_oi'] - data_2003['open_oi']
data_2003 = data_2003.drop(['timestamp','open_oi'], axis=1)
data_2003 = data_2003[(data_2003.time>='2020-01-02 0:00:00')& (data_2003.time<='2020-01-22 0:00:00')]

data_2004 = pd.read_csv('ni2004.csv')
data_2004.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2004['diff_holdvolume'] = data_2004['close_oi'] - data_2004['open_oi']
data_2004 = data_2004.drop(['timestamp','open_oi'], axis=1)
data_2004 = data_2004[(data_2004.time>='2020-01-23 0:00:00')& (data_2004.time<='2020-03-02 0:00:00')]

data_2006 = pd.read_csv('ni2006.csv')
data_2006.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2006['diff_holdvolume'] = data_2006['close_oi'] - data_2006['open_oi']
data_2006 = data_2006.drop(['timestamp','open_oi'], axis=1)
data_2006 = data_2006[(data_2006.time>='2020-03-03 0:00:00')& (data_2006.time<='2020-04-28 0:00:00')]

data_2007 = pd.read_csv('ni2007.csv')
data_2007.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2007['diff_holdvolume'] = data_2007['close_oi'] - data_2007['open_oi']
data_2007 = data_2007.drop(['timestamp','open_oi'], axis=1)
data_2007 = data_2007[(data_2007.time>='2020-04-29 0:00:00')& (data_2007.time<='2020-05-25 0:00:00')]

data_2008 = pd.read_csv('ni2008.csv')
data_2008.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2008['diff_holdvolume'] = data_2008['close_oi'] - data_2008['open_oi']
data_2008 = data_2008.drop(['timestamp','open_oi'], axis=1)
data_2008 = data_2008[(data_2008.time>='2020-05-26 0:00:00')& (data_2008.time<='2020-07-02 0:00:00')]

data_2010 = pd.read_csv('ni2010.csv')
data_2010.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2010['diff_holdvolume'] = data_2008['close_oi'] - data_2010['open_oi']
data_2010 = data_2010.drop(['timestamp','open_oi'], axis=1)
data_2010 = data_2010[(data_2010.time>='2020-07-03 0:00:00')& (data_2010.time<='2020-08-24 0:00:00')]

data_2011 = pd.read_csv('ni2011.csv')
data_2011.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2011['diff_holdvolume'] = data_2011['close_oi'] - data_2011['open_oi']
data_2011= data_2011.drop(['timestamp', 'open_oi'], axis=1)
data_2011 = data_2011[(data_2011.time>='2020-08-25 0:00:00')& (data_2011.time<='2020-09-23 0:00:00')]

data_2012 = pd.read_csv('ni2012.csv')
data_2012.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2012['diff_holdvolume'] = data_2012['close_oi'] - data_2012['open_oi']
data_2012= data_2012.drop(['timestamp', 'open_oi'], axis=1)
data_2012 = data_2012[(data_2012.time>='2020-09-24 0:00:00')& (data_2012.time<='2020-11-02 0:00:00')]

data_2102 = pd.read_csv('ni2102.csv')
data_2102.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
data_2102['diff_holdvolume'] = data_2102['close_oi'] - data_2102['open_oi']
data_2102= data_2102.drop(['timestamp','open_oi'], axis=1)
data_2102 = data_2102[(data_2102.time>='2020-11-03 0:00:00')& (data_2102.time<='2020-12-21 0:00:00')]

# data_2002 = pd.read_csv('ni2002.csv')
# data_2002.columns = ['time','timestamp','open','high','low','close','volume','open_oi','close_oi']
# data_2002['diff_holdvolume'] = data_2002['close_oi'] - data_2002['open_oi']
# data_2002= data_2002.drop(['timestamp', 'open_oi'], axis=1)
# data_2002 = data_2002[(data_2002.time>='2019-11-14 0:00:00')& (data_2002.time<='2019-12-31 0:00:00')]
#%%
frame = [data_2003,data_2004,data_2006,data_2007,data_2008,data_2010,data_2011,data_2012,data_2102]
data = pd.concat(frame)
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
data_time.to_csv('ni_2020.csv')
#%%
import ta
data_time = ta.utils.dropna(data_time)
data_time = ta.add_all_ta_features(data_time, "open", "high", "low", "close", "volume", fillna=True)

#%%
civ_df = pd.read_csv('output_feature_detail.csv')
# 删除iv值过小的变量
iv_thre = 0.01
iv = civ_df[['var_name', 'iv']].drop_duplicates()
x_columns = iv.var_name[iv.iv > iv_thre]
y_columns = x_columns.tolist()
#%%
data = data_time.reindex(columns=y_columns)
#%%
drop_list = ['open','high','low','volume_vwap','volatility_bbm','volatility_bbh',
             'volatility_kcc','volatility_kch','volatility_dcl','volatility_dch','trend_sma_fast',
             'trend_ema_fast','trend_ema_slow','trend_ichimoku_conv','trend_ichimoku_base',
             'trend_ichimoku_a','trend_visual_ichimoku_a','trend_visual_ichimoku_b','trend_psar_up']
data = data.drop(drop_list, axis=1)

#%%
data['target'] = data_time['close']
data['time'] = data_time['time']
#%%
data.to_csv('test_2020.csv')