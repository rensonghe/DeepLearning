#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
import tensorflow_core.python.data.experimental
from statsmodels.tsa import stattools

data_frame = pd.read_csv('2020-2021.csv',
                         names=['name', 'tradingprice', 'tradingvolume',
        'time','sellprice','sellvolume',
                        'buyprice','buyvolume','holdvolume','exchange'],
                         index_col=0, parse_dates=True)


#%%
data_frame['time'] = pd.to_datetime(data_frame['time'])
data_frame = data_frame.fillna(data_frame['sellprice'].mean())
data_frame = data_frame.fillna(data_frame['buyprice'].mean())

#%%
def get_index(data):
    sp = data_frame['sellprice']
    bp = data_frame['buyprice']
    sv = data_frame['sellvolume']
    bv = data_frame['buyvolume']
    data_frame['pricediff'] = sp-bp
    data_frame['volumedeep'] = sv+bv
    data_frame['volumeratio'] = np.log(sv/bv)
    data_frame['pricemean'] = (sp*bv+bp*sv)/(sv+bv)
    data_frame['buyratio'] = np.log(bv/sv+bv)
    data_frame['sellratio'] = np.log(sv/sv+bv)

    return data
#%%
data_index = data_frame.apply(get_index)
#%%
data_index['sellratio'] = data_index['sellratio'].fillna(0)

#%%
freq = '2min'
data_2min_bar = data_index.set_index(['time']).groupby(pd.Grouper(freq=freq)).agg(np.mean)
#%%
# bv1 = data_2min_bar['buyvolume']
# sv1 = data_2min_bar['sellvolume']
# data_2min_bar['buyratio'] = np.log(bv1/sv1+bv1)
# data_2min_bar['sellratio'] = np.log(sv1/sv1+bv1)
# data_2min_bar['ROE'] = data_2min_bar['tradingprice'].pct_change()
#%%
data_2min_bar = data_2min_bar.dropna(axis=0, how='all')
#%%
data_2min_bar['ROE'] = data_2min_bar['tradingprice'].pct_change()
#%%
# data_2min_bar['volumeratio'] = data_2min_bar['volumeratio'].fillna(0)
data_2min_bar['ROE'] = data_2min_bar['ROE'].fillna(0)
data_2min_bar['volumeratio'] = data_2min_bar['volumeratio'].fillna(0)
data_2min_bar['buyratio'] = data_2min_bar['buyratio'].fillna(0)
#%%
data_2min_bar= data_2min_bar.fillna(data_2min_bar['sellratio'].mean())
data_2min_bar= data_2min_bar.fillna(data_2min_bar['buyratio'].mean())
#%%
data_2min_bar['volumeratio'][np.isinf(data_2min_bar['volumeratio'])] = 0
data_2min_bar['buyratio'][np.isinf(data_2min_bar['buyratio'])] = 0
#%%
data_2min_bar.to_csv('data_2min_bar_2020_2021.csv')
#%%
## ACF
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import acf
# data_2min_bar['volumeratio'] = data_2min_bar['volumeratio'].fillna(0)
#%%
#calling auto correlation function
lag_acf = stattools.pacf(data_2min_bar['sellratio'], nlags=50)
#Plot ACF:
plt.figure(figsize=(16, 7), dpi=72)
plt.plot(lag_acf,marker='+')
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_2min_bar['sellratio'])),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_2min_bar['sellratio'])),linestyle='--',color='gray')
plt.title('Autocorrelation sellratio')
plt.xlabel('number of lags')
plt.ylabel('correlation')
plt.tight_layout()
plt.show()
#%%
print('自相关系数: ', stattools.acf(data_2min_bar['buyratio'], nlags=100,missing='conservative'))
