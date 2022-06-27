import datetime

import matplotlib.sphinxext.plot_directive
import tensorflow as tf

# gpus = tf.config.list_physical_devices("GPU")

# if gpus:
#     tf.config.experimental.set_memory_growth(gpus[0], True)  #设置GPU显存用量按需使用
#     tf.config.set_visible_devices([gpus[0]],"GPU")

import pandas            as pd
import tensorflow        as tf
import numpy             as np
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from numpy                 import array
from sklearn               import metrics
from sklearn.preprocessing import MinMaxScaler
from keras.models          import Sequential
from keras.layers          import Dense,LSTM,Bidirectional
# import matplotlib; matplotlib.use('Qt5Agg')

# 确保结果尽可能重现
from numpy.random import seed
seed(1)
tf.random.set_seed(1)
#%%
# 设置相关参数
n_timestamp = 30    # 时间戳
n_epochs = 20    # 训练轮数

#%%
data = pd.read_csv('data_2min_bar_new.csv')
#%%
# data = data['close'].astype('float64')

#%%
# data = data.fillna(data['volumeratio'].mean())
data = data.drop(['time','sellratio','buyratio','tradingprice'],axis=1)
#%%

training_set = data[:52000]
test_set = data[53000:]
#%%
#将数据归一化，范围是0到1
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(test_set)
print(len(training_set_scaled))
#%%
