import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import array
import tensorflow as tf
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

import math
import sklearn.metrics as skm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

#%%
data = pd.read_csv('test.csv')
#%%
data['log_return'] = np.log(data['target'] / data['target'].shift(1))
data = data.dropna(axis=0, how='any')
#%%
data['target'].replace(-np.inf,-99)
data['target'].replace(np.inf, 99)
#%%
data['log_return'][data.log_return>0]=1
data['log_return'][data.log_return<0]=-1
#%%
# data['target'][np.isinf(data['target'])] = 0
data = data.iloc[:,1:]
#%%
training_set = data[:100000]
test_set = data[-6000:]
#%%
#将数据归一化，范围是0到1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(test_set)
print(len(training_set_scaled))
#%%
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
#%%
n_steps_in, n_steps_out = 30, 10
#%%
X_train, y_train = split_sequences(training_set_scaled, n_steps_in,n_steps_out)
X_test, y_test = split_sequences(testing_set_scaled, n_steps_in,n_steps_out)
#%%
n_features = X_train.shape[2]
#%%
from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(50, activation='relu'))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',metrics=['accuracy'])
#%%
n_epochs = 50
# train model
history = model.fit(X_train,y_train,      #revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(X_test, y_test),  #revise
                    validation_freq=1)                  #测试的epoch间隔数

model.summary()