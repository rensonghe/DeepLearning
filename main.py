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
data = pd.read_csv('test2103_1_04_1_08.csv')
#%%
data['target'] = np.log(data['wap1_mean'] / data['wap1_mean'].shift(1)) * 100
# data['log_return'][np.isinf(data['log_return'])] = 0
data['target'] = data['target'].shift(-1)
data = data.dropna(axis=0, how='any')
#%%
data = data.iloc[:,2:]
#%%
data = data.drop(['datetime'],axis=1)
#%%
# time = np.array(data['time'])
# print(time)
# a=[]
# for i in range(len(time)):
#     timeArray = datetime.datetime.strptime(time[i],'%Y-%m-%d %H:%M:%S')
#     timeStamp = datetime.datetime.timestamp(timeArray)
#     a.append(timeStamp)
# data['time'] = a
# print(data)
#%%
training_set = data[:200000]
test_set = data[200000:250000]
#%%
#将数据归一化，范围是0到1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(-1, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(test_set)
print(len(training_set_scaled))
#%% 多变量单步预测
# 取前 n_timestamp 天的数据为 X；n_timestamp+1天数据为 Y。
# def data_split(sequence, n_timestamp):
#     X = []
#     y = []
#     for i in range(len(sequence)):
#         end_ix = i + n_timestamp
#
#         if end_ix > len(sequence) - 1:
#             break
#
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#         # print(len(X))
#         # print(y)
#     return array(X), array(y)
#%% 多变量多步预测
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
# X_train = X_train.reshape(1, X_train.shape[1],X_train)
#%%
# x_input = array([[60, 65, 125], [70, 75, 145], [80, 85, 165]])
# x_input = x_input.reshape((1, 3, 3))
#%%
# X_train, y_train = data_split(training_set_scaled, n_timestamp)
# X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],-1)
# # X_train = X_train.reshape(-1,X_train.shape[1],1)
# # y_train = y_train.reshape(-1,1)
# #%%
# X_test, y_test = data_split(testing_set_scaled, n_timestamp)
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],-1)
# # X_test = X_test.reshape(-1, X_test.shape[1], 1)
# # y_test = y_test.reshape(-1,1)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

#%%
n_features = X_train.shape[2]
from tensorflow.keras.layers import Dropout
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
# model.add(Dropout(0.8))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(50, activation='relu', return_sequences=True))
# model.add(Dropout(0.8))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='Adam', loss='mean_squared_error',metrics=['accuracy'])
#%%
model.summary()  # 输出模型结构

#%%
# activate model
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
# model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
#               loss='mean_squared_error',
# 			  metrics=['accuracy'])  # 损失函数用均方误差

#%%
n_epochs = 50
# train model
history = model.fit(X_train,y_train,      #revise
                    batch_size=128,
                    epochs=n_epochs,
                    validation_data=(X_test, y_test),  #revise
                    validation_freq=1)                  #测试的epoch间隔数

model.summary()
#%%
model.save('lstm_model_30step_10min.h5')
#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
# plt.savefig('loss.png')
#%%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy vs val_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()   # 输出训练集测试集结果
#%%
predicted_contract_price = model.predict(X_test)        # 测试集输入模型进行预测
#%%
def inverse_transform_col(scaler, y, n_col):
    '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
    y = y.copy()
    y -= sc.min_[n_col]
    y /= sc.scale_[n_col]
    return y

#%%
# 第3列归一化的值
predict = predicted_contract_price[:, -1]

# 第3列反归一化
predict_col_0 = inverse_transform_col(sc, predict, n_col=-1)

#%%
# 真实值返归一化
true = y_test[:,-1]
true_col_0 = inverse_transform_col(sc, true, n_col=-1)
#%%
truth = true_col_0[:, -1]
forecast = predict_col_0[:,-1]
#%%
# 画出真实数据和预测数据的对比曲线
plt.plot(true_col_0[-1000:], color='red',label='真实值')
plt.plot(predict_col_0[-1000:], color='blue',label='预测值')
plt.title('contract close Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
# plt.legend()
# plt.savefig('predict_close.png')
plt.show()
plt.close()
#%%
from sklearn import metrics
'''
MSE  ：均方误差    ----->  预测值减真实值求平方后求均值
RMSE ：均方根误差  ----->  对均方误差开方
MAE  ：平均绝对误差----->  预测值减真实值求绝对值后求均值
R2   ：决定系数，可以简单理解为反映模型拟合优度的重要的统计量
'''
MSE = metrics.mean_squared_error(truth,forecast)
RMSE = metrics.mean_squared_error(truth,forecast)**0.5
MAE = metrics.mean_absolute_error(truth,forecast)
R2 = metrics.r2_score(truth,forecast)
# ACC = metrics.accuracy_score(true_col_0,predict_col_0)

print('均方误差: %.5f' % MSE)
print('均方根误差: %.5f' % RMSE)
print('平均绝对误差: %.5f' % MAE)
print('R2: %.5f' % R2)
# print('ACC:%.5f' % ACC)

#%%
from sklearn.metrics import explained_variance_score

accuracy = explained_variance_score(truth, forecast)
print('准确率：%.5f' % accuracy)

#%%
true_result = pd.DataFrame(truth)
predict_result = pd.DataFrame(forecast)

test_set.astype(int)
true_result.astype(int)

prepare_data = test_set[34:]
#%%
col = ['predict']
predict_result.columns = col
col1 = ['close']
true_result.columns = col1
#%%
a = prepare_data.reset_index()
result = a.join(predict_result)
#%%
a = len(result[(result.target>0)&(result.predict>0)|((result.target<0)&(result.predict<0))])/len(result)
#%%
result['close'] = result['predict']
result = result.drop(['predict'], axis=1)
#%%
result.to_csv('predict_25var_30_5_6month.csv')
a.to_csv('true_25var_30_5_5_6month.csv')
#%%
# b = a.set_index(['time'])
# b = b[27900:].index
#%%
p = result['close']
b = a['close']
c = true_result.join(predict_result)