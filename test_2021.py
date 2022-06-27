import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np

#%%
data = pd.read_csv('test_2021.csv')
#%%
import ta
data_time = ta.utils.dropna(data)
data_time = ta.add_all_ta_features(data_time, "open", "high", "low", "close", "volume", fillna=True)
#%%

civ_df = pd.read_csv('output_feature_detail.csv')
# 删除iv值过小的变量
iv_thre = 0.01
iv = civ_df[['var_name', 'iv']].drop_duplicates()
x_columns = iv.var_name[iv.iv > iv_thre]
y_columns = x_columns.tolist()

data = data_time.reindex(columns=y_columns)

drop_list = ['open','high','low','volume_vwap','volatility_bbm','volatility_bbh',
             'volatility_kcc','volatility_kch','volatility_dcl','volatility_dch','trend_sma_fast',
             'trend_ema_fast','trend_ema_slow','trend_ichimoku_conv','trend_ichimoku_base',
             'trend_ichimoku_a','trend_visual_ichimoku_a','trend_visual_ichimoku_b','trend_psar_up']
data = data.drop(drop_list, axis=1)

data['target'] = data_time['close']
data['time'] = data_time['time']
#%%
data.to_csv('test_2021.csv')
#%%
import tensorflow as tf
import math
import sklearn.metrics as skm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import RepeatVector, TimeDistributed
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
#%%
data = data.set_index(['time'])
#%%
test_set = data[(data.index>='2021-01-01 0:00:00')& (data.index<='2021-12-31 23:59:59')]
#%%
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
testing_set_scaled = sc.fit_transform(test_set)
#%%
from numpy import array

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
X_test, y_test = split_sequences(testing_set_scaled, n_steps_in,n_steps_out)
#%%
from tensorflow.keras.models import load_model
model = load_model('lstm_model_35var_2019+2020_30min_10min.h5')
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

forecast = predict_col_0[:,-1]
#%%
# 真实值返归一化
true = y_test[:,-1]
true_col_0 = inverse_transform_col(sc, true, n_col=-1)
truth = true_col_0[:,-1]
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
#%%
test_set.astype(int)
true_result.astype(int)
#%%
prepare_data = test_set[34:]

#%%
col = ['predict']
predict_result.columns = col
col1 = ['close']
true_result.columns = col1
#%%
a = prepare_data.reset_index()
result = a.join(predict_result)
result.to_csv('test_result_2021_30min_10min.csv')
#%%
plt.plot(truth, color='red')
plt.plot(forecast, color='blue')
plt.title('Close Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
#%%
data_test = pd.read_csv('test_result_2021_30min_10min.csv')
#%%
# data_test = pd.to_datetime(data_test['time'])
#%%
data_test = data_test.set_index(['time'])
#%%
test_set_9 = data_test[(data_test.index>='2021-09-01 0:00:00')& (data_test.index<='2021-09-31 23:59:59')]
test_set_10 = data_test[(data_test.index>='2021-10-22 10:54:00')& (data_test.index<='2021-10-23 0:00:00')]
test_set_11 = data_test[(data_test.index>='2021-11-01 0:00:00')& (data_test.index<='2021-11-30 23:59:59')]
test_set_10_15 = data_test[(data_test.index>='2021-10-01 0:00:00')& (data_test.index<='2021-10-15 23:59:59')]
#%%
plt.plot(test_set_10['target'][0:125], color='red')
plt.plot(test_set_10['predict'][0:125], color='blue')
plt.title('10 month')
plt.xlabel('Time')
plt.ylabel('Price')
# plt.legend()
plt.show()
