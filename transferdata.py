import os.path
import numpy
import numpy as np
import pandas as pd
import random

from pandas import DataFrame
#%%
txt_path='Z:/TickFile_test/ni1910/'   #获取txt文件路径
print(txt_path)
txt_list=os.listdir(txt_path)    #将txt文件存入列表中
print(txt_list)
print(len(txt_list))
csv_file=[]   #创建保存csv文件内容的列表
csv_file_title=['name', 'tradingprice', 'tradingvolume', 'date', 'time','sellprice','sellvolume',
                        'buyprice','buyvolume','holdvolume','exchange','none']    #设置行索引
for txt in txt_list:    #遍历txt文件
      f=np.loadtxt(txt_path+txt, dtype=str, delimiter=',')    #读入当前txt文件的内容
      csv_file.extend(f)    #将处理之后的内容放入列表中
csv_file=numpy.array(csv_file) #将列表转换为数组
df=DataFrame(csv_file)
#%%
# print(type(csv_file))
# df=numpy.savetxt('ni1910.csv',csv_file)    #用numpy保存数据为csv。此时里面内容没有行索引，且数据为浮点型，丧失了csv文件的格式
# df=pd.read_csv('ni1910.csv',names=csv_file_title)    #用pd读存好的csv。设置行索引
df.to_csv('ni.csv',index=True)    #将具有索引的数据重新保存成csv文件，此时不会丧失csv文件的格式。数据会以原来的形式保存。最终结果如下
print(df)

#
# df['sellprice'] = df['sellprice'].fillna(df['sellprice'].median)
# df['sellvolume'] = df['sellvolume'].fillna(df['sellvolume'].median)
# df['buyprice'] = df['buyprice'].fillna(df['buyprice'].median)
# df['buyvolume'] = df['buyvolume'].fillna(df['buyvolume'].median)
#
# data = df.drop(['name','date','none'], axis=1)
#%%
