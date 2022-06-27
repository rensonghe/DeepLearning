import pandas as pd
import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
import warnings
from collections import deque
warnings.filterwarnings('ignore')
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)
#%%
#检查na缺失值
def nan_checker(df):
    df_nan = pd.DataFrame([[var, df[var].isna().sum(), df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'number', 'dtype'])
    df_nan = df_nan.sort_values(by='number', ascending=False).reset_index(drop=True)
    print(df_nan)

#除了郑商所的合约（合约抬头为大写如CY等），其他合约的夜盘时间为第二日，为防止时间戳sort之后紊乱，将其日期往前调整一日
def prev_day(date_str):
    ds = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    dt = datetime.timedelta(hours=24)
    return (ds-dt).strftime("%Y-%m-%d")

#对temp1的数据进行处理，如果同一秒内只有一个数据，则其时间戳格式为00:00:00.0，如有2个数据，第一个数据的时间戳
#格式为00:00:00.0，第二个为00:00:00.500000， 如有3个数据，第一个数据的时间戳格式为00:00:00.0，
#第二个和第三个皆为00:00:00.500000
#注意，500000f这个值为python中固定半秒值，切不可增大或减小，因为之后的dates间隔就是按照500000f来排序的
def clean(df):
    df['temp1'] = df['temp1'].astype(int)
    for i in range(2, len(df)):
        time = df['updatetime'][i].split(':')
        time1 = df['updatetime'][i - 1].split(':')
        time2 = df['updatetime'][i - 2].split(':')
        date = df['tradingday'][i]
        date1 = df['tradingday'][i-1]
        date2 = df['tradingday'][i-2]
        if time == time1 and time != time2 and date == date1: #检查上下日期是否一致
            df['temp1'][i] = df['temp1'][i] + 500000  #这个数值是毫秒的一半，需要非常注意，如果是另外的值，会导致接下来的时间线无法对齐
        elif time == time1 == time2 and date == date1 == date2:
            df['temp1'][i] = df['temp1'][i] + 500000
    df.drop(df.columns[1], axis=1, inplace=True)
    df['date'] = df[df.columns[-2:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
    df['date'] = pd.to_datetime(df['date'], format='%H:%M:%S %f').dt.time
    return df

#将合约的抬头提取出来，如果不是郑商所的合约，则对其夜盘日期往前推一个
def mysplit(s):
    head = s.rstrip('0123456789')
    return head
#最关键的function，对数据进行预处理，将每一个csv表格进行转换，使得其时间能够对齐
#contract是一个string
def preprocess(contract):
    cNames = ['exchangeid', 'instrumentid', 'time', 'lastprice', 'openinterest', 'optigap', 'turnover', 'volume',
              'openvol', 'closevol', 'tradetype', 'tradedir', 'bidprice', 'askprice', 'bidvol', 'askvol']
    df = pd.read_csv(contract, names=cNames, encoding='gbk') #提起数据
    df = df[['instrumentid', 'time', 'lastprice', 'openinterest', 'volume']]
    df['tradingday'], df['updatetime'], df['temp1'] = df['time'].str.split('[ .]').str
    df['temp1']=0 #有些合约的temp1为乱码，将其均设置为0 ，通过clean function对其进行清洗
    df = clean(df)
    grouped = df.groupby(['tradingday', 'date'])
    dfnew = grouped.agg({'instrumentid': 'last', 'lastprice': 'last', 'openinterest': 'last', 'volume': 'sum',\
                         'tradingday': 'last', 'updatetime': 'last', 'date': 'last'})
    #对于同一秒内有3个数据，第一个数据不变，第二和第三个数据因为时间戳一样，通过groupby提起第三个数据的price和interest，将第二个和
    #第三个的volume相加，如此同一秒内最后只有2个数据
    dfnew.reset_index(drop=True, inplace=True) #重新设置index
    dfnew['lp_x_opti'] = dfnew['lastprice'] * dfnew['openinterest'] #计算指数
    df_list = []
    contract_month = contract.split('\\')[-2][-2:] #合约月份，有些合约会有前一个月数据
    heyue=contract.split('\\')[-1].split('.')[0]
    head = mysplit(heyue) #合约名称，如AP，ag，a等
    #将每份csv文件，按照天数，每天分成8段日期，对其进行填充，如果一段时间内的数据量少于5则不进行填充
    for i in dfnew['tradingday'].unique():
        month=i.split('-')[-2]
        if contract_month == month:
            df_split = dfnew[(dfnew['tradingday'] == i)]
            prevday=prev_day(i)
            df11 = df_split[(df_split["updatetime"] >= '00:00:00') & (df_split["updatetime"] < '01:00:00')]
            if len(df11) > 5:
                start = pd.to_datetime('00:00:00')
                end = pd.to_datetime('00:59:59.500000')
                dates = pd.date_range(start=start, end=end, freq='500000U').time
                df11 = df11.set_index('date').reindex(dates).reset_index().reindex(columns=df11.columns)
                cols = df11.columns
                df11[cols] = df11[cols].ffill()
                df11 = df11.fillna(method='bfill')
                df_list.append(df11)

            df12 = df_split[(df_split["updatetime"] >= '01:00:00') & (df_split["updatetime"] < '02:30:00')]
            if len(df12) > 5:
                start = pd.to_datetime('01:00:00')
                end = pd.to_datetime('02:30:00')
                dates = pd.date_range(start=start, end=end, freq='500000U').time
                df12 = df12.set_index('date').reindex(dates).reset_index().reindex(columns=df12.columns)
                cols = df12.columns
                df12[cols] = df12[cols].ffill()
                df12 = df12.fillna(method='bfill')
                df_list.append(df12)

            df2 = df_split[(df_split["updatetime"] >= '09:00:00') & (df_split["updatetime"] <= '10:15:00')]
            if len(df2) > 5:
                start = pd.to_datetime('09:00:00')
                end = pd.to_datetime('10:15:00')
                dates = pd.date_range(start=start, end=end, freq='500000U').time  # 每秒2个数值
                df2 = df2.set_index('date').reindex(dates).reset_index().reindex(columns=df2.columns)
                cols = df2.columns
                df2[cols] = df2[cols].ffill()  # 根据上一个数据向下填充数据
                df2 = df2.fillna(method='bfill')  # 如果一开始没有数据，则根据最新数值向上填充
                df_list.append(df2)

            df3 = df_split[(df_split["updatetime"] >= '10:30:00') & (df_split["updatetime"] <= '11:30:00')]
            if len(df3) > 5:
                start = pd.to_datetime('10:30:00')
                end = pd.to_datetime('11:30:00')
                dates = pd.date_range(start=start, end=end, freq='500000U').time
                df3 = df3.set_index('date').reindex(dates).reset_index().reindex(columns=df3.columns)
                cols = df3.columns
                df3[cols] = df3[cols].ffill()
                df3 = df3.fillna(method='bfill')
                df_list.append(df3)

            df4 = df_split[(df_split["updatetime"] >= '13:30:00') & (df_split["updatetime"] <= '15:00:00')]
            if len(df4) > 5:
                start = pd.to_datetime('13:30:00')
                end = pd.to_datetime('15:00:00')
                dates = pd.date_range(start=start, end=end, freq='500000U').time
                df4 = df4.set_index('date').reindex(dates).reset_index().reindex(columns=df4.columns)
                cols = df4.columns
                df4[cols] = df4[cols].ffill()
                df4 = df4.fillna(method='bfill')
                df_list.append(df4)

            df51 = df_split[(df_split["updatetime"] >= '21:00:00') & (df_split["updatetime"] < '23:00:00')]
            if len(df51) > 5:
                start = pd.to_datetime('21:00:00')
                end = pd.to_datetime('22:59:59.500000')
                dates = pd.date_range(start=start, end=end, freq='500000U').time
                df51 = df51.set_index('date').reindex(dates).reset_index().reindex(columns=df51.columns)
                cols = df51.columns
                df51[cols] = df51[cols].ffill()
                df51 = df51.fillna(method='bfill')
                if not head.isupper(): #如果合约不是郑商所的
                    df51['tradingday'] = prevday  #夜盘日期往前调一天
                df_list.append(df51)

            df52 = df_split[(df_split["updatetime"] >= '23:00:00') & (df_split["updatetime"] < '23:30:00')]
            if len(df52) > 5:
                start = pd.to_datetime('23:00:00')
                end = pd.to_datetime('23:29:59.500000')
                dates = pd.date_range(start=start, end=end, freq='500000U').time
                df52 = df52.set_index('date').reindex(dates).reset_index().reindex(columns=df52.columns)
                cols = df52.columns
                df52[cols] = df52[cols].ffill()
                df52 = df52.fillna(method='bfill')
                if not head.isupper():
                    df52['tradingday'] = prevday
                df_list.append(df52)

            df53 = df_split[(df_split["updatetime"] >= '23:30:00') & (df_split["updatetime"] < '24:00:00')]
            if len(df53) > 5:
                start = pd.to_datetime('23:00:00')
                end = pd.to_datetime('23:59:59.500000')
                dates = pd.date_range(start=start, end=end, freq='500000U').time
                df53 = df53.set_index('date').reindex(dates).reset_index().reindex(columns=df53.columns)
                cols = df53.columns
                df53[cols] = df53[cols].ffill()
                df53 = df53.fillna(method='bfill')
                if not head.isupper():
                    df53['tradingday'] = prevday
                df_list.append(df53)

    if len(df_list)>0:
        df_new = pd.concat(df_list) #将填充完毕的数据concat
        df_nan = nan_checker(df_new)
        print(df_nan)  # 检查缺失值
        df_new['time'] = df_new['tradingday'].astype(str) + (' ') + df_new['date'].astype(str)
        #df_new['time'] = pd.to_datetime(df_new['time'], format='%Y-%m-%d %H:%M:%S.%f')
        #df_new.drop(df_new.columns[-5:-1], axis=1, inplace=True)
        df_new.sort_values(by='time', inplace=True, ascending=True) #根据时间戳进行sort
        df_new = df_new.reset_index(drop=True)
    elif len(df_list)==0:
        print('合约为非主力合约')
        df_new = pd.DataFrame()
    print(df_new.head(5))
    return df_new