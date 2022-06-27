import collections
import time
import os
import glob
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv


def data_aggregation_1s(filename, time_interval):
    """
        filename为数据路径，time_interval为时间间隔，单位为1s
        适用于整秒级的数据处理
        输入的数据需包含price,ts数据，如没有volume或有其他的数据，需在后面cnames增减
    """
    # cnames = ['price', 'ts', 'volume', 'direction']
    df = pd.read_csv(filename, header=0)
    ts = df['time'].values
    time = []
    for i in ts:
        t = datetime.datetime.fromtimestamp(i // 1000.0).isoformat()
        time.append(t)
    df['time'] = time
    df['tradingday'], df['updatetime'] = df['time'].str.split('[ T]').str
    df = df.drop(columns=['time'], axis=1)
    df = df.dropna()

    grouped = df.groupby(['updatetime'])
    df = grouped.agg({'ts': 'last', 'price': 'last', 'volume': 'last', \
                         'tradingday': 'last', 'updatetime': 'last', \
                         'direction': 'last'})
    # print(df)
    df.reset_index(drop=True, inplace=True)
    df['date'] = pd.to_datetime(df['updatetime'], format='%H:%M:%S').dt.time
    start = df['updatetime'][0]
    # start = next_minute(start, int(time_interval[:-1]))
    end = df['updatetime'].iloc[-1]
    dates = pd.date_range(start=start, end=end, freq=time_interval).time
    dfnew = df.set_index('date').reindex(dates).reset_index().reindex(columns=df.columns)
    dfnew = dfnew.drop(columns=['updatetime'], axis=1)
    # dfnew = dfnew.drop(index=[0]).reset_index()
    cols = dfnew.columns
    dfnew[cols] = dfnew[cols].ffill()
    print(dfnew)
    return dfnew