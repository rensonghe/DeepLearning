import pandas as pd
import numpy as np
#%%
data = pd.read_csv('data_ni_2019.csv')
#%%
data = data.iloc[:,1:]
#%%
import scipy.stats as st

def factor_IC_analysis(factorData, Field, begin_date, end_date, rule='normal'):
    dateList = get_period_date('M', begin_date, end_date)
    IC = {}
    R_T = pd.DataFrame()
    for date in dateList[:-1]:
        #取股票池
        stockList = list(factorData[date].index)
        #获取横截面收益率
        df_close=get_price(stockList, date, dateList[dateList.index(date)+1], 'daily', ['close'])
        if df_close.empty:
            continue
        df_pchg=df_close['close'].iloc[-1,:]/df_close['close'].iloc[0,:]-1
        R_T['pchg']=df_pchg
        #获取因子数据
        factor_data = factorData[date][Field]
        #数据标准化
        factor_data = standardlize(factor_data, inf2nan=True, axis=0)
        R_T['factor'] = factor_data
        R_T = R_T.dropna()
        if rule=='normal':
            IC[date]=st.pearsonr(R_T.pchg, R_T['factor'])[0]
        elif rule=='rank':
            IC[date]=st.pearsonr(R_T.pchg.rank(), R_T['factor'].rank())[0]
    IC = pd.Series(IC).dropna()
    print 'IC 值序列的均值大小',IC.mean()
    print 'IC值序列绝对值的均值大小',np.mean(np.abs(IC))
    print 'IC 值序列的标准差',IC.std()
    print 'IR 比率（IC值序列均值与标准差的比值）',IC.mean()/IC.std()
    n = [x for x in IC.values if x>0]
    print 'IC 值序列大于零的占比',len(n)/float(len(IC))
factor_IC_analysis(factorData, 'Neg_pos_ID', begin_date, end_date)