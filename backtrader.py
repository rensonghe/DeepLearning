import matplotlib.figure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('ni1507.csv')
data['sellprice'] = data['sellprice'].fillna(data['sellprice'].median)
data['sellvolume'] = data['sellvolume'].fillna(data['sellvolume'].median)
data['buyprice'] = data['buyprice'].fillna(data['buyprice'].median)
data['buyvolume'] = data['buyvolume'].fillna(data['buyvolume'].median)

plt.figure(figsize=(16,8), dpi=72)
plt.plot(data['time'], data['tradingprice'])
plt.legend(loc=0, frameon=True)
plt.ylabel('Price')
plt.show()