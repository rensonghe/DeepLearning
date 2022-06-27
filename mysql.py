import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

conn = MySQLdb.connect(host="rm-bp1yvd1jr36kmmv834o.mysql.rds.aliyuncs.com", user="root", passwd="HZLJDcl123456",
                       db="datainfo", port=3306, charset='utf8')
cursor = conn.cursor()
sql04 = "select date,contract,close,open,high,low from day_price where date >='20150327' and contract = 'ni1507' order by date"
cursor.execute(sql04)
all_data = np.array(cursor.fetchall())
all_data = pd.DataFrame(all_data)
# all_date = pd.DataFrame({'date': all_date[:, 0]})

plt.figure(figsize=(16,8), dpi=72)
plt.plot(all_data[0], all_data[3])
plt.legend(loc=0, frameon=True)
plt.ylabel('Price')
plt.show()