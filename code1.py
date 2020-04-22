# 从yahoo上面爬取tesla的股票信息
# 画价格走势,还有成交量

import datetime as dt 
import matplotlib.pyplot as plt 
from matplotlib import style,use
import pandas as pd 
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2015,1,1)
end = dt.datetime.now()

#print("The data starts from {} ends {}".format(start,end))

df =  web.DataReader("TSLA",'yahoo',start,end)
df.reset_index(inplace = True)
df.set_index("Date",inplace = True)

# 存入csv文件
df.to_csv('TSLA.csv')

df = pd.read_csv('TSLA.csv',parse_dates = True, index_col = 0)

#我们求一个MA moving average
df['100ma'] = df['Adj Close'].rolling(window = 100).mean()
    
#构建subplot 上面是价格, 下面是成交量
#subplot2grid 是用来作出分割的plot的图的
ax1 = plt.subplot2grid((6,1),(0,0),rowspan = 5, colspan =1)
ax2 = plt.subplot2grid((6,1),(5,0),rowspan = 1, colspan =1, sharex = ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()