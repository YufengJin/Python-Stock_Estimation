#encoding:utf8

# 从wiki上下载S&P500 500家公司的信息

import bs4 as bs 
import json
import pickle
import requests
import os
import datetime as dt
import pandas_datareader as web
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from collections import Counter
from sklearn import svm,neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table',{'class':'wikitable sortable'})
    tickers = []
# 对每一个缩写提取 然后存在SP500ticker.pickle的文件中
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.rstrip()
        tickers.append(ticker)

#    with open("sp500tickers.json","w") as f:
#        json.dump(tickers,f,sort_keys=True)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)

    return tickers

# 从yahoo上提取信息
def get_data_from_yahoo(reload_sp500 = False):
    #重新从网上load sp500 公司list
    if reload_sp500:
        tickers = save_sp500_tickers()
    #直接从pickle文件中提取500家公司的缩写
    else:
        with open("sp500tickers.pickle",'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010,1,1)
    end = dt.datetime.now()

    for ticker in tickers:
        try:
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                df = web.DataReader(ticker,'yahoo',start,end)
                df.reset_index(inplace=True)
                df.set_index("Date",inplace=True)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
            else:
                print('{}.csv already exists.'.format(ticker))
        except:
            print('{}.csv failed to generate.'.format(ticker))

#存储500 公司Adj Close数据在.csv文件
def add_moving_average():

    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    for ticker in tickers:
        try:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date',inplace=True)
            df['MA50']=df['Adj Close'].rolling(window=50).mean()
            start = '2015-01-01'
            end = '2020-04-29'
            mean = df['Adj Close'][start:end].mean()

            if df['Adj Close'][end] < df['MA50'][end] and df['Adj Close'][end] < 0.8 * mean:
                data = df['Adj Close'][end]/mean
                print("{}: {}".format(ticker,data))
        except:
            print("{} invalid.".format(ticker))
    

def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            print('{}.csv doesn\'t exist'.format(ticker))
        else:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)

            #将每一个csv文件只留下Adj Close然后将500家公司合起来
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
            #存储500 公司Adj Close数据
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            if count % 10 == 0:
                print(count)
    main_df.to_csv('sp500_joined_closes.csv')


def visualize_data():
    style.use('ggplot')
    df = pd.read_csv('sp500_joined_closes.csv')
    df.reset_index(inplace=True)
    df.set_index('Date',inplace=True)
    del df['index']

    #求出correlation的表格,500家公司之间
    df_corr = df.corr()
    df_corr.to_csv('sp500corr.csv')

    #绘制热图
    data1 = df_corr.values   #生成2维对角数表
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1) #定义plot的color
    #加上旁边的colorbar
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    #y axis 颠倒，x axis 变到上面
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X,y,df

def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    print()
    return confidence


add_moving_average()