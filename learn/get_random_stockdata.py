import tushare as ts
import random
import time
import sys
import os
import pandas as pd
from datetime import datetime
from stockstats import StockDataFrame as Sdf
from utils import config
import matplotlib.pyplot as plt

sys.path.append("..")
data_dir = "../new_data_file"


def create_data_dir(data_dir = data_dir):
    """创建存储数据的文件夹"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("{} 文件夹创建成功!".format(data_dir))
    else:
        print("{} 文件夹已存在!".format(data_dir))


def get_random_data():
    tushare_tocken = config.Tushare_Tocken
    ts.set_token(tushare_tocken)
    pro = ts.pro_api()

    while True:
        # 在上市公司中随机抽取股票一支股票代码
        data = pro.stock_basic(exchange='', list_status='L', fields='ts_code, name, area,industry,list_date')
        tic = random.choice(data["ts_code"])

        # 从2008年到2020年随机选取10年的数据
        start_time = "2008-01-01"
        end_time = "2010-01-01"
        timestamp_s = time.mktime(time.strptime(start_time, "%Y-%m-%d"))
        timestamp_e = time.mktime(time.strptime(end_time, "%Y-%m-%d"))
        chose_time = random.randint(timestamp_s, timestamp_e)

        selected_start_time = time.strftime("%Y-%m-%d", time.localtime(chose_time))
        selected_end_time = time.strftime("%Y-%m-%d", time.localtime(chose_time + 315532800))

        data_tmp = ts.pro_bar(ts_code=tic, adj='qfq',
                          start_date=selected_start_time, end_date=selected_end_time)
        if (data_tmp is not None) and (len(data_tmp)>2000):
            print(f"the selected stock for training model is: {tic}\nHere is the information")
            print(data[data["ts_code"] == tic])
            break
    return data_tmp


def preprocessor():
    data_df = get_random_data()
    data_df = data_df.set_index("trade_date", drop=True)  # 将 trade_date 列设为索引
    data_df = data_df.reset_index()

    # 删除一些列并更改列名
    data_df = data_df.drop(["pre_close", "change", "pct_chg", "amount"], axis=1)
    data_df.columns = ["date", "tic", "open", "high", "low", "close", "volume"]

    plt.figure(figsize=(20, 8))
    plt.plot(data_df["date"], data_df["close"])
    plt.show()
    # 更改 date 列数据格式, 添加 day 列(星期一为 0), 再将格式改回成 str
    data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
    data_df["day"] = data_df["date"].dt.dayofweek
    data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
    # 删除为空的数据行
    data_df = data_df.dropna()
    data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
    return data_df


def add_technical_indicator():
    """对数据添加技术指标"""
    df = preprocessor()
    df = df.sort_values(by=['tic', 'date'])
    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST
    # 获取 Sdf 的对象
    stock = Sdf.retype(df.copy())
    unique_ticker = stock.tic.unique()  # stock.tic是一个pandas.Series

    # 添加技术指标
    for indicator in tech_indicator_list:
        indicator_df = pd.DataFrame()
        for ticker in unique_ticker:
            tmp_df = pd.DataFrame(stock[stock.tic == ticker][indicator])
            tmp_df['tic'] = ticker
            tmp_df['date'] = df[df.tic == ticker]['date'].to_list()
            indicator_df = indicator_df.append(tmp_df, ignore_index = True)
        df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
    df = df.sort_values(by=['date', 'tic'])
    return df


def feature_engineer():
    processed_df = add_technical_indicator()
    processed_df['amount'] = processed_df.volume * processed_df.close
    processed_df['change'] = (processed_df.close - processed_df.open) / processed_df.close
    processed_df['daily_variance'] = (processed_df.high - processed_df.low) / processed_df.close
    processed_df = processed_df.fillna(0)

    print("技术指标列表: ")
    print(config.TECHNICAL_INDICATORS_LIST)
    print("技术指标数: {}个".format(len(config.TECHNICAL_INDICATORS_LIST)))
    print(processed_df.head())

    processed_df.to_csv(os.path.join(data_dir, "data.csv"), index=False)
    print("DataFrame 的大小: ", processed_df.shape)
    return processed_df


def data_split():
        """将数据分为训练数据集和交易数据集"""
        df = feature_engineer()
        split_point = int(len(df)*0.8)
        train_data = df[-split_point:]
        train_data = train_data.sort_values(['date', 'tic'], ignore_index=True)
        train_data.index = train_data.date.factorize()[0]
        train_data.to_csv(os.path.join(data_dir, "train.csv"), index=False)

        trade_data = df[:-split_point]
        trade_data = trade_data.sort_values(['date', 'tic'], ignore_index=True)
        trade_data.index = trade_data.date.factorize()[0]
        trade_data.to_csv(os.path.join(data_dir, "trade.csv"), index=False)
        return train_data, train_data


if __name__ == "__main__":
    create_data_dir()
    data_split()
