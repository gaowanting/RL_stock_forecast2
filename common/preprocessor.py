import os
import pandas as pd
from datetime import datetime
from stockstats import StockDataFrame as Sdf
import matplotlib.pyplot as plt
from MDPPplus import MDPP

class Preprocessor:

    def __init__(self,raw_data) -> None:
        self.raw_data = pd.read_csv(raw_data,index_col=0)
        self.data_dir = "./common/new_data_file"
        self.tech_indicator_list = ['kdjk', 'kdjd', 'kdjj', "rsi_6", "rsi_12", "rsi_24"]


    def create_data_dir(self):
        """创建存储数据的文件夹"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print("{} 文件夹创建成功!".format(self.data_dir))
        else:
            print("{} 文件夹已存在!".format(self.data_dir))

    def data_cleaning(self):
        data_df = self.raw_data
        data_df = data_df.set_index("trade_date", drop=True)  # 将 trade_date 列设为索引
        data_df = data_df.reset_index()
        
        # 删除一些列并更改列名
        data_df = data_df.drop(["pre_close", "change", "pct_chg", "amount"], axis=1)
        data_df.columns = ["date", "tic", "open", "high", "low", "close", "volume"]

        # 更改 date 列数据格式, 添加 day 列(星期一为 0), 再将格式改回成 str
        data_df[["date"]] = data_df[["date"]].astype(str)
        data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
        data_df["day"] = data_df["date"].dt.dayofweek
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        print(data_df.head())

        # 画出收盘价走势 
        plt.figure(figsize=(20, 8))
        plt.plot(data_df["date"], data_df["close"])
        plt.show()

        # 删除为空的数据行
        data_df = data_df.dropna()
        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)
        return data_df


    def add_technical_indicator(self,data_df):
        """对数据添加技术指标"""
        df = data_df
        df = df.sort_values(by=['tic', 'date'])
        # 获取 Sdf 的对象
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()  # stock.tic是一个pandas.Series

        # 添加技术指标
        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            for ticker in unique_ticker:
                tmp_df = pd.DataFrame(stock[stock.tic == ticker][indicator])
                tmp_df['tic'] = ticker
                tmp_df['date'] = df[df.tic == ticker]['date'].to_list()
                indicator_df = indicator_df.append(tmp_df, ignore_index = True)
            df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
        indicator_df = df.sort_values(by=['date', 'tic'])
        return indicator_df


    def feature_engineer(self,indicator_df):
        processed_df = indicator_df
        processed_df['amount'] = processed_df.volume * processed_df.close
        processed_df['change'] = (processed_df.close - processed_df.open) / processed_df.close
        processed_df['daily_variance'] = (processed_df.high - processed_df.low) / processed_df.close
        processed_df = processed_df.fillna(0)

        print("技术指标列表: ")
        print(self.tech_indicator_list)
        print("技术指标数: {}个".format(len(self.tech_indicator_list)))
        print(processed_df.head())

        processed_df.to_csv(os.path.join(self.data_dir, "data.csv"), index=False)
        print("DataFrame 的大小: ", processed_df.shape)
        return processed_df

    def landmark(self,processed_df):
        d = processed_df
        mdpp_df = MDPP(10,0.03,d)
        mdpp_df.to_csv('./common/new_data_file/data.csv')
        return mdpp_df

    def data_split(self,mdpp_df):
            """将数据分为训练数据集和交易数据集"""
            df = mdpp_df
            split_point = int(len(df)*0.8)
            train_data = df[-split_point:]
            train_data = train_data.sort_values(['date', 'tic'], ignore_index=True)
            train_data.index = train_data.date.factorize()[0]
            train_data.to_csv(os.path.join(self.data_dir, "train.csv"), index=False)

            trade_data = df[:-split_point]
            trade_data = trade_data.sort_values(['date', 'tic'], ignore_index=True)
            trade_data.index = trade_data.date.factorize()[0]
            trade_data.to_csv(os.path.join(self.data_dir, "trade.csv"), index=False)
            return train_data, train_data


if __name__ == "__main__":
    pre = Preprocessor(raw_data = "./common/raw_data.csv")
    pre.create_data_dir()
    data_df = pre.data_cleaning()
    indicator_df = pre.add_technical_indicator(data_df)
    processed_df = pre.feature_engineer(indicator_df)
    mdpp_df = pre.landmark(processed_df)
    pre.data_split(mdpp_df)


