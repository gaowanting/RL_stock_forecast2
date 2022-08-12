import sys
sys.path.append('./VAE_stock')

import os
import pandas as pd
from datetime import datetime
from stockstats import StockDataFrame as Sdf
from MDPPplus import MDPP
from sklearn import preprocessing
from vae import VAE
from torch.autograd import Variable
import torch
import sqlite3


class Preprocessor:

    def __init__(self) -> None:
        # 从数据库中读出数据
        con = sqlite3.connect(os.path.join(os.getcwd(), 'data.db'))
        self.raw_data = pd.read_sql('SELECT * FROM raw_data',con,index_col='index')
        # 从csv中读出数据
        # self.raw_data = pd.read_csv(raw_data,index_col=0)
        # self.data_dir = "./util/new_data_file"
        self.tech_indicator_list = ['kdjk', 'kdjd', 'kdjj', "rsi_6", "rsi_12", "rsi_24"]
        self.normalization = 'standardization' # 'div_close'/ 'standardization'
        self.windowsize = 20

    # def create_data_dir(self):
    #     """创建存储数据的文件夹"""
    #     if not os.path.exists(self.data_dir):
    #         os.makedirs(self.data_dir)
    #         print("{} 文件夹创建成功!".format(self.data_dir))
    #     else:
    #         print("{} 文件夹已存在!".format(self.data_dir))

    def data_cleaning(self):
        data_df = self.raw_data
        data_df = data_df.set_index("trade_date", drop=True)  # 将 trade_date 列设为索引
        data_df = data_df.reset_index()
        
        # 删除一些列并更改列名
        data_df = data_df.drop(["pre_close", "change", "pct_chg", "amount"], axis=1)
        data_df.columns = ["date", "tic", "open", "high", "low", "close", "volume"]

        # 更改 date 列数据格式
        data_df[["date"]] = data_df[["date"]].astype(str)
        data_df["date"] = data_df.date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))

        # 删除为空的数据行
        data_df = data_df.dropna()
        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)

        # 添加 day 列,依次递增
        data_df['day'] = range(len(data_df))
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
        print("DataFrame 的大小: ", processed_df.shape)
        return processed_df
    
    def do_normalization(self,processed_df):
        df = processed_df
        norm_list = ["open", "close", "high", "low", "volume"]
        after_norm = ["open_", "close_", "high_", "low_", "volume_"]
        formater = '{0:.04f}'.format
        if self.normalization == 'div_self':
            # 将ochlv处理为涨跌幅
            df[after_norm] = df[norm_list].pct_change(-1).applymap(formater)
            df = df.dropna()
        elif self.normalization == 'div_close':
            # 将ochl处理为相较于前一天close的比例
            temp = df[["open", "close", "high", "low"]].values[1:] / df[["close"]].values[:-1]
            df = df[1:]
            # volume 单独处理
            df[["volume_"]] = df[["volume"]].pct_change(-1)
            df[["open_", "close_", "high_", "low_"]] = temp
        elif self.normalization == 'standardization':
            # do standardization in a sliding window
            norm_df = df.copy()
            for i in range(self.windowsize,len(df) - self.windowsize):
                temp = df.loc[df.index[i]:df.index[i + self.windowsize - 1], norm_list]
                scaler = preprocessing.StandardScaler().fit(temp)
                norm_df.loc[norm_df.index[i]:norm_df.index[i + self.windowsize - 1], after_norm] = scaler.transform(temp)
            norm_df = norm_df.dropna()
            df = norm_df.copy()
        df = df.round(4)
        return df

    def landmark(self,norm_df):
        d = norm_df
        mdpp_df = MDPP(10,0.03,d)
        return mdpp_df
    
    def embedding(self,mdpp_df):
        final_df = mdpp_df
        embedding = []
        model = VAE()
        if torch.cuda.is_available(): model.cuda()
        model.load_state_dict(torch.load(r'./VAE_STOCK/vae_stock1.pth'))
        for i in range(len(final_df)-self.windowsize):
            temp = final_df.iloc[i:20+i,-6:-1].to_numpy()
            data = torch.tensor(temp,dtype=torch.float32)  # torch.Size([20, 5])
            data = data.reshape(1,100)
            data = Variable(data)
            mu, logvar = model.encode(data)
            z = model.reparametrize(mu, logvar).data.numpy()
            # breakpoint()
            embedding.append(z)
        final_df = final_df.iloc[0:len(final_df)-self.windowsize,:]
        final_df['embedding'] = embedding
        
        return final_df


if __name__ == "__main__":
    # input 
    pre = Preprocessor()
    # pre.create_data_dir()
    # preprocess
    data_df = pre.data_cleaning()
    indicator_df = pre.add_technical_indicator(data_df)
    processed_df = pre.feature_engineer(indicator_df)
    norm_df = pre.do_normalization(processed_df)
    mdpp_df = pre.landmark(norm_df)
    # reset_index
    mdpp_df = mdpp_df.sort_values(['date', 'tic'], ignore_index=True)
    mdpp_df.index = mdpp_df.date.factorize()[0]
    # vae
    final_df = pre.embedding(mdpp_df)
    con = sqlite3.connect(os.path.join(os.getcwd(), 'data.db'))
    final_df.to_sql('train_data',con,if_exists='replace')

    # output
    # mdpp_df.to_csv('./util/new_data_file/data.csv',index=None)
    # final_df.to_csv('./util/new_data_file/data_embedding.csv',index=None)
# 2025