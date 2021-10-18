import sys
import tushare as ts
import pandas as pd
sys.path.append("..")
from utils import config
import datetime

# print(sys.path)
# noinspection PyBroadException
class Watchlist:
    # class of loved stock
    def __init__(self):
        # df: a DataFrame for save stock short time
        self.df = self.read_saved_stock()
        pass

    def insert_stock(self, stock_name: str, history_stock: bool = False, start_time = "",
                     end_time = "", replace:bool = True):
        # to insert a stock into df in a short time
        ts.set_token(config.Tushare_Tocken)
        pro = ts.pro_api()
        date_now = datetime.datetime.now()

        # if end_time == "":
        #     end_time = date_now
        if not history_stock:
            # start_time = (date_now + datetime.timedelta(days=-1)).strftime('%Y%m%d')
            # end_time = date_now.strftime('%Y%m%d')
            start_time = "20210520"
            end_time = "20210521"
            if replace:
                self.df = pro.daily(ts_code = stock_name, start_date = start_time, end_date = end_time)
            else:
                pass    # 等会写，写入新股数据不覆盖
        else:
            self.df = pro.daily(ts_code = stock_name, start_date = start_time, end_date = end_time)


    def read_saved_stock(self, save_path = "loved_stock.csv"):
        # to load the long time saved stock to the class
        try:
            df = pd.read_csv(save_path)
        except:
            df = None
        return df

    def drop_short_time_save(self):
        # 清除暂时存在类中的股票数据
        self.df = None

    def save_stock(self, save_path = "loved_stock.csv"):
        # 覆盖式存放喜爱的股票
        self.df.to_csv("loved_stock.csv")

    def predict(self):
        # 用于对当前股票的当前日期下做出预测
        pass




if __name__ == '__main__':
    w = Watchlist()
    w.insert_stock("600000.SH")
    w.save_stock()
    print(w.df)
    # tushare数据更新的时效性
    pass
