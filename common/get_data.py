import tushare as ts
import sys
import time
import random

sys.path.append("..")
data_dir = "new_data_file"

def get_random_data():
    tushare_tocken = "c576df5b626df4f37c30bae84520d70c7945a394d7ee274ef2685444"
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

        data_tmp = ts.pro_bar(ts_code=tic, adj='qfq', freq="d",
                          start_date=selected_start_time, end_date=selected_end_time)
        if (data_tmp is not None) and (len(data_tmp)>2000):
            print(f"the selected stock for training model is: {tic}\nHere is the information")
            print(data[data["ts_code"] == tic])
            break
    return data_tmp

if __name__ == "__main__":
    raw_data = get_random_data()
    raw_data.to_csv('./common/raw_data.csv')