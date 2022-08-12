import os
import torch
import util.preprocessor
import datetime
import tushare as ts

def watchlist_predict(self, loved_stock = None):
    # 搭建并加载最新的网络
    for a, b, c in os.walk('DQN/outputs/StockLearningEnv'):
        # print(b)
        li = b
        # print(os.getcwd())
        break

    network = torch.load('DQN/outputs/StockLearningEnv/' + li[-1] + '/models/full_dqn_model.pth')
    # 获取数据并预处理
    if loved_stock is None:
        return "please select your loved stock"
    tushare_tocken = "c576df5b626df4f37c30bae84520d70c7945a394d7ee274ef2685444"
    ts.set_token(tushare_tocken)
    start_time = (datetime.datetime.now()-datetime.timedelta(days=50)).strftime("%Y-%m-%d") # 目前模型的windows_size是20天，索性获取50天，避免非交易日的尴尬
    end_time = datetime.datetime.now().strftime('%Y-%m-%d')
    data_tmp = ts.pro_bar(ts_code=loved_stock, adj='qfq', freq="d",
                        start_date=start_time, end_date=end_time)
    
    if data_tmp is None:
        return "please use formed or correct stock_code, example:'002364.SZ'"

    pre = util.preprocessor.Preprocessor(df=data_tmp)
    data = pre.process()
    
    # 环比部分采用自比的方法，详情见env2环比处理
    norm_list = ["open", "close", "high", "low", "volume"]
    after_norm = ["open_", "close_", "high_", "low_", "volume_"]
    data[after_norm] = data[norm_list].pct_change(-1)
    data = data.dropna()
    state = data.loc[range(data.__len__()-20, data.__len__()), ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open_", "close_", "high_", "low_", "volume_"]]
    state = torch.Tensor(state.values).flatten()
    
    # 预测
    q_values = network(state)
    print(q_values)
    print(q_values.shape)
    action = q_values.tolist().index(q_values.max())
    return action - 1