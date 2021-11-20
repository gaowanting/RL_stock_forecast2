# import pandas as pd
# df = pd.read_csv(r"../common/new_data_file/data.csv")


# s = pd.Series(range(5))
# window = s.rolling(window=2)
# for i in window:
#     print(i)


# df2 = df[['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open", "close", "high", "low", "volume"]]
# rw = df2.rolling(window = 20)
#
# # type of i is pd.DataFrame, can state run on it?
# for i in rw:
#     print(i)
# df3 = df[["open", "close", "high", "low", "volume"]]
# print(df3.head())
# df3 = df3.pct_change(-1)+1
# print(df3.head())


# state = [df.loc[0, ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open", "close", "high", "low", "volume"]]]
# rw = df.rolling(window = 20)
# print(len(df)) # 2284
# #
# state_list = ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open", "close", "high", "low", "volume"]
# state = df.loc[:,]
# rw = df.loc[1:20,['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open", "close", "high", "low", "volume"]]
# print(rw)
# rw.append([0]*len(state_list))
# print("---rw---")
# print(rw.iloc[-1,:])
#
# # df['rw'] = rw
#
# print(df.head())
# rw = df.loc[10:20,state_list]
# print(rw)
# l1 = [0]*9 + [10,20]
# rw.loc[21,:] = l1
# print(l1)
# print(rw)

# a = [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0., 0.],
#        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#         0., 0., 0.,3.]]
# print(a[-1][-1])

# rolling_window = []
# temp_df = self.df.loc[:, self.state_list]
# for i in temp_df.rolling(window = self.window_size):
#     rolling_window.append(dict(i))
#     print("dict(i)")
#     print(dict(i))
# rolling_window = pd.Series(rolling_window)
# self.df['rolling_window'] = rolling_window
# print(self.df.head())

# print(1e+6)

# import torch
# tensor_0 = torch.arange(3,12).view(3,3)
# print(tensor_0[0][1])
# print(tensor_0[0,1])
# '''
# tensor([[ 3,  4,  5],
#         [ 6,  7,  8],
#         [ 9, 10, 11]])
# '''
# index = torch.tensor([[2,1,0]])  # torch.Size([1, 3])
# tensor_1 = tensor_0.gather(0,index)
# print(tensor_1)
# tensor([[9, 7, 5]])

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
#
# r = np.random.rand(10,5)
# scaler = preprocessing.StandardScaler().fit(r)
# r_ = scaler.transform(r)
# # plt.subplots()
# # plt.plot(r)
# plt.plot(r_)
# plt.show()
# print(r)
# print(r_)

# import torch
# # matrix = torch.tensor([[2,3,4],[5,6,7]])
# matrix = torch.tensor([[[1117.5198, -302.7872,  583.7730]]])
# print(matrix.shape)
# print(matrix[0][0])
# print(matrix[0][0].max())
#
#
# print('1')
# breakpoint()
# print('2')
# a = 7
# print('3')
# breakpoint()
# print('111')


# print(q_values.max(1)[1])
# action_probs = F.softmax(q_values/self.REWARD_SCALE, -1)
# action_dist = Categorical(action_probs)
# action = action_dist.sample().view(-1, 1)

# q_values : tensor([[[1117.5198, -302.7872,  583.7730]]])

# action = action_probs.max(1)[1].item()
# size = 100
# x = np.linspace(1,size,size)
# data = np.random.randint(1,size,size)

# plt.plot(x,data)
# y = savgol_filter(data,5,3,mode = 'nearest')
# plt.plot(x,y,'b',label = 'savgol')
# plt.show()

# def variation(Y,gap):    
#     li = []    
#     for i in range(gap,len(Y)-gap): 
#         li.append(np.var(Y[i-gap:i])/np.var(Y[i:i+gap]))
#     return li
# data = np.random.randint(1,100,100)
# li = variation(data,5)
# plt.plot(range(100),data)
# plt.plot(range(10,100),li)
# plt.show()

# def MDPP2(D,P,picked):
#     picked2 = []
#     for i in range(len(picked)-D):
#         x = picked[i][0]
#         x_ = picked[i+D][0]
#         y = picked[i][1]
#         y_ = picked[i+D][1]
#         A = abs(y_ - y)
#         B = (y_ + y)/2
#         if (A/B)>P:
#             picked2.append((x,y))
#     return picked2
# picked2 = MDPP2(1,0.05,picked)


# from functools import reduce
# import numpy as np
# import matplotlib.pyplot as plt
# from pandas.core.base import DataError
# from scipy.signal import savgol_filter
# import pandas as pd


# df = pd.read_csv(r"./common/new_data_file/data.csv")  # 2284

# def MDPP(D,P,df):
#     # original MDPP algorithm
#     picked = []
#     for i in range(len(df)-D):
#         x = df.index[i] 
#         x_ = x + D
#         A = abs(df.iloc[x_,:].close - df.iloc[x,:].close)
#         B = (df.iloc[x_,:].close + df.iloc[x,:].close)/2
#         if (A/B)>P:
#             picked.append((x,df.iloc[x,:].close))
#     # remove dense points
#     sub = []
#     flag = 0
#     for i in range(len(picked)-2):
#         if (picked[i+1][0] - picked[i][0]) != 1:
#             sub.append(picked[flag:i+1])
#             sub.append(picked[i+1])
#             flag = i+2
#     final_picked = []
#     for i in sub:
#         if type(i) == tuple:
#             final_picked.append(i)
#         elif type(i) == list:
#             for e in i:
#                 max = i[0]
#                 if e[1] > max[1]:
#                     max = e
#             final_picked.append(e)
#     func = lambda x,y:x if y in x else x + [y]
#     final_picked = reduce(func,[[], ] + final_picked)
#     return final_picked


# picked = MDPP(10,0.09,df)


# x = []
# y = []
# for i in picked:
#     x.append(i[0])
#     y.append(i[1])

# # x_ = []
# # y_ = []
# # for i in final_picked:
# #     x_.append(i[0])
# #     y_.append(i[1])

# plt.plot(df.index, df.close)
# plt.plot(x,y)
# # plt.plot(x_,y_)
# plt.show()

# from numpy.lib import npyio
# import numpy as np
# import torch
# import pickle

# list = [torch.tensor([1,2]),torch.tensor([2,3])]
# ValueError: only one element tensors can be converted to Python scalars
    # a = np.array(list)
    # np.save("numpy.npy",a)
    # b = np.load('numpy.npy')
    # print(type(b[0]))
# pickle
# with open('datafile.pkl', 'wb') as f:
#    pickle.dump(list, f)
# pickle_off = open ('datafile.pkl', "rb")
# emp = pickle.load(pickle_off)
# print(emp)

# class A():
#     def __init__(self,attr3) -> None:
#         self.attr1 = 1,
#         self.attr2 = 2,
#         self.attr3 = attr3

#     @property
#     def reward(self):
#         return 10
    
#     def state(self):
#         a = [[1,2],[3,4]]
#         return a


# if __name__ == "__main__":
#     a = A('str')
#     print(a.reward)
#     print(a.state())
#     print(a.attr3)


import pandas as pd
from datetime import datetime

df = pd.read_csv('./common/raw_data.csv')
df["date"].dtypes()
df["date"] = df.trade_date.apply(lambda x: datetime.strptime(x[:4] + '-' + x[4:6] + '-' + x[6:], "%Y-%m-%d"))
print(df.head())
# print(type(df.trade_date.values[:1]))
