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