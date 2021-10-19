import pandas as pd
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

df = pd.read_csv(r"../common/new_data_file/data.csv")
# state = [df.loc[0, ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open", "close", "high", "low", "volume"]]]
# rw = df.rolling(window = 20)
# print(len(df)) # 2284
# #
state_list = ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open", "close", "high", "low", "volume"]
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
rw = df.loc[10:20,state_list]
print(rw)
l1 = [0]*9 + [10,20]
rw.loc[21,:] = l1
print(l1)
print(rw)