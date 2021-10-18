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
state = [df.loc[0, ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open", "close", "high", "low", "volume"]]]
print(state)