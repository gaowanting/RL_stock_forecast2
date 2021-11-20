from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def MDPP(D,P,df):
    df['landmark'] = '-'
    df.loc[(df.close.diff(1) > 0) & (df.close.diff(-1) > 0) & (df.landmark != 1), 'landmark'] =  '^'
    df.loc[(df.close.diff(1) < 0) & (df.close.diff(-1) < 0) & (df.landmark != 1), 'landmark'] =  'v'
    d = df[df['landmark'].isin(['^', 'v'])]
    for i in range(0,len(d)-2,2):
        P_ = abs(d.loc[d.index[i+1],'close'] - d.loc[d.index[i],'close']) /((d.loc[d.index[i+1],'close'] + d.loc[d.index[i],'close'])/2)
        if (d.index[i+1] - d.index[i] < D) and (P_<P):
            df.loc[d.index[i+1],'landmark'] = '-'
            df.loc[d.index[i],'landmark'] = '-'
    df.to_csv('mdpp_test.csv')
    return df

if __name__ == "__main__":
    df = pd.read_csv(r"./common/new_data_file/data.csv")  # 2284
    plt.plot(df.index, df.close)
    df = MDPP(10,0.03,df)
    d = df[df['landmark'].isin(["V","^"])]
    plt.plot(d.index, d.close)
    plt.show()
