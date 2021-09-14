import pandas as pd
import matplotlib.pyplot as plt


def landmark(df, MDPP_D=5, MDPP_P=0.03):
    d = df.copy()
    d.close = d.close.astype("float")

    for _ in range(MDPP_D):
        # 找出相邻点满足变化<P的数据行
        d = d[~(2 * abs(d.close.diff(1)) / (abs(d.close) + abs(d.close.shift(1))) < MDPP_P)]

    # find the local max & local min
    d['landmark'] = '-'
    d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) < 0), 'landmark'] = '/'
    d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) > 0), 'landmark'] = '\\'
    d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) > 0), 'landmark'] = '^'
    d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) < 0), 'landmark'] = 'v'
    # may be here is not needed, when the close price has not changed, just use "-" is ok ~
    # d.loc[(d.close.diff(1) == 0) & (d.close.diff(-1) < 0), 'landmark'] = '-/'
    # d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) == 0), 'landmark'] = '\\-'
    # d.loc[(d.close.diff(1) == 0) & (d.close.diff(-1) > 0), 'landmark'] = "'-\\"  # 这里直接"-\\"会导致csv中第一个字符输出为"="，所以-前加一个'
    # d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) == 0), 'landmark'] = '/-'
    #
    # # 曲线平滑处理后就基本没后四个了
    # # d.to_csv("d.csv")
    # df['landmark'] = d.landmark

    df["best_reward"] = best_action(df)

    d = d[d.landmark.isin(['^', 'v'])]
    # 为了匹配后面的二分类这里把它这么处理一下
    df['landmark2'] = '-'
    df.loc[d[d.landmark == '/'].index, 'landmark2'] = 'v'
    df.loc[d[d.landmark == '\\'].index, 'landmark2'] = '^'
    df.loc[d[d.landmark == '^'].index, 'landmark2'] = '^'
    df.loc[d[d.landmark == 'v'].index, 'landmark2'] = 'v'

    # modify the df

    # df.loc[d[d.landmark == '^'].index, 'landmark'] = '^'
    # df.loc[d[d.landmark == 'v'].index, 'landmark'] = 'v'
    # df.loc[d[d.landmark == '/'].index, 'landmark'] = '/'
    # df.loc[d[d.landmark == '\\'].index, 'landmark'] = '\\'
    # df.loc[d[d.landmark == '-/'].index, 'landmark'] = '-/'
    # df.loc[d[d.landmark == '\\-'].index, 'landmark'] = '\\-'
    # df.loc[d[d.landmark == "'-\\"].index, 'landmark'] = "'-\\"
    # df.loc[d[d.landmark == '/-'].index, 'landmark'] = '/-'

    # df.to_csv("look.csv") # for_debug

    return df


def best_action(action_df):
    # action_df: landmarked df(from dir new_train_file)
    # choose action based on the landmarke action df. and return a Series of best reward

    assets = action_df["total_assets"][0]  # assets: get Initial funds
    best_action_assestslist = []  # 最好资产列表，最后直接填入df
    hold_tic = 0  # 当前持股
    count = 0
    base_assets = assets  # 持有资金, 其实可以直接把assets 删了

    # debug
    hold_tic_list = []
    # debug
    for i in action_df["landmark"]:
        if i == "v" or i == "/":
            # 上升or触底则买
            zeta_hold_tic = int(base_assets / (action_df["close"][count] * 100)) * 100  # 触底信号在本日收盘时发生，此时全仓买入
            hold_tic += zeta_hold_tic  # 成手买入
            base_assets -= zeta_hold_tic * action_df["close"][count]
            # 考虑开盘价加入？
        if i == "^" or i == "\\":
            # 反之
            zeta_hold_tic = hold_tic  # 全仓卖出
            hold_tic -= zeta_hold_tic
            base_assets += zeta_hold_tic * action_df["close"][count]
        best_action_assestslist.append(hold_tic * action_df["close"][count] + base_assets)
        hold_tic_list.append(hold_tic)
        count += 1

    # pd.DataFrame(hold_tic_list).to_csv("hold_tic.csv")    # 取消注释以输出一个持有股票量的表格
    return pd.Series(best_action_assestslist)


def cul_earnings(df):
    df['earnings'] = abs(df.close.pct_change())
    return df

    # def landmarks_after(datetime, n=6):
    #     '''future n landmarks after the day '''
    #     lm = series[series.landmark.isin(['v', '^'])]  # series来自函数外部
    #     lm_after = lm[lm.time > datetime]  # .copy() # .copy() makes sure not to modify the original data
    #     return lm_after[0:n]  # at the end of stock series, there may be less than 6.
    #
    # series = df  # df without .copy(), the df will be modified
    # _mark(series)  # calculate and mark field 'landmark' with 'L' for low or 'H' for high
    # return landmarks_after  # returns a function for stockmarket to use


if __name__ == "__main__":
    df = pd.read_csv('new_train_file\\train_action30.csv')
    landmark(df)

    # # 原始图
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(df.index, df.close)
    # ax1.set_xlabel('day')
    # ax1.set_ylabel('close')
    #
    # # landmark 标注后的图
    # df2 = df[df.landmark.isin(['^', 'v'])]
    # ax2.plot(df2.index, df2.close)
    # ax2.set_xlabel('day')
    # ax2.set_ylabel('close')
    # print(df2)
    # print(df.head(50))
    # plt.show()

    # two lines in one
    df2 = df[df.landmark.isin(['^', 'v'])]
    fig, ax = plt.subplots()
    ax.plot(df.index, df.close, label='before')  # Plot some data on the axes.
    ax.plot(df2.index, df2.close, label='after')  # Plot more data on the axes...
    ax.set_xlabel('day')
    ax.set_ylabel('close')
    ax.legend()
    plt.show()

    cul_earnings(df2)
