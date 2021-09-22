import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sklearn.metrics as metrics
from landmark import landmark


class Evaluation():
    def __init__(self, df):
        self.df = df

    @staticmethod
    def plot_trade_information():
        """
        plot the reward, close, total_assets, respectively
        """
        # 这里面要用到子图
        x = df['date']
        y1 = df["close"]
        y2 = df['reward']
        y3 = df['total_assets']
        y4 = df["assets_baseline"]
        y5 = df["regret"]
        plt.figure(figsize=(20, 8))

        ax1 = plt.subplot(311)
        ax1.set_title("trade_information")
        # plt.plot(x, y1, label='close price', color="g") 我觉得收盘价没必要画出来了，因为assets_baseline就能体现
        plt.plot(x, y5, label='regret', color="g")

        plt.legend(loc='upper left')
        ax1.axes.xaxis.set_visible(False)

        ax2 = plt.subplot(312, sharex=ax1)
        plt.plot(x, y2, label='reward', color="b")
        plt.legend(loc='upper left')
        ax2.axes.xaxis.set_visible(False)

        ax3 = plt.subplot(313, sharex=ax1)
        plt.plot(x, y3, label='total_assets', color="g")

        plt.plot(x, y4, label='assets_baseline', color="b")

        plt.legend(loc='upper left')
        ax2.axes.xaxis.set_visible(False)

        # ax4 = plt.subplot(414, sharex=ax1)
        # plt.plot(x, y4, label="assets_baseline")
        # plt.legend(loc="upper left")

        plt.xticks(range(0, len(df['date'])), df['date'], rotation=45, ha='right', size=8)
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))

        plt.savefig('./trade_information_episode1.jpg')
        plt.show()


    # def landmark(self, MDPP_D=5, MDPP_P=0.05):
    #     """
    #     find the right trading point
    #     """
    #     d = df.copy()
    #     d.close = d.close.astype("float")
    #
    #     # find the local max & local min
    #     d['landmark'] = '-'
    #     d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) < 0), 'landmark'] = '/'
    #     d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) > 0), 'landmark'] = '\\'
    #     d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) > 0), 'landmark'] = '^'
    #     d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) < 0), 'landmark'] = 'v'
    #
    #     # modify the df
    #     df.loc[d[d.landmark == '/'].index, 'landmark'] = '/'
    #     df.loc[d[d.landmark == '\\'].index, 'landmark'] = '\\'
    #     df.loc[d[d.landmark == '^'].index, 'landmark'] = '^'
    #     df.loc[d[d.landmark == 'v'].index, 'landmark'] = 'v'
    #
    #     df['landmark2'] = '-'
    #     df.loc[d[d.landmark == '/'].index, 'landmark2'] = 'v'
    #     df.loc[d[d.landmark == '\\'].index, 'landmark2'] = '^'
    #     df.loc[d[d.landmark == '^'].index, 'landmark2'] = '^'
    #     df.loc[d[d.landmark == 'v'].index, 'landmark2'] = 'v'
    #     df.to_csv("train_action1.csv", index=False)
    #
    #     d = d[d.landmark.isin(['^', 'v'])]
    #
    #     for _ in range(MDPP_D):
    #         # 找出相邻点满足变化<P的数据行
    #         d = d[~(2 * abs(d.close.diff(1)) / (abs(d.close) + abs(d.close.shift(1))) < MDPP_P)]
    #
    #     df2 = df[df.landmark.isin(['^', 'v'])]
    #     fig, ax = plt.subplots()
    #     ax.plot(df.index, df.close, label='before')  # Plot some data on the axes.
    #     ax.plot(df2.index, df2.close, label='after')  # Plot more data on the axes...
    #     ax.set_xlabel('day')
    #     ax.set_ylabel('close')
    #     ax.legend()
    #     # plt.savefig('./landmark_episode1.jpg')
    #     plt.show()
    #     return df

    def plot_signal(self):
        df.actions = [float(i[1:len(i) - 1]) for i in df.actions]
        df["actions_signal"] = '-'
        df.loc[df[df.actions > 0].index, 'actions_signal'] = '^'
        df.loc[df[df.actions < 0].index, 'actions_signal'] = 'v'
        df.to_csv("for_debug.csv", index=False)

        df_actions_sign = np.sign(df["actions"])
        buying_signal = df_actions_sign.apply(lambda x: True if x > 0 else False)
        selling_signal = df_actions_sign.apply(lambda x: True if x < 0 else False)

        tic_plot = df['close']
        tic_plot.index = df.index

        plt.figure(figsize=(20, 8))
        plt.plot(tic_plot, color='g', lw=2.)
        plt.plot(tic_plot, '^', markersize=3, color='m', label='buying signal', markevery=buying_signal)
        plt.plot(tic_plot, 'v', markersize=3, color='k', label='selling signal', markevery=selling_signal)
        plt.title('actions signal')
        plt.legend()
        plt.xticks(range(0, len(df['date'])), df['date'], rotation=45, ha='right')
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(25))
        plt.savefig('./actions_signal.jpg')

    def calculate_metrics(self):
        landmark(df)   # landmark.py
        # 基于landmark后的 best_action_reward 计算regret
        self.regret()
        self.plot_signal()
        accuracy_score = metrics.accuracy_score(df["landmark2"], df["actions_signal"], normalize=True,
                                                sample_weight=None)
        precision_score = metrics.precision_score(df["landmark2"], df["actions_signal"], average='weighted',
                                                  labels=np.unique(df["actions_signal"]))
        f1_score = metrics.f1_score(df["landmark2"], df["actions_signal"], average='weighted')
        recall_score = metrics.recall_score(df["landmark2"], df["actions_signal"], average='weighted')
        result = {"accuracy_score": accuracy_score,
                  "precision_score": precision_score,
                  "f1_score": f1_score,
                  "recall_score": recall_score}
        print(result)
        return result

    def regret(self):
        self.df["regret"] = self.df["total_assets"]/self.df["best_reward"]


if __name__ == "__main__":
    df = pd.read_csv("new_train_file\\train_action15.csv")
    evaluation = Evaluation(df)

    evaluation.calculate_metrics()
    evaluation.plot_trade_information()
