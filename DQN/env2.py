from typing import Any, List, Tuple
import numpy as np
import pandas as pd
import random
import gym
import time
from gym import spaces
from sklearn import preprocessing

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

class StockLearningEnv(gym.Env):

    def render(self, mode="human"):
        pass


    def __init__(
            self,
            df: pd.DataFrame,
            buy_cost_pct: float = 3e-3,
            sell_cost_pct: float = 3e-3,
            print_verbosity: int = 10,
            initial_amount: int = 1e6,
            patient: bool = False,
            currency: str = "￥",
            is_train: bool = True,
    ) -> None:

        self.df = df
        self.dates = df['date']
        self.df = self.df.set_index('date')  # 把date设为df的索引
        self.assets = df['tic']
        self.patient = patient
        self.currency = currency
        self.is_train = is_train
        self.initial_amount = initial_amount
        self.print_verbosity = print_verbosity
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.window_size = 20
        self.state_list = self.state
        self.state_space = len(self.state_list)*self.window_size
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.seed()
        self.episode = -1
        self.episode_history = []
        self.printed_header = False
        self.max_total_assets = 0
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": []
        }
        self.rolling_window = True
        self.normalization = 'standardization' # choose thw way to process normalization ('div_self' / 'div_close' / 'standardization')
        self.do_normalization()


    def do_normalization(self):
        norm_list = ["open", "close", "high", "low", "volume"]
        after_norm = ["open_", "close_", "high_", "low_", "volume_"]
        if self.normalization == 'div_self':
            # 将ochlv处理为涨跌幅
            self.df[after_norm] = self.df[norm_list].pct_change(-1)
        elif self.normalization == 'div_close':
            # 将ochl处理为相较于前一天close的比例
            temp = self.df[["open", "close", "high", "low"]].values[1:] / self.df[["close"]].values[:-1]
            self.df = self.df[1:]
            # volume 单独处理
            self.df[["volume_"]] = self.df[["volume"]].pct_change(-1)
            self.df[["open_", "close_", "high_", "low_"]] = temp
        elif self.normalization == 'standardization':
            pass
            # for i in range(0, len(self.dates) - self.window_size,self.window_size):
            #     temp = self.df.loc[self.df.index[i]:self.df.index[i + self.window_size - 1],norm_list]
            #     scaler = preprocessing.StandardScaler().fit(temp)
            #     self.df.loc[self.df.index[i]:self.df.index[i + self.window_size - 1],after_norm] = scaler.transform(temp)
            # self.df = self.df.dropna()
            # self.df.to_csv("test_st.csv")


    def reset(self) -> np.ndarray:
        self.seed()
        self.max_total_assets = self.initial_amount
        self.starting_point = 0
        self.date_index = self.starting_point
        self.episode += 1
        self.actions_memory = []
        self.transaction_memory = []
        self.state_memory = []
        self.coh_memory = [1e+6]
        self.holdings_memory = [0]
        self.account_information = {
            "cash": [],
            "asset_value": [],
            "total_assets": [],
            "reward": []
        }
        init_state = np.zeros(self.state_space)
        self.state_memory.append(init_state)
        # else:
        #     init_state = [0] * self.state_space
        #     self.state_memory.append(init_state + [0, 0])
        return init_state

    def step(
            self, actions: np.ndarray
    ) -> Tuple[list, float, bool, dict]:
    # def step(
    #         self, actions: np.ndarray, action_p
    # ) -> Tuple[list, float, bool, dict]:
        # log
        self.log_header()
        if (self.current_step + 1) % self.print_verbosity == 0:
            self.log_step(reason="update")
        # save evaluate information on last step for each episode
        if self.date_index == len(self.dates) - 1:
            if self.is_train:
                save_path = f"train_record/train_action{self.episode}.csv"
                self.save_transaction_information().to_csv(save_path)
            return self.return_terminal(reward=self.reward)
        else:
            real_action = actions - 1
            transactions = real_action * 1000
            begin_cash = self.cash_on_hand
            assert_value = np.dot(self.holdings, self.closings)  # 目前总持股金额
            reward = self.reward

            self.account_information["cash"].append(begin_cash)
            self.account_information["asset_value"].append(assert_value)
            self.account_information["total_assets"].append(begin_cash + assert_value)
            self.account_information["reward"].append(reward)
            self.actions_memory.append(real_action)
            self.transaction_memory.append(transactions)

            sells = -np.clip(transactions, -np.inf, 0)
            proceeds = np.dot(sells, self.closings)
            costs = proceeds * self.sell_cost_pct
            coh = begin_cash + proceeds  # 计算现金的数量
            buys = np.clip(transactions, 0, np.inf)
            spend = np.dot(buys, self.closings)
            costs += spend * self.buy_cost_pct

            # 如果为负数，我们可以理解为做空，因此暂时不用考虑买不买得起，
            # if (spend + costs) > coh:  # 如果买不起
            #     if self.patient:
            #         transactions = np.where(transactions > 0, 0, transactions)
            #         spend = 0
            #         costs = 0
            #     else:
            #         return self.return_terminal(
            #             reason="CASH SHORTAGE", reward=self.reward
            #         )
            # assert (spend + costs) <= coh
            
            coh = coh - spend - costs
            holdings_updated = self.holdings + transactions
            self.date_index += 1


            #if (self.date_index + self.window_size - 1) < len(self.df):
                # do standardization in a sliding window
            if self.normalization == "standardization":
                temp = self.df.loc[
                        self.df.index[self.date_index]:self.df.index[self.date_index + self.window_size - 1],
                        ["open", "close", "high", "low", "volume"]]
                scaler = preprocessing.StandardScaler().fit(temp)
                self.df.loc[self.df.index[self.date_index]:self.df.index[self.date_index + self.window_size - 1],
                ["open_", "close_", "high_", "low_", "volume_"]] = scaler.transform(temp)
            state = self.df.loc[self.df.index[self.date_index]:self.df.index[self.date_index + self.window_size - 1],
                    self.state_list]
            state = state.values.reshape(1,220)
            self.state_memory.append(state)
            self.coh_memory.append(coh)
            self.holdings_memory.append(holdings_updated)
            return state, reward, False, {}

            # if self.rolling_window:
            #     # state - rolling_window
            #     state = self.df.loc[self.df.index[self.date_index]:self.df.index[self.date_index + self.window_size-1],
            #             self.state_list]
            #     state = state.values
            #     # # 存入state_memory之前加入coh & holding_updated
            #     # state_memory = state.loc[self.df.index[self.date_index + self.window_size]] = [0] * (len(self.state_list)-2) + [coh, holdings_updated]
            #     self.state_memory.append(state)
            # else:
            #     # state - single_day
            #     state = [self.df.loc[self.df.index[self.date_index], self.state_list]]
            #     self.state_memory.append(state + [coh, holdings_updated])



    def seed(self, seed: Any = None) -> None:
        """设置随机种子"""
        if seed is None:
            seed = int(round(time.time() * 1000))
        random.seed(seed)

    def log_step(
            self, reason: str, terminal_reward: float = None
    ) -> None:
        """打印"""
        if terminal_reward is None:
            terminal_reward = self.account_information["reward"][-1]
        assets = self.account_information["total_assets"][-1]
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount  # GAINLOSS_PCT

        rec = [
            self.episode,
            self.date_index - self.starting_point,
            reason,
            f"{self.currency}{'{:0,.0f}'.format(float(self.account_information['cash'][-1]))}",
            f"{self.currency}{'{:0,.0f}'.format(float(assets))}",
            f"{terminal_reward * 100:0.5f}%",
            f"{(gl_pct - 1) * 100:0.5f}%",
        ]
        self.episode_history.append(rec)
        print(self.template.format(*rec))

    def return_terminal(
            self, reason: str = "Last Date", reward: int = 0
    ) -> Tuple[list, int, bool, dict]:
        """terminal 的时候执行的操作"""
        state = self.state_memory[-1]
        self.log_step(reason=reason, terminal_reward=reward)
        gl_pct = self.account_information["total_assets"][-1] / self.initial_amount
        reward_pct = gl_pct
        return state, reward, True, {}

    def log_header(self) -> None:
        """Log 的列名"""
        if not self.printed_header:
            self.template = "{0:4}|{1:4}|{2:15}|{3:15}|{4:15}|{5:10}|{6:10}"
            # 0, 1, 2, ... 是序号
            # 4, 4, 15, ... 是占位格的大小
            print(
                self.template.format(
                    "EPISODE",
                    "STEPS",
                    "TERMINAL_REASON",
                    "CASH",
                    "TOT_ASSETS",
                    "TERMINAL_REWARD",
                    "GAINLOSS_PCT",
                )
            )
            self.printed_header = True

    # 存交易信息
    def save_transaction_information(self) -> pd.DataFrame:
        if self.current_step == 0:
            return None
        else:
            action_df = pd.DataFrame(
                {
                    "close": self.df["close"][-len(self.account_information["cash"]):],
                    "episode": self.episode,
                    "actions": self.actions_memory,
                    "score0": self.action_probability0,
                    "score1": self.action_probability1,
                    "score2": self.action_probability2,
                    "transactions": self.transaction_memory,
                    "total_assets": self.account_information["total_assets"],
                    "reward": self.get_reward(),
                    "assets_baseline": self.assets_baseline()
                })
            return action_df

    # buy and hold
    def assets_baseline(self):
        # 基于首次交易全仓买入股票后不再交易的assets量，反映个股自身变动
        close = self.df["close"][0]
        initial_assets = int(self.account_information["total_assets"][0] / (close * 100)) * 100
        return self.df["close"][-len(self.account_information["cash"]):] * initial_assets

    @property
    def current_step(self) -> int:
        """当前回合的运行步数"""
        return self.date_index - self.starting_point

    @property
    def cash_on_hand(self) -> float:
        """当前拥有的现金"""
        coh = self.coh_memory[-1]
        return coh

    @property
    def holdings(self) -> List:
        """当前的持仓数据"""
        holdings = self.holdings_memory[-1]
        return holdings

    @property
    def closings(self) -> List:
        """每支股票当前的收盘价"""
        close = self.df.loc[self.df.index[self.date_index], "close"]
        return close

    @property
    def state(self):
        state1 = ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24']
        state2 = ['kdjk', 'kdjd', 'kdjj', 'rsi_6', 'rsi_12', 'rsi_24', "open_", "close_", "high_", "low_", "volume_"]
        return state2

    @property
    def reward(self) -> float:
        if self.current_step == 0:
            return 0
        else:
            # reward1: only total assets
            reward1 = self.account_information["total_assets"][-1]
            # reward2: total assets + retreat
            assets = self.account_information["total_assets"][-1]
            retreat = 0
            if assets >= self.max_total_assets:
                self.max_total_assets = assets
            else:
                retreat = assets / self.max_total_assets - 1

            reward2 = assets / self.initial_amount - 1
            reward2 += retreat
            return reward1


