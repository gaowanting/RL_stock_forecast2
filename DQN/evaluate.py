import os
import warnings

import numpy as np

from task0_train import *
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

evaluation_config = {
}


class Evaluation:
    # 评估类， auc(), 绘制滑动auc曲线
    #        accuracy()， 返回准确率
    #        precision()， 返回三种多分类精准度
    #        recall()， 返回三种多分类召回率
    #        F1_score()， 返回三种多分类F1分数
    def __init__(self, cfg=None, env=None, agent=None, MDPP_D=5, MDPP_P=0.03, single_true=True, windows_size=50):
        # 评估训练集动作集时可不输入配置，环境，代理这三个参数。
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.MDPP_D = MDPP_D
        self.MDPP_P = MDPP_P
        self.single_true = single_true  # auc c(1,n)默认单个元素为真值
        self.windows_size = windows_size  # auc滑动窗口大小
        # self.auc_showbuffer = []
        # self.auc_show = []
        # self.accuracy_show = []

    def landmark(self, df):
        # signal the right action trading
        # df:  some dataframe with column "close"
        d = df.copy()
        d.index = range(len(d))
        d['landmark'] = -1  # 初始化

        # 有问题，这里得改一下。
        # print(d.shape)

        d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) < 0) & (d.landmark != 1), 'landmark'] = 2  # '/'
        d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) > 0) & (d.landmark != 1), 'landmark'] = 0  # '\\'
        d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) > 0) & (d.landmark != 1), 'landmark'] = 0  # '^'
        d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) < 0) & (d.landmark != 1), 'landmark'] = 2  # 'v'
        d = d[d.landmark.isin([0, 2])]
        for _ in range(self.MDPP_D):
            # 找出相邻点满足变化<P的数据行，这里的MDPP_D作用和论文中有些许差别？
            d = d[~(2 * abs(d.close.diff(1)) / (abs(d.close) + abs(d.close.shift(1))) < self.MDPP_P)]
        # print(d)
        d.loc[(d.landmark == -1), 'landmark'] = 0  # 以上算法会让某些位残留-1，这里改为0避免后面出错
        # assert d.groupby('landmark').__len__() == 2
        d.reset_index(inplace=True)
        return d

    def action_model(self):
        # use trained model to predict on eval data once
        if self.agent is None:
            raise Exception("please use method 'sliding_curve' to evaluate train_action")
        self.landmark()
        state = self.env.reset()
        step_num = 0
        ture_num = 0
        self.d['action', 'score0', 'score1', 'score2'] = 0
        while True:
            action, score = self.agent.predict(state)
            if action == self.d.landmark[step_num]:
                ture_num += 1
            self.d.loc[step_num, 'actions'] = action
            self.d.loc[step_num, ['score0', 'score1', 'score2']] = score
            step_num += 1

            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            if done:
                break
        self.confusion_matrix = self.cm(np.array(self.d.action), np.array(self.d.landmark))
        # print(step_num, ture_num, ture_num / step_num)
        # print(self.d)

    def cm(self, x: np.array, y: np.array, cc: list = [0, 1, 2]):
        # method for Multi classification
        # x:predict_result y:real|target_result cc: list of unique x|y class container
        assert x.__len__() == y.__len__()
        res = pd.DataFrame(index=['TP', 'FN', 'FP'], columns=cc)  #
        for i in cc:
            # cm = pd.DataFrame(np.zeros((cc.__len__(), cc.__len__())), columns=cc, index=cc)  # cm: confusion matrix
            TP = 0
            FN = 0
            FP = 0
            for j in range(x.__len__()):
                if x[j] == i:
                    if y[j] == i:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if y[j] == i:
                        FN += 1
            res[i] = pd.Series(index=['TP', 'FN', 'FP'], data=[TP, FN, FP])
        assert all(res.columns == cc)
        return res
        # print(res)

    def auc_(self, y: pd.Series, y_score: pd.DataFrame) -> dict:
        # input: y: 多分类的真值输入, y_score: 真直得分, single_true: 是否以单个的那个元素为真值
        # output: list n分类的C(1,n)个子二分类auc字典       方法有个y_score的列顺序问题，
        # 警告：仅用于多分类，二分类直接调用roc_curve就行了
        cc = y.unique()  # 获取多分类种类
        res = {}
        for i in range(cc.__len__()):
            # print(cc)
            # print(cc.__len__())
            li_ = []  # 构造以i为二分类中的一项的二分类列表
            li_score = []  # 二分类score列表
            for j in range(y_score.__len__()):
                if y[j] == cc[i]:
                    li_.append(1)
                    li_score.append(y_score.iloc[j, i])
                else:
                    li_.append(0)
                    li_score.append(y_score.iloc[
                                        j, i])  # 真值不是它，则二分类score为其他的所有score合 y_score.iloc[j, list(set(range(cc.__len__())) - {i})].sum()
            if self.single_true:
                tpr, fpr, _ = roc_curve(li_, li_score, pos_label=1)
            else:
                tpr, fpr, _ = roc_curve(li_, li_score, pos_label=0)
            res[cc[i]] = auc(tpr, fpr)
        # print(res)
        return res

    def auc(self, d, show_=None):
        # use windows on target df and calculate bunch of auc
        d = self.landmark(d)
        la = []
        for i in range(d.__len__() - self.windows_size):
            y = d.loc[range(i, i + self.windows_size), 'landmark']
            y = y.reset_index(drop=True)
            score = d.loc[range(i, i + self.windows_size), ['score0', 'score1', 'score2']]
            score = score.reset_index(drop=True)
            # print(y)
            # print(score)
            res = self.auc_(y, score)
            la.append(pd.Series(res.values()).mean())
            # print(res)
            # auc_list.append(pd.Series(res.values()).mean())
            # self.auc_showbuffer += [pd.Series(res.values()).mean()]
            # print(self.auc_showbuffer)
            # if i % 20 == 0:
            #     self.auc_show += self.auc_showbuffer
            #
            #     # print(self.auc_show.__len__())
            #     if show_ is not None:
            #         show_.add_rows(self.auc_showbuffer)
            #     self.auc_showbuffer = []
            # print(self.auc_show)
        # plt.plot(range(auc_list.__len__()), auc_list)
        # plt.show()
        return la

    def accuracy(self, df):
        count = 0
        for i in df.index:
            if df.actions[i] == df.landmark[i]:
                count += 1
        return count / df.__len__()

    def precision(self, cm, d):
        # for Accuracy，Precision，Recall and F1 score
        # print(self.confusion_matrix)
        # cm: from method cm(),  d: dataframe of any windows
        precision_list = []
        weighted_p = 0
        _ = pd.Series([0, 0, 0], index=[0, 1, 2])
        __ = d.groupby('landmark')['landmark'].count()
        for k in __.index:
            _[k] = __[k]
        # print(cm, _)
        for i in cm.columns:
            m = cm[i]['TP'] + cm[i]['FP']
            if m == 0:
                warnings.warn("self.confusion_matrix[i]['TP'] + self.confusion_matrix[i]['FP'] == 0")
                continue
            pre_i = cm[i]['TP'] / m
            precision_list.append(pre_i)

            weighted_p += _[i] * pre_i
            # print(weighted_p)

        Macro_precision = pd.Series(precision_list).mean()  # 简单平均

        Weighted_precision = weighted_p / d.__len__()  # 加权平均，权：真实分类的权重
        micro_precision = cm.loc['TP', :].sum() / (cm.loc['TP', :].sum() + cm.loc['FP', :].sum())
        return Macro_precision, Weighted_precision, micro_precision

    def recall(self, cm, d):
        # for Accuracy，Precision，Recall and F1 score
        # print(self.confusion_matrix)
        # # cm: from method cm(),  d: dataframe of any windows
        recall_list = []
        weighted_r = 0
        _ = d.groupby('landmark')['landmark'].count()
        for i in cm.columns:
            m = cm[i]['TP'] + cm[i]['FN']
            if m == 0:
                warnings.warn("self.confusion_matrix[i]['TP'] + self.confusion_matrix[i]['FN'] == 0")
                continue
            rec_i = cm[i]['TP'] / m
            recall_list.append(rec_i)
            weighted_r += _[i] * rec_i
            # print(weighted_p)

        Macro_recall = pd.Series(recall_list).mean()  # 简单平均

        Weighted_recall = weighted_r / d.__len__()  # 加权平均，权：真实分类的权重
        # print(weighted_p, self.d.__len__())
        micro_recall = cm.loc['TP', :].sum() / (cm.loc['TP', :].sum() + cm.loc['FN', :].sum())
        return Macro_recall, Weighted_recall, micro_recall

    def F1_score(self, cm, windows):
        # F1
        pma, pwr, pmi = self.precision(cm, windows)
        rma, rwr, rmi = self.recall(cm, windows)

        return pma, pwr, pmi, rma, rwr, rmi, self.F1_(pma, rma), self.F1_(pma, rma), self.F1_(pma, rma)

    def F1_(self, p, r):
        # for calculate F1
        return 2 * p * r / (p + r)

    def sliding_curve(self, action_df):
        # just use train_action_memory without loading agents
        df = self.landmark(action_df)
        df.reset_index(inplace=True)
        df['actions'] = df['actions'] + 1
        showbuffer = pd.DataFrame(columns=[0, 1, 2, 3, 4, 5, 6, 7])
        for i in range(df.__len__() - self.windows_size):
            windows = df.loc[range(i, i + self.windows_size), :]
            cm = self.cm(np.array(windows.actions), np.array(windows.landmark), windows.actions.unique())
            accuracy = self.accuracy(windows)
            # p1, p2, p3 = self.precision(cm, windows)
            # r1, r2, r3 = self.recall(cm, windows)
            p1, p2, p3, r1, r2, r3, f1, f2, f3 = self.F1_score(cm, windows)
            showbuffer.loc[i] = pd.Series([accuracy, p1, p2, p3, r1, r2, r3, f1])
            # if i % 20 == 0:
            #     # self.auc_show += self.auc_showbuffer
            #
            #     # print(self.auc_show.__len__())
            #     if show_ is not None:
            #         show_.add_rows(showbuffer)
            #     showbuffer = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'F1_score'])
            #     k = 0
            # print(accuracy)
        print("done")
        showbuffer.columns = ['accuracy', 'precision1', 'precision2', 'precision3', 'recall1', 'recall2', 'recall3',
                              'F1_score']
        return showbuffer

    def memory_tracer(self):
        # to find all the action_memory in documents
        for a, b, c in os.walk("train_record"):
            self.train_record = c

    def main(self, **kwargs):
        # 集成接口，参数为展现的内容
        # self_action: 自己加载模型，获得数据进行决策与评估，用于对模型的评估。
        self.auc



if __name__ == '__main__':
    # cfg = DQNConfig()
    # env, agent = env_agent_config(cfg, seed=1)
    # agent.load('outputs/StockLearningEnv/20210922-172114/models/')
    a = Evaluation()
    # a.action_model()
    # b = a.cm(np.array(a.d['action']), np.array(a.d['landmark']))
    # z, x, c = a.F1_score()
    # print(z, x ,c)
    # print(type(z),type(x),type(c))
    df = pd.read_csv("train_record/train_action0.csv")
    # a.auc(df)
    print(a.sliding_curve(df))
    # a.cm([0, 1, 2, 2, 2, 1, 1, 2, 0 , 0, 1, 2 ,0], [1, 1 ,1, 2 ,2 ,1, 1, 0 ,0 , 1, 1, 2, 0])
    # print(a.env.df)
    # print(agent.predict([3, 3, 3, 3, 3, 3]))
