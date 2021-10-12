import warnings

import numpy as np

from task0_train import *
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class Evaluation:
    # 评估类， auc(), 绘制滑动auc曲线
    #        accuracy()， 返回准确率
    #        precision()， 返回三种多分类精准度
    #        recall()， 返回三种多分类召回率
    #        F1_score()， 返回三种多分类F1分数
    def __init__(self, cfg, env, agent, MDPP_D=5, MDPP_P=0.03, single_true=True, windows_size=50):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.confusion_matrix = None
        self.MDPP_D = MDPP_D
        self.MDPP_P = MDPP_P
        self.single_true = single_true   # auc c(1,n)默认单个元素为真值
        self.windows_size = windows_size   # auc滑动窗口大小
        self.auc_showbuffer = []
        self.auc_show = []

    def landmarket(self):
        # signal the right action trading
        d = self.env.df.copy()
        d.index = range(len(d))
        d['landmark'] = -1  # 初始化
        for _ in range(self.MDPP_D):
            # 找出相邻点满足变化<P的数据行
            d.loc[
                d[~(2 * abs(d.close.diff(1)) / (abs(d.close) + abs(d.close.shift(1))) < self.MDPP_P)].index, 'landmark'] = 1
            # 有问题，这里得改一下。
            # print(d.shape)

        d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) < 0) & (d.landmark != 1), 'landmark'] = 2  # '/'
        d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) > 0) & (d.landmark != 1), 'landmark'] = 0  # '\\'
        d.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) > 0) & (d.landmark != 1), 'landmark'] = 0  # '^'
        d.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) < 0) & (d.landmark != 1), 'landmark'] = 2  # 'v'

        # print(d)
        d.loc[(d.landmark == -1), 'landmark'] = 0  # 以上算法会让某些位残留-1，这里改为0避免后面出错
        self.d = d

    def action_model(self):
        # use trained model to predict on eval data once
        self.landmarket()
        state = self.env.reset()
        step_num = 0
        ture_num = 0
        self.d['action', 'score0', 'score1', 'score2'] = 0
        while True:
            action, score = self.agent.predict(state)
            if action == self.d.landmark[step_num]:
                ture_num += 1
            self.d.loc[step_num, 'action'] = action
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

    def auc(self, show_ = None):
        # use windows on target df and calculate bunch of auc

        for i in range(self.d.__len__() - self.windows_size):
            y = self.d.loc[range(i, i + self.windows_size), 'landmark']
            y = y.reset_index(drop=True)
            score = self.d.loc[range(i, i + self.windows_size), ['score0', 'score1', 'score2']]
            score = score.reset_index(drop=True)
            # print(y)
            # print(score)
            res = self.auc_(y, score)
            # print(res)
            # auc_list.append(pd.Series(res.values()).mean())
            self.auc_showbuffer += [pd.Series(res.values()).mean()]
            # print(self.auc_showbuffer)
            if i % 20 == 0:
                self.auc_show += self.auc_showbuffer

                print(self.auc_show.__len__())
                if show_ is not None:
                    show_.add_rows(self.auc_showbuffer)
                self.auc_showbuffer = []
                # print(self.auc_show)
        # plt.plot(range(auc_list.__len__()), auc_list)
        # plt.show()

    def accuracy(self):
        count = 0
        for i in self.d.index:
            if self.d.action[i] == self.d.landmark[i]:
                count += 1
        return count / self.d.__len__()

    def precision(self):
        # for Accuracy，Precision，Recall and F1 score
        # print(self.confusion_matrix)
        precision_list = []
        weighted_p = 0
        _ = self.d.groupby('landmark')['landmark'].count()
        if self.confusion_matrix is None:
            raise Exception("未计算的属性confusion_matrix")
        for i in self.confusion_matrix.columns:
            m = self.confusion_matrix[i]['TP'] + self.confusion_matrix[i]['FP']
            if m == 0:
                warnings.warn("self.confusion_matrix[i]['TP'] + self.confusion_matrix[i]['FP'] == 0")
                continue
            pre_i = self.confusion_matrix[i]['TP'] / m
            precision_list.append(pre_i)
            weighted_p += _[i] * pre_i
            # print(weighted_p)

        Macro_precision = pd.Series(precision_list).mean()  # 简单平均

        Weighted_precision = weighted_p/self.d.__len__()  # 加权平均，权：真实分类的权重
        # print(weighted_p, self.d.__len__())
        micro_precision = self.confusion_matrix.loc['TP', :].sum()/(self.confusion_matrix.loc['TP', :].sum() + self.confusion_matrix.loc['FP', :].sum())
        return Macro_precision, Weighted_precision, micro_precision

    def recall(self):
        # for Accuracy，Precision，Recall and F1 score
        # print(self.confusion_matrix)
        recall_list = []
        weighted_r = 0
        _ = self.d.groupby('landmark')['landmark'].count()
        if self.confusion_matrix is None:
            raise Exception("未计算的属性confusion_matrix")
        for i in self.confusion_matrix.columns:
            m = self.confusion_matrix[i]['TP'] + self.confusion_matrix[i]['FN']
            if m == 0:
                warnings.warn("self.confusion_matrix[i]['TP'] + self.confusion_matrix[i]['FN'] == 0")
                continue
            rec_i = self.confusion_matrix[i]['TP'] / m
            recall_list.append(rec_i)
            weighted_r += _[i] * rec_i
            # print(weighted_p)

        Macro_recall= pd.Series(recall_list).mean()  # 简单平均

        Weighted_recall = weighted_r/self.d.__len__()  # 加权平均，权：真实分类的权重
        # print(weighted_p, self.d.__len__())
        micro_recall = self.confusion_matrix.loc['TP', :].sum()/(self.confusion_matrix.loc['TP', :].sum() + self.confusion_matrix.loc['FN', :].sum())
        return Macro_recall, Weighted_recall, micro_recall

    def F1_score(self):
        # F1
        pma, pwr, pmi = self.precision()
        rma, rwr, rmi = self.recall()

        return self.F1_(pma, rma), self.F1_(pma, rma), self.F1_(pma, rma)

    def F1_(self, p, r):
        # for calculate F1
        return 2*p*r/(p+r)


if __name__ == '__main__':
    cfg = DQNConfig()
    env, agent = env_agent_config(cfg, seed=1)
    agent.load('outputs/StockLearningEnv/20210922-172114/models/')
    a = Evaluation(cfg, env, agent)
    a.action_model()
    # b = a.cm(np.array(a.d['action']), np.array(a.d['landmark']))
    # z, x, c = a.F1_score()
    # print(z, x ,c)
    # print(type(z),type(x),type(c))
    a.auc()
    # a.cm([0, 1, 2, 2, 2, 1, 1, 2, 0 , 0, 1, 2 ,0], [1, 1 ,1, 2 ,2 ,1, 1, 0 ,0 , 1, 1, 2, 0])
    # print(a.env.df)
    # print(agent.predict([3, 3, 3, 3, 3, 3]))
