import pandas as pd
import re
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import predict

def action_abstract(df):
    df["abstract_actions"] = "0"
    for i in range(df["actions"].__len__()):
        real_action = float(re.findall(r'\[(.*)\]', df["actions"][i])[0])
        if real_action > 0:
            df.loc[i, "abstract_actions"] = "1"
        elif real_action == 0:
            pass
        else:
            df.loc[i, "abstract_actions"] = "-1"
    df.to_csv("loock.csv")
    return df["abstract_actions"]

def roc(sample_num):
    # df is from new_data_file\\train.csv
    predict_ = predict.Predicter(model_name="ppo")
    predict_.load_params()
    df = pd.read_csv("..\\new_data_file\\train.csv")
    # df = pd.concat([df, landmark.best_action()])
    d = df.copy()
    df.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) < 0), 'landmark'] = "/"
    df.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) > 0), 'landmark'] = "\\"
    df.loc[(d.close.diff(1) > 0) & (d.close.diff(-1) > 0), 'landmark'] = '^'
    df.loc[(d.close.diff(1) < 0) & (d.close.diff(-1) < 0), 'landmark'] = 'v'
    list_sample = []   # 概率
    list_sample_x = []   # 分类
    for i in range(sample_num):  # df.__len__()
        x_score = 0  # -1
        y_score = 0  # 0
        z_score = 0  # 1
        x = 0   # 同上
        y = 0
        z = 0
        if df.loc[i, 'landmark'] == "^":
            x = 1
        elif df.loc[i, 'landmark'] == "v":
            z = 1
        else:
            y = 1
        for j in range(100):
            score = predict_.predict(i)[0]
            if 0.1 < score <= 1:
                z_score += 1
            if 0.1 >= score >= -0.1:
                y_score += 1
            if -0.1 > score >= -1:
                x_score += 1
        list_sample.append([x_score/100, y_score/100, z_score/100])
        list_sample_x.append([x, y, z])
        print(i)
    list_sample = np.array(list_sample)
    list_sample_x = np.array(list_sample_x)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(list_sample_x[:, i], list_sample[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(list_sample_x.ravel(), list_sample.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             color='deeppink', linestyle=':', linewidth=4)
    print("auc:", roc_auc["micro"])
    plt.show()


if __name__ == '__main__':
    # int(action_abstract(pd.read_csv("new_train_file\\train_action30.csv"))[0])
    print(type(action_abstract(pd.read_csv("new_train_file\\train_action30.csv"))[0]))
