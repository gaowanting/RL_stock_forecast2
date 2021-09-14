import pandas as pd
import re


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


if __name__ == '__main__':
    # int(action_abstract(pd.read_csv("new_train_file\\train_action30.csv"))[0])
    print(type(action_abstract(pd.read_csv("new_train_file\\train_action30.csv"))[0]))
