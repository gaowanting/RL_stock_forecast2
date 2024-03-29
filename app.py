import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from integration import Integration
# MDPP_D = st.sidebar.slider("MDPP_D", 1, 10, 5, 1)
# MDPP_P = st.sidebar.slider("MDPP_D", 0.001, 0.050, 0.03, 0.001)
# single_true = st.sidebar.checkbox("single_true", True)
# windows_size = st.sidebar.slider("windows_size", 10, 500, 50, 10)

# for a, b, c in os.walk("../DQN/train_record"):
#     train_record = c
# select_actiondf = st.sidebar.selectbox("choice_action_meomry", train_record)
# print(select_actiondf)

# a = Evaluation(MDPP_D=MDPP_D, MDPP_P=MDPP_P, single_true=single_true, windows_size=windows_size)

# dd = pd.read_csv("train_record/" + select_actiondf)

# show1 = a.auc(dd)
# show = a.sliding_curve(dd)
# st.line_chart(show1)
# chart2 = st.line_chart(show)

for a, b, c in os.walk("reward_out"):
    train_record = c

select_ = st.sidebar.selectbox("train", train_record)
with open('reward_out/' + select_, 'rb') as data:
    json = pickle.load(data)[int(select_[:-4])]
    details = json['reward_details']
st.text("本次训练轮数:" + str(json['train_episode']))
st.text("股票代码:" + str(json['stock_code']))
select_details = st.sidebar.selectbox("detail_reward", details.keys())
windows_size = st.sidebar.slider("windows_size: ma detail reward", 10, 500, 50, 10)
input_code = st.sidebar.text_input("input stock_code you want to predict")
action = Integration().watchlist_predict(input_code)
if action == -1:
    st.sidebar.text("result: sell")
elif action == 0:
    st.sidebar.text("result: hold")
elif action == 1:
    st.sidebar.text("result: buy")
else:
    st.sidebar.text(action)
"每个step的reward"
chart0 = st.line_chart(details[select_details])
"滑动窗口step的平均reward"
chart1 = st.line_chart(pd.Series(details[select_details]).rolling(windows_size).mean())
"本次train中的reward"
chart2 = st.line_chart(json['reward'])
"本次train中的移动平均"
chart3 = st.line_chart(json['ma_reward'])


if __name__ == '__main__':
    ...
