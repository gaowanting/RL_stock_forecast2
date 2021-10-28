import streamlit as st
from evaluate import Evaluation
from task0_train import *
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

MDPP_D = st.sidebar.slider("MDPP_D", 1, 10, 5, 1)
MDPP_P = st.sidebar.slider("MDPP_D", 0.001, 0.050, 0.03, 0.001)
single_true = st.sidebar.checkbox("single_true", True)
windows_size = st.sidebar.slider("windows_size", 10, 500, 50, 10)

for a, b, c in os.walk("../DQN/train_record"):
    train_record = c
select_actiondf = st.sidebar.selectbox("choice_action_meomry", train_record)
print(select_actiondf)

a = Evaluation(MDPP_D=MDPP_D, MDPP_P=MDPP_P, single_true=single_true, windows_size=windows_size)

dd = pd.read_csv("train_record/" + select_actiondf)

show1 = a.auc(dd)
show = a.sliding_curve(dd)
st.line_chart(show1)
chart2 = st.line_chart(show)



