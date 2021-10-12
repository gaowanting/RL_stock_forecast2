import streamlit as st
from evaluate import Evaluation
from task0_train import *
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

MDPP_D = st.sidebar.slider("MDPP_D", 1, 10, 5, 1)
MDPP_P = st.sidebar.slider("MDPP_D", 0.001, 0.050, 0.03, 0.001)
single_true = st.sidebar.checkbox("single_true", True)
windows_size = st.sidebar.slider("windows_size", 10, 500, 50, 10)
random_model = st.sidebar.checkbox("use_random_model?", True)

cfg = DQNConfig()
env, agent = env_agent_config(cfg, seed=1)
if not random_model:
    agent.load('outputs/StockLearningEnv/20210922-172114/models/')
a = Evaluation(cfg, env, agent, MDPP_D, MDPP_P, single_true, windows_size)
chart = st.line_chart(a.auc_show)

a.action_model()
mat = pd.DataFrame(index=['Macro', 'Weighted', 'micro'], columns=['accuracy', 'precision', 'recall', 'F1_score'])

mat['accuracy'] = pd.Series(data=[a.accuracy(), '-', '-'], index=['Macro', 'Weighted', 'micro'])
mat['precision'] = pd.Series(data=a.precision(), index=['Macro', 'Weighted', 'micro'])
mat['recall'] = pd.Series(data=a.recall(), index=['Macro', 'Weighted', 'micro'])
mat['F1_score'] = pd.Series(data=a.F1_score(), index=['Macro', 'Weighted', 'micro'])
st.write("evaluate matrix", mat)


a.auc(chart)



