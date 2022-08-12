#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:48:57
@LastEditor: Wanting Gao
LastEditTime: 2021-11-21
@Discription: 
@Environment: python 3.8
'''
import os
import pandas as pd
import torch
import datetime
from agent import DQN
from env import StockLearningEnv
import pickle
import numpy as np
import sqlite3
from util.visualize import plot_signal

config = {
    "algo": "DQN",
    "train_eps": 10,
    "eval_eps": 5,
    "gamma": 0.95,
    "epsilon_start": 0.90,
    "epsilon_end": 0.01,
    "epsilon_decay": 500,
    "lr": 0.0001,
    "memory_capacity": 100000,
    "batch_size": 64,
    "target_update": 4,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "hidden_dim": 256, 
    "save_model": 2,   # 多少episode保存一次模型
    "save_result": 1   # 多少episode保存一次结果
}

class Trainer:
    def __init__(self, config, agent, env) -> None:
        self.config = config
        self.agent = agent
        self.env = env
        self.result_dir = './outputs/result'
        self.model_dir = './outputs/model'

    def create_data_dir(self):
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            print("{} 文件夹创建成功!".format(self.result_dir))
        else:
            print("{} 文件夹已存在!".format(self.result_dir))
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            print("{} 文件夹创建成功!".format(self.model_dir))
        else:
            print("{} 文件夹已存在!".format(self.model_dir))

    def train(self):
        print('Start to train !')
        print(f'Env:{env}, Algorithm:{self.config["algo"]}, Device:{self.config["device"]}')
        rewards = []
        ma_rewards = []  # moving average reward
        for i_ep in range(self.config["train_eps"]):
            reward_detail = []
            state = env.reset()
            done = False
            ep_reward = 0
            ep_step = 0
            while ep_step < (len(self.env.df) - 2*env.window_size):
                ep_step += 1
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                agent.memory.push(state, action, reward, next_state, done)
                reward_detail.append(reward)
                state = next_state
                agent.update()
                if done:
                    break
            # save ma rewards
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)
            if (i_ep + 1) % self.config["target_update"] == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            if (i_ep + 1) % 10 == 0:
                print('Episode:{}/{}, Reward:{}'.format(i_ep + 1, self.config["train_eps"], ep_reward))

            if (i_ep + 1) % self.config["save_model"] == 0:
                # agent.save("./outputs/model/"+ str(i_ep) + ".pth")
                torch.save(agent.target_net.state_dict(), "./outputs/model/"+ str(i_ep) + ".pth")
            if (i_ep + 1) % self.config["save_result"] == 0:
                result = env.save_transaction_information()
                # visualize signal
                plot_signal(result)
                result.to_csv("./outputs/result/"+ str(i_ep) + ".csv")

                # with open("./outputs/result/"+ str(i_ep) + ".pkl", 'wb') as res:
                #     pickle.dump(reward_detail, res)
            rewards.append(ep_reward)

        print('Complete training！')
        return rewards, ma_rewards


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    # df = pd.read_csv(r"./util/new_data_file/data.csv")
    con = sqlite3.connect(os.path.join(os.getcwd(), 'data.db'))
    df = pd.read_sql('SELECT * FROM train_data',con,index_col='index')
    df['embedding'] = [float(i) for i in df['embedding']]
    env = StockLearningEnv(df)
    agent = DQN(env.state_space, env.action_space.n, **config)
    trainer = Trainer(config, agent, env)
    trainer.create_data_dir()
    trainer.train()
    end_time = datetime.datetime.now()
    print("training finished, time {}".format(end_time - start_time))
  
