import codecs
import os
import sys
import pandas as pd
from argparse import ArgumentParser
from stable_baselines3.common.vec_env import DummyVecEnv
import datetime

sys.path.append("..")
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from utils import config
from utils.env import StockLearningEnv
from utils.models import DRL_Agent
# from data import Data


class Trainer(object):
    """用来训练的类

    Attributes:
        model_name: 强化学习的算法名称，用来调用指定的算法
        total_timesteps: 总的训练步数
    """

    def __init__(self, model_name = 'ddpg' ,
                        total_timesteps = 200000) -> None:
        self.model_name = model_name
        self.total_timesteps = total_timesteps
        self.train_dir = "train_file"
        self.data_dir = "data_file"
        self.create_train_dir()
    
    def create_train_dir(self) -> None:
        """创建存储训练结果的文件夹"""
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
            print("{} 文件夹创建成功!".format(self.train_dir))
        else:
            print("{} 文件夹已存在!".format(self.train_dir))
    
    def train(self) -> None:
        """开始训练"""
        train_data, trade_data = pd.read_csv("..\\new_data_file\\train.csv"), pd.read_csv("..\\new_data_file\\trade.csv")
        env_train, env_trade = self.get_env(train_data, trade_data)

        agent = DRL_Agent(env = env_train)

        model = agent.get_model(self.model_name,  
                                model_kwargs = config.__dict__["{}_PARAMS".format(self.model_name.upper())], 
                                verbose = 0)
        model.learn(total_timesteps = self.total_timesteps, 
                    eval_env = env_train, # trade
                    eval_freq = 500,
                    log_interval = 1, 
                    tb_log_name = 'env_cashpenalty_highlr',
                    n_eval_episodes = 1)
        self.save_model(model)
    '''
    def get_data(self):
        """获取训练数据集和交易数据集"""
        train_data_path = os.path.join(self.data_dir, "train.csv")
        trade_data_path = os.path.join(self.data_dir, "trade.csv")
        if not (os.path.exists(train_data_path) or
                os.path.exists(trade_data_path)):
            print("数据不存在，开始下载")
            Data().pull_data()
        
        train_data = pd.read_csv(train_data_path)
        trade_data = pd.read_csv(trade_data_path)
        print("数据读取成功!")
        
        return train_data, trade_data
    '''
    def get_env(self, 
                train_data: pd.DataFrame, 
                trade_data: pd.DataFrame) -> DummyVecEnv:
        """分别返回训练环境和交易环境"""
        e_train_gym = StockLearningEnv(df = train_data,
                                       is_train= True,
                                                    random_start = True,
                                                    **config.ENV_PARAMS)
        env_train, _ = e_train_gym.get_sb_env()

        e_trade_gym = StockLearningEnv(df = trade_data,
                                       is_train=False,
                                                    random_start = False,
                                                    **config.ENV_PARAMS)
        env_trade, _ = e_trade_gym.get_sb_env()

        return env_train, env_trade

    def save_model(self, model) -> None:
        model_path = os.path.join(self.train_dir, "{}.model".format(self.model_name))
        model.save(model_path)
    # .model 就是个zip，用import zipfile可以展开

def start_train():
    parser = ArgumentParser(description="set parameters for train mode")
    parser.add_argument(
        '--model', '-m',
        dest='model',
        default='ppo',
        help='choose the model you want to train',
        metavar="MODEL",
        type=str
    )

    parser.add_argument(
        '--total_timesteps', '-tts',
        dest='total_timesteps',
        default=5000,
        help='set the total_timesteps when you train the model',
        metavar="TOTAL_TIMESTEPS",
        type=int
    )

    options = parser.parse_args()
    Trainer(model_name = options.model,
            total_timesteps = options.total_timesteps).train()

if __name__ == "__main__":
    # from data_file import train_creat
    # train_creat.runnum(10)
    start_time = datetime.datetime.now()
    start_train()
    end_time = datetime.datetime.now()
    print("training finished, time{}".format(end_time - start_time))