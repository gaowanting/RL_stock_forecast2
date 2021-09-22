from utils.env import StockLearningEnv
from utils.models import DRL_Agent
from utils import config
import pandas as pd
# import sys
# sys.path.append('.')

import trainer
import warnings


class Predicter(trainer.Trainer):
    # original method is from base_class(satblebaseline3)
    def __init__(self, model_name="ppo"):
        self.model_name = model_name

    def get_Agent(self) -> DRL_Agent:
        # use father class Trainer's method get_env to creat a DummyVecEnv
        # then creat a Agent based on stable_baseiine3
        #

        self.train_data, trade_data = pd.read_csv("..\\new_data_file\\train.csv"), pd.read_csv(
            "..\\new_data_file\\trade.csv")

        env_train, env_trade = self.get_env(self.train_data, trade_data)

        agent = DRL_Agent(env=env_train)

        return agent

    def load_params(self):
        # load model params from train_file

        agent = self.get_Agent()
        # try:
        self.model = agent.get_model(self.model_name,
                                     model_kwargs=config.__dict__["{}_PARAMS".format(self.model_name.upper())],
                                     verbose=0)
        self.model.load(path="train_file\\{}.model".format(self.model_name))
        # except:
        #
        #     raise "you have not trained model{}".format(self.model_name)

    def predict(self):
        df = self.train_data.copy()
        predict_obs = list(df.loc[0, :].drop(["date", "tic", "day", "volume_120_sma"]))
        # print(predict_obs)
        # print(type(predict_obs))
        # method path stable_baseline3/common/base_class predict()
        return self.model.predict(predict_obs)


if __name__ == '__main__':
    a = Predicter(model_name="ppo")
    a.load_params()
    print(a.predict())
    pass
