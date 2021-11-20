import torch
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import numpy as np

class StockDataset(Dataset):
    def __init__(self,df):
        self.df = df
        stock_data = []
        for i in range(len(self.df)-20):
            temp = self.df.iloc[i:20+i,2:].to_numpy()
            stock_data.append(temp)
        self.stock_data = stock_data


    def __getitem__(self,index):
        data = self.stock_data[index]
        data = torch.tensor(data,dtype=torch.float32)
        return data

    def __len__(self):
        return len(self.stock_data)

if __name__ == "__main__":
    
    df = pd.read_csv(r".\data\STOCK\data.csv")
    train_data = StockDataset(df)
    dataloader = DataLoader(train_data,batch_size=8,shuffle=False)
    for i in enumerate(dataloader):
        # i[1] get data len(i[1]) is batch_size
        breakpoint()