import pandas as pd

# df = pd.read_csv("new_train_file\\train_action30.csv")
#
# df["reward"] = df.loc[:, "reward"][-20:]*100
# df["sb"] = pd.Series(["a", "b"]) # 直接放series则不报错 自动填补NAN
# df["/"] = df["assets_baseline"]/df["close"]
# S = df.loc[0, :]
# # pd.Series().drop
# print(S.drop(["close", "date"]))
# print(list(S))

class A:
    def __init__(self):
        self.a = 1
        self.b = 2
    def out(self):
        print(self.__dict__)
        print(self.c)

class B(A):
    def __init__(self):
        self.c = 3

if __name__ == '__main__':
    pass
    # print(float("-1.2")+1)
    # b = B()
    # b.out()
    # import zipfile
    #
    # zipFile = zipfile.ZipFile("train_file\\ppo.model")
    # for file in zipFile.namelist():
    #     zipFile.extract(file, 'Work')
    # zipFile.close()

    # import torch
    #
    # static_dict = torch.load("Work\\policy.pth")
    # # pytorch_variables.pth
    # print(type)
    # for i in static_dict:
    #     print(i)
    #     # print(static_dict[i])
    #     print(type(static_dict[i]))
    import datetime

    now_time = datetime.datetime.now()
    noww_time = now_time.strftime('%Y%m%d')
    print("now time: ", noww_time)

    # 获取前一天时间
    end_time = now_time + datetime.timedelta(days=-1)

    # 前一天时间只保留 年-月-日
    enddate = end_time.strftime('%Y%m%d')  # 格式化输出
    print("end date: ", enddate)

    # print(datetime.datetime.now().strftime("%Y%m%D"))
