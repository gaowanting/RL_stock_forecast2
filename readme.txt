------readme------
基于stable_baseline3的部分：
前言：基于stable_baseline3进行训练时，需要train,test两个环境，train与test的episode会分别计算并print到终端
    可以用同一个数据集创建两个环境（注意test环境的is_Train属性为False）

learn:
    action_abstract: 抽象action为三点离散？（待定）， 计算auc_roc的方法，输入参数为对某一测试state的predict次数
    new_train_file： 存放每个episode的action_memory
	train_file：存放训练后的模型.model，通过stable_baseline3，model.load方法载入。可以通过zipfile解压缩
	actions_signal.jpg: landmark 标点后的 图，对买卖信号的标记，输出于evaluate
	evaluation: 评估训练的模块，基于episode的action_memory
	get_random_stockdata: 随机获取股票，基于十年的区段，获取个股，个股需满足高于一定程度的交易日，并以一定比例分割train与trade数据集
	                        其实象征性的创建一个trade数据集就OK了，目前还没有看见trade数据集的用处，因为episode的步数上限都是按train的长度
	                        限制的
	landmark：对获取到的个股进行标点，标记最佳买卖信号 # 已经写入evaluation中
	trainer: 训练
	trade_information_episode1.jpg:  evaluate的主要输出
	predict: 读取.model，创建agent，输入一条观测值，并可以进行决策
new_data_file:
	get_random_stockdata的主要输出
utils:
	存放强化学习智能体的工具
	env: 主要的几个函数： __init__, reset, step, get_sub_env,
	config: 包含1.股票技术指标列表，2.
	models: 返回一个stable_baseline3的某个强化学习算法实例
	backtest: ?

watch_list: 
	存放心仪的股票并可用于预测

