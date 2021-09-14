------readme------
learn:
    action_abstract: 将action抽象为三点输出
    new_train_file,,存放每个episode的action_memory
	train_file：存放训练后的模型
	actions_signal.jpg: landmark 标点后的 图，对买卖信号的标记，输出于evaluate
	evaluation: 评估训练的模块，基于episode的action_memory
	get_random_stockdata: 随机获取股票，基于十年的区段，获取个股，个股需满足高于一定程度的交易日
	landmark：对获取到的个股进行标点，标记最佳买卖信号
	trainer: 训练
	trade_information_episode1.jpg:  evaluate的主要输出
	predict: 读取.model，创建agent，输入一条观测值，并可以进行决策
new_data_file:
	get_random_stockdata的主要输出
utils:
	存放强化学习智能体的工具

watch_list: 
	存放心仪的股票并可用于预测

