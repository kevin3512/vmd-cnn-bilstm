# config.py - 专门存放变量的文件
class Config:
    # VMD参数配置
    K=7
    alpha=500
    tau=1
    DC=1
    init=1
    tol=1e-05
    N=3000
    hasResidual=False  # 是否添加残差分量作为最后一个IMF
    vmd_enable=True  # 是否启用VMD参数调优
    vmd_single_model = True # 是否只使用单个模型进行VMD参数调优
    single_model = "rnn"  # 使用单个模型预测（需要enable=False生效），可选值: "cnn", "lstm", "cnn_lstm", "cnn_bilstm", "tcn", "rnn"
    
    # 数据集配置
    file_name = "河南数据集.xlsx"
    date_col = "date"
    value_col = "value"
    window = 24
    nrows = 0   # 设为0表示所有行
    test_percent = 0.2
    train_percent = 0.8

    # 训练配置
    epochs = 500
    lr = 0.001   #学习率

    #模型选择
    cnn_bilstm_threshold = 0.1 #如：大于0.15选择CNN-BiLSTM
    cnn_lstm_threshold = 0.05  #如：大于0.05选择CNN-LSTM
    
    
