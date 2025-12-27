# config.py - 专门存放变量的文件
class Config:
    # VMD参数配置
    K=4
    alpha=500
    tau=1
    DC=1
    init=1
    tol=1e-05
    N=3000
    
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
    
    
