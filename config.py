# config.py - 专门存放变量的文件
class Config:
    # VMD参数配置
    K = 4
    alpha = 3000
    tau = 0
    DC = 0
    init = 1
    tol = 1e-7
    
    # 数据集配置
    date_col = "date"
    value_col = "value"
    window = 24
    nrows = 0   # 设为0表示所有行
    test_percent = 0.2
    train_percent = 0.8

    # 训练配置
    epochs = 3000
    lr = 0.001   #学习率
    
    
