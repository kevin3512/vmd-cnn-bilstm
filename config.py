# config.py - 专门存放变量的文件
class Config:
    ####################################----main_run_all.py 中的配置----####################################

    # 多个模型对比试验
    RUN_COMPARE = True  # 改为 True 来运行多模型放到一起对比
    # 实验一：多个单模型和本文模型对比
    # compare_models = [
    #         {"label":"单模型:RNN", "vmd_enable": False, "vmd_single_model": True, "single_model":"rnn"},
    #         {"label":"单模型:CNN", "vmd_enable": False, "vmd_single_model": True, "single_model":"cnn"},
    #         {"label":"单模型:LSTM", "vmd_enable": False, "vmd_single_model": True, "single_model":"lstm"},
    #         {"label":"单模型:BiLSTM", "vmd_enable": False, "vmd_single_model": True, "single_model":"cnn"},
    #         {"label":"本文模型", "vmd_enable": True, "vmd_single_model": False, "single_model":""}
    #     ]
    # # 实验二：组合模型和本文模型对比
    # compare_models = [
    #         {"label":"组合模型:CNN-LSTM", "vmd_enable": False, "vmd_single_model": True, "single_model":"cnn_lstm"},
    #         {"label":"组合模型:CNN-BiLSTM", "vmd_enable": False, "vmd_single_model": True, "single_model":"cnn_bilstm"},
    #         {"label":"本文模型", "vmd_enable": True, "vmd_single_model": False, "single_model":""}
    #     ]
    # 实验三：固定VMD模型和本文模型对比
    compare_models = [
            {"label":"组合模型:CNN-LSTM", "vmd_enable": False, "vmd_single_model": True, "single_model":"cnn_lstm"},
            {"label":"组合模型:CNN-BiLSTM", "vmd_enable": False, "vmd_single_model": True, "single_model":"cnn_bilstm"},
            {"label":"VMD模型:CNN", "vmd_enable": True, "vmd_single_model": True, "single_model":"cnn"},
            {"label":"VMD模型:CNN-LSTM", "vmd_enable": True, "vmd_single_model": True, "single_model":"cnn_lstm"},
            {"label":"VMD模型:CNN-BiLSTM", "vmd_enable": True, "vmd_single_model": True, "single_model":"cnn_bilstm"},
            {"label":"本文模型", "vmd_enable": True, "vmd_single_model": False, "single_model":""}
        ]
    
    ############################################################################################################




    ####################################----main.py 中的配置----####################################
    model_predict_file = "result/模型运行结果.xlsx"  # 模型预测结果保存文件名

    # VMD参数配置
    K=7
    alpha=500
    tau=1
    DC=1
    init=1
    tol=1e-05
    N=3000
    hasResidual=True  # 是否添加残差分量作为最后一个IMF
    vmd_enable=True  # 是否启用VMD参数调优
    vmd_single_model = False # 是否只使用单个模型进行VMD参数调优
    single_model = "cnn_bilstm"  # 使用单个模型预测（需要enable=False生效），可选值: "cnn", "lstm", "cnn_lstm", "cnn_bilstm", "tcn", "rnn"
    
    # 数据集配置
    file_name = "data_set/河北数据集.xlsx"
    date_col = "date"
    value_col = "value"
    window = 24
    nrows = 0   # 设为0表示所有行
    test_percent = 0.2
    train_percent = 0.8

    # 训练配置
    epochs = 500
    lr = 0.001   #学习率

    # 模型选择阈值（基于 IMF 预测复杂度：谱熵，范围 0-1）
    complexity_bilstm_threshold = 0.6  # 大于此值选择 CNN-BiLSTM
    complexity_lstm_threshold = 0.4   # 大于此值选择 CNN-LSTM
    # 兼容旧配置（基于标准差） - 如需改回，请使用 std_* 变量
    std_bilstm_threshold = 0.1
    std_lstm_threshold = 0.05

    ############################################################################################################
    
    
