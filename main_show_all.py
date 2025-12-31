    

import os
from matplotlib import pyplot as plt
import pandas as pd
import metrics
from config import Config

plt.rcParams['font.sans-serif'] = ['SimHei']   # 中文
plt.rcParams['axes.unicode_minus'] = False     # 负号


def show_all_models_pred(file_name, first_row_list):
    """
    读取 Excel 中指定列并绘制预测结果
    :param file_name: Excel 文件名
    :param first_row_list: 需要绘制的列名列表
    """
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"{file_name} 不存在")

    df = pd.read_excel(file_name)

    plt.figure(figsize=(12, 6))

    for col in first_row_list:
        if col not in df.columns:
            print(f"⚠️ 警告：列 '{col}' 不存在，已跳过")
            continue
        plt.plot(df[col], label=col)
        if col != "true_value":
            print(f"\n\n\n\n\n模型 '{col}' 性能指标:")
            metrics.evaluate(y_true=df["true_value"], y_pred=df[col])

    plt.xlabel("时间步 / 样本点")
    plt.ylabel("预测值")
    plt.title("不同模型预测结果对比")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 指定要绘制的列名列表
    
    show_all_models_pred(Config.model_predict_file, Config.columns_to_plot)