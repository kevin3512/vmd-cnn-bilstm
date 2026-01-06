import os
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
import metrics
import numpy as np



plt.rcParams['font.sans-serif'] = ['SimHei']   # 中文
plt.rcParams['axes.unicode_minus'] = False     # 负号

def show_all_models_pred(
        file_name,
        sheet_name,
        true_col="TRUE_VALUE"
):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"{file_name} 不存在")

    df = pd.read_excel(file_name, sheet_name=sheet_name)

    if true_col not in df.columns:
        raise ValueError(f"真实值列 '{true_col}' 不存在")

    plt.figure(figsize=(12, 6))

    # 先画真实值
    y_true_full = df[true_col].values
    plt.plot(y_true_full, label=true_col, linewidth=2)

    # 为每条曲线准备不同的虚线样式（dash lengths）
    dash_patterns = [
        [10, 5],
        [2, 2],
        [5, 1, 1, 1],
        [1, 5],
        [5, 2, 2, 2],
        [10, 3, 1, 3],
        [3, 3, 1, 3],
    ]

    for idx, col in enumerate(df.columns):
        if col == true_col:
            continue

        y_pred = df[col]

        # 有效索引（同时非 NaN）
        valid_mask = (~pd.isna(y_pred)) & (~pd.isna(df[true_col]))

        if valid_mask.sum() == 0:
            print(f"⚠️ 列 '{col}' 无有效数据，跳过")
            continue

        y_true = df.loc[valid_mask, true_col].values
        y_pred = df.loc[valid_mask, col].values

        # 绘图：先画线再设置 dash pattern，以确保每条线都是虚线但样式不同
        line, = plt.plot(
            np.where(valid_mask, df[col], np.nan),
            linestyle='--',
            label=col
        )
        pattern = dash_patterns[idx % len(dash_patterns)]
        line.set_dashes(pattern)

        print(f"模型 '{col}' 性能指标:")
        metrics.evaluate(y_true, y_pred)

    plt.xlabel("时间步 / 样本点")
    plt.ylabel("预测值")
    plt.title("不同模型预测结果对比")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 指定要绘制的列名列表

    show_all_models_pred(Config.model_predict_file, '模型预测值')