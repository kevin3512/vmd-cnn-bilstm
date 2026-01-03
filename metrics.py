import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score


def evaluate(y_true, y_pred):
    """评估模型性能"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n=== 模型评估结果 ===")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"R2 (决定系数): {r2:.4f}")
    return {
        'RMSE': rmse,
        'MAPE': mape,
        'MAE': mae,
        'R2': r2
    }

def smape(y_true, y_pred):
    """
    计算对称平均绝对百分比误差 (sMAPE)。
    
    参数:
        y_true : 真实值数组
        y_pred : 预测值数组
    
    返回:
        sMAPE 值，以百分比形式表示
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # sMAPE 的核心公式
    # 分母是真实值和预测值的绝对值之和，避免了传统MAPE在真实值接近0时放大的问题
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    
    # 为避免分母为零的情况，可以添加一个极小的常数 (epsilon)，例如1e-10
    epsilon = 1e-10
    smape_value = np.mean(numerator / (denominator + epsilon)) * 100
    
    return smape_value

def evaluate_smape(y_true, y_pred):
    """评估模型性能"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    s_mape = smape(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n=== 模型评估结果 ===")
    print(f"RMSE (均方根误差): {rmse:.4f}")
    print(f"SMAPE (平均绝对百分比误差): {s_mape:.2f}%")
    print(f"MAE (平均绝对误差): {mae:.4f}")
    print(f"R2 (决定系数): {r2:.4f}")
    return {
        'RMSE': rmse,
        'SMAPE': s_mape,
        'MAE': mae,
        'R2': r2
    }


def save_evaluation(y_true, y_pred, config_obj=None, out_dir='results', filename=None):
    """
    计算评估指标并将结果连同 `config_obj` 的所有配置项保存为 JSON 文件。

    参数:
        y_true: 真实值（列表或ndarray）
        y_pred: 预测值（列表或ndarray）
        config_obj: 配置对象或模块（例如从 `config import Config`，或字典）。
                    如果为 None，会尝试从 `config` 模块中导入 `Config` 类并读取其属性。
        out_dir: 输出目录
        filename: 输出文件名（可选），默认为 evaluation_YYYYmmdd_HHMMSS.json

    返回:
        保存文件的路径
    """
    # 计算指标
    my_metrics = evaluate(np.array(y_true), np.array(y_pred))

    # 提取配置项
    config_dict = None
    if config_obj is None:
        try:
            import config as _config_mod
            if hasattr(_config_mod, 'Config'):
                cfg = getattr(_config_mod, 'Config')
                # 支持类和实例
                if isinstance(cfg, type):
                    config_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith('__') and not callable(v)}
                else:
                    config_dict = {k: v for k, v in vars(cfg).items() if not k.startswith('__') and not callable(v)}
        except Exception:
            config_dict = {}
    else:
        # 支持字典、类、模块或实例
        if isinstance(config_obj, dict):
            config_dict = config_obj
        elif isinstance(config_obj, type):
            config_dict = {k: v for k, v in config_obj.__dict__.items() if not k.startswith('__') and not callable(v)}
        else:
            # 可能是模块或实例
            try:
                config_dict = {k: v for k, v in vars(config_obj).items() if not k.startswith('__') and not callable(v)}
            except Exception:
                config_dict = {}

    # 准备要保存的内容（以追加模式写入纯文本，使用多行'===='分隔每次记录）
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if filename is None:
        filename = 'evaluation_log.txt'

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    sep = '\n================================================================\n================================================================\n'
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(sep)
        f.write(f'Timestamp: {timestamp}\n')
        f.write('--------------- Metrics ---------------\n')
        for k, v in my_metrics.items():
            try:
                val = float(v)
                f.write(f'{k}: {val:.6f}\n')
            except Exception:
                f.write(f'{k}: {v}\n')

        f.write('--------------- Config ---------------\n')
        if config_dict:
            for k, v in config_dict.items():
                f.write(f'{k}: {v}\n')
        else:
            f.write('No config found or could not import config.Config\n')

        f.write('\n')

    print(f"评估结果已追加到: {out_path}")
    return out_path

import numpy as np


def save_imf_evaluation(y_true, y_pred, imf_index, config_obj=None, out_dir='results', filename=None):
    # 计算指标
    my_metrics = evaluate_smape(np.array(np.abs(y_true)), np.array(np.abs(y_pred)))
    # 准备要保存的内容（以追加模式写入纯文本，使用多行'===='分隔每次记录）
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if filename is None:
        filename = 'evaluation_log.txt'

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    sep = '\n================================================================\n================================================================\n'
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write(sep)
        f.write(f'Timestamp: {timestamp}\n')
        f.write(f'IMF Index: {imf_index}\n')
        f.write('--------------- Metrics ---------------\n')
        for k, v in my_metrics.items():
            try:
                val = float(v)
                f.write(f'{k}: {val:.6f}\n')
            except Exception:
                f.write(f'{k}: {v}\n')
        f.write('\n')

    print(f"评估结果已追加到: {out_path}")
    return out_path