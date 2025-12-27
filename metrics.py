import numpy as np
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
    print(f"R² (决定系数): {r2:.4f}")

    return {
        'RMSE': rmse,
        'MAPE': mape,
        'MAE': mae,
        'R2': r2
    }