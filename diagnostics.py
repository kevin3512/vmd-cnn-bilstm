import numpy as np
import matplotlib.pyplot as plt
from config import Config
import main


def compute_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    bias = np.mean(y_pred) - np.mean(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    return {"RMSE": rmse, "MAE": mae, "Bias": bias, "MAPE(%)": mape}


if __name__ == '__main__':
    file_path = "安徽数据集.xlsx"

    series, scaler = main.load_data(file_path)
    imfs = main.vmd_decompose(series, K=Config.K)

    window = Config.window
    predictions = []
    per_imf_stats = []

    print(f"scaler.data_min_: {scaler.data_min_}, scaler.data_max_: {scaler.data_max_}")

    for idx, imf in enumerate(imfs):
        model = main.select_model(imf, window)
        pred, loss_hist = main.train_and_predict(imf, model, window)
        predictions.append(pred)

        # 对应的真实值（缩放后）
        y_true_imf_scaled = imf[-len(pred):]

        # 反归一化单个 IMF 的预测与真实（使用整体 scaler）
        try:
            y_true_imf_inv = scaler.inverse_transform(y_true_imf_scaled.reshape(-1, 1)).flatten()
            pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"反归一化失败: {e}")
            y_true_imf_inv = y_true_imf_scaled
            pred_inv = pred

        stats = compute_metrics(y_true_imf_inv, pred_inv)
        stats.update({"IMF_index": idx + 1, "IMF_std_scaled": float(np.std(imf))})
        per_imf_stats.append(stats)

        print(f"IMF-{idx+1} | std(scaled)={np.std(imf):.4f} | RMSE={stats['RMSE']:.6f} | MAE={stats['MAE']:.6f} | Bias={stats['Bias']:.6f} | MAPE%={stats['MAPE(%)']:.4f}")

        # 保存 IMF 预测图（覆盖 main.py 的图名不会影响）
        plt.figure(figsize=(6, 2.5))
        plt.plot(y_true_imf_inv, label='True (inv)')
        plt.plot(pred_inv, '--', label='Pred (inv)')
        plt.title(f"IMF-{idx+1} Diagnostic")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"diag_imf_{idx+1}.png", dpi=150)
        plt.close()

    # 合成最终预测
    final_pred_scaled = np.sum(predictions, axis=0)
    y_true_scaled = series[-len(final_pred_scaled):]

    # 检查缩放边界
    print(f"final_pred_scaled min/max: {final_pred_scaled.min():.6f}/{final_pred_scaled.max():.6f}")
    print(f"y_true_scaled min/max: {y_true_scaled.min():.6f}/{y_true_scaled.max():.6f}")

    final_pred = scaler.inverse_transform(final_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    overall_stats = compute_metrics(y_true, final_pred)
    print("Overall:")
    print(f"RMSE={overall_stats['RMSE']:.6f} | MAE={overall_stats['MAE']:.6f} | Bias={overall_stats['Bias']:.6f} | MAPE%={overall_stats['MAPE(%)']:.4f}")

    # 残差分析图
    residuals = final_pred - y_true
    plt.figure(figsize=(8, 3))
    plt.plot(residuals)
    plt.title('Residuals (final_pred - y_true)')
    plt.tight_layout()
    plt.savefig('diag_residuals.png', dpi=150)
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.hist(residuals, bins=30)
    plt.title('Residuals Histogram')
    plt.tight_layout()
    plt.savefig('diag_residuals_hist.png', dpi=150)
    plt.close()

    # 保存 per-IMF 统计到文件
    with open('diag_per_imf_stats.txt', 'w', encoding='utf-8') as f:
        for s in per_imf_stats:
            f.write(str(s) + '\n')

    print('诊断完成，已保存 diag_imf_*.png, diag_residuals.png, diag_residuals_hist.png, diag_per_imf_stats.txt')
