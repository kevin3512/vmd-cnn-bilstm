import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from cnn import CNN, CNN_LSTM, CNN_BiLSTM, TCN, LSTM
import matplotlib.pyplot as plt
from config import Config
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import my_vmd
import metrics




plt.rcParams['font.sans-serif'] = ['SimHei']   # 中文
plt.rcParams['axes.unicode_minus'] = False     # 负号




def load_data(file_path):
    if(Config.nrows == 0):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_excel(file_path, nrows=Config.nrows)
    print(f"读取数据长度：{len(df)}")
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df = df.sort_values(Config.date_col)

    values = df[Config.value_col].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    values = scaler.fit_transform(values).flatten()

    return values, scaler

def create_dataset(series, window=Config.window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)


def select_model(imf, window):
    std = np.std(imf)

    if std > Config.cnn_bilstm_threshold:
        print(f"选择 CNN-BiLSTM 模型，IMF 标准差: {std:.4f}")
        return CNN_BiLSTM(window)
    elif std > Config.cnn_lstm_threshold:
        print(f"选择 CNN-LSTM 模型，IMF 标准差: {std:.4f}")
        return CNN_LSTM(window)
    else:
        print(f"选择 CNN 模型，IMF 标准差: {std:.4f}")
        return CNN(window)

def train_and_predict(series, model, window, epochs=Config.epochs, per_imf_normalize=False, batch_size=32, loss_type='mse'):
    """
    series: 1D numpy array (already globally scaled if applicable)
    per_imf_normalize: if True, normalize this series to zero-mean unit-std for training,
                       then inverse the predictions back to the input series scale before returning.
    """
    series_used = series.copy()
    mu = 0.0
    sigma = 1.0
    if per_imf_normalize:
        mu = np.mean(series_used)
        sigma = np.std(series_used) if np.std(series_used) > 0 else 1.0
        series_used = (series_used - mu) / sigma

    X, y = create_dataset(series_used, window)

    split = int(len(X) * Config.train_percent)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    if loss_type == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_type == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_type == 'huber':
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    loss_history = []

    num_train = X_train.size(0)
    for epoch in range(epochs):
        epoch_losses = []
        # iterate by sequential mini-batches (no shuffle for time series)
        for start in range(0, num_train, batch_size):
            end = start + batch_size
            xb = X_train[start:end]
            yb = y_train[start:end]

            optimizer.zero_grad()
            output = model(xb).squeeze()
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        loss_history.append(avg_loss)
        if (epoch + 1) % 100 == 0 or epoch == 0: # 100的倍数打印一次
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    preds = model(X_test).detach().numpy().flatten()

    if per_imf_normalize:
        preds = preds * sigma + mu

    return preds, loss_history

def plot_vmd_imfs(signal, imfs, save_path=None):
    """
    signal: 原始序列
    imfs: shape = (K, N)
    """
    K = imfs.shape[0]

    plt.figure(figsize=(10, 2 * (K + 1)))

    # 原始信号
    plt.subplot(K + 1, 1, 1)
    plt.plot(signal, color='black')
    plt.title("Original Carbon Emission Series")
    plt.grid(alpha=0.3)

    # IMF 分量
    for i in range(K):
        plt.subplot(K + 1, 1, i + 2)
        plt.plot(imfs[i])
        plt.title(f"IMF-{i + 1}")
        plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_imf_prediction(imf_true, imf_pred, imf_index, save_path=None):
    plt.figure(figsize=(8, 3))

    plt.plot(imf_true, label="True IMF", linewidth=2)
    plt.plot(imf_pred, '--', label="Predicted IMF", linewidth=2)

    plt.title(f"IMF-{imf_index} Prediction Result")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}_imf_{imf_index}_prediction.png", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_true_vs_vmd_sum(signal, imfs, test_percent=Config.test_percent, save_path=None):
    """
    Plot test-set true values vs sum of VMD IMF components over the test period.

    signal: full original series (1D numpy array)
    imfs: array shape (K, N)
    test_percent: fraction of data used as test (e.g., 0.2)
    """
    K, N = imfs.shape
    split = int(N * test_percent)

    if split <= 0:
        raise ValueError("test_percent results in zero-length test set")

    y_true = signal[-split:]
    vmd_sum = np.sum(imfs[:, -split:], axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Test True', linewidth=2)
    plt.plot(vmd_sum, '--', label='VMD Sum (test)', linewidth=2)
    plt.title('Test True vs VMD Components Sum')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def vmd_cnn_bilstm_pipeline(file_path):
    series, scaler = load_data(file_path)

    imfs = my_vmd.vmd_decompose(series)
    #vmd全部数据
    plot_vmd_imfs(series, imfs,save_path="whole_vmd_imfs.png")
    #预测的vmd分解数据
    split = int(len(imfs[1]) * Config.test_percent)
    plot_vmd_imfs(series[-split:], imfs[:, -split:],save_path="test_vmd_imfs.png")
    window = Config.window

    predictions = []
    loss_records = []

    for idx, imf in enumerate(imfs):
        # 使用单独新增的 TCN 模型来预测 IMF-1（索引 0），不替换其他 IMF 的原有选择逻辑
        # if idx == (len(imfs) - 1):
        if idx == -1:  #不使用TCN模型
            model = TCN(window)
            print(f"IMF-{idx+1}: 使用 TCN 模型预测")
        else:
            print(f"IMF-{idx+1}: 模型选择如下：")
            model = select_model(imf, window)

        # 对每个 IMF 先做去均值标准化再训练（防止不同 IMF 幅度差异导致偏差）
        pred, loss_hist = train_and_predict(imf, model, window, per_imf_normalize=True, batch_size=32, loss_type='huber')
        predictions.append(pred)
        loss_records.append(loss_hist)
        plot_imf_prediction(
            imf_true=imf[-len(pred):],
            imf_pred=pred,
            imf_index=idx + 1,
            save_path="imf_predictions"
        )

    final_pred = np.sum(predictions, axis=0)

    y_true = series[-len(final_pred):]

    final_pred = scaler.inverse_transform(final_pred.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    return y_true, final_pred, loss_records

def no_cmd_pipeline(file_path, select_model="cnn"):
    series, scaler = load_data(file_path)

    imfs = my_vmd.vmd_decompose(series)
    window = Config.window

    predictions = []
    loss_records = []

    if(select_model == "cnn"):
        model = CNN(window)
    elif(select_model == "lstm"):
        model = LSTM(window)
    elif(select_model == "cnn_lstm"):
        model = CNN_LSTM(window)
    elif(select_model == "cnn_bilstm"):
        model = CNN_BiLSTM(window)

    # 对每个 IMF 先做去均值标准化再训练（防止不同 IMF 幅度差异导致偏差）
    pred, loss_hist = train_and_predict(series, model, window, per_imf_normalize=True, batch_size=32, loss_type='huber')
    predictions.append(pred)
    loss_records.append(loss_hist)
    final_pred = np.sum(predictions, axis=0)

    y_true = series[-len(final_pred):]

    final_pred = scaler.inverse_transform(final_pred.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    return y_true, final_pred, loss_records

def regression_metrics(y_true, y_pred, eps=1e-8):
    """
    y_true, y_pred: shape (N,)
    """

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    map_ = 100 - mape

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "MAP(%)": map_
    }

def plot_prediction(y_true, y_pred, title="碳排放预测结果对比"):
    plt.figure(figsize=(10, 5))

    plt.plot(y_true, label='真实值', linewidth=2)
    plt.plot(y_pred, label='预测值', linestyle='--', linewidth=2)

    plt.xlabel("时间步")
    plt.ylabel("碳排放值")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("碳排放预测结果对比", dpi=300)
    plt.show()

def plot_loss_curve(loss_history, save_path, title):
    plt.figure(figsize=(7, 4))

    plt.plot(loss_history, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def delete_all_png_files():
    """
    删除当前目录下所有 .png 文件
    """
    current_dir = os.getcwd()  # 获取当前目录
    
    png_count = 0  # 统计删除的文件数量
    
    for filename in os.listdir(current_dir):
        if filename.endswith('.png'):
            file_path = os.path.join(current_dir, filename)
            
            try:
                os.remove(file_path)
                print(f"已删除: {filename}")
                png_count += 1
            except Exception as e:
                print(f"删除失败 {filename}: {e}")
    
    print(f"总共删除了 {png_count} 个 .png 文件")



if __name__ == '__main__':
    #删除所有.png文件
    delete_all_png_files()

    # 调用 VMD 参数自动调优（必要时会写回 config.py）
    # try:
    #     print('Running VMD parameter tuner...')
    #     from vmd_param_tuner import tune_vmd_params
    #     best_params, best_mape = tune_vmd_params(max_evals=200, early_stop_mape=0.1)
    #     print(f'VMD tuner finished. best_mape={best_mape:.6f}, best_params={best_params}')
    # except Exception as e:
    #     print('VMD tuner failed:', e)

    # 运行 VMD-CNN-BiLSTM 模型获取预测结果
    if(Config.vmd_enable):
        print("使用 VMD-CNN-BiLSTM 组合模型进行预测...")
        y_true, y_pred, loss_records = vmd_cnn_bilstm_pipeline(Config.file_name)
    else:
        print("使用单模型进行预测...")
        y_true, y_pred, loss_records = no_cmd_pipeline(Config.file_name, Config.single_model)

    # 计算并打印保存回归指标
    # metrics.evaluate(y_true=y_true, y_pred=y_pred)
    metrics.save_evaluation(y_true, y_pred, filename="模型性能指标保存.txt", out_dir="result")

    # 绘制预测结果对比图
    plot_prediction(y_true, y_pred)
    for i, loss_hist in enumerate(loss_records):
        plot_loss_curve(
            loss_hist,
            save_path=f"loss_imf_{i+1}.png",
            title=f"IMF-{i+1} 训练损失曲线"
        )
