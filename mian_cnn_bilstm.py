import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from cnn import CNN, CNN_LSTM, CNN_BiLSTM, TCN
import matplotlib.pyplot as plt
from config import Config
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import my_vmd
import metrics
import cnn




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


def vmd_cnn_bilstm_pipeline(file_path):
    series, scaler = load_data(file_path)

    window = Config.window

    predictions = []
    loss_records = []

    # model = CNN_BiLSTM(window)
    # model = TCN(window=window, channels=[16, 32, 64], kernel_size=3)
    model = CNN_LSTM(window)

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

    # 运行 VMD-CNN-BiLSTM 模型获取预测结果
    y_true, y_pred, loss_records = vmd_cnn_bilstm_pipeline(Config.file_name)

    # 计算并打印回归指标
    metrics.evaluate(y_true=y_true, y_pred=y_pred)

    # 绘制预测结果对比图
    plot_prediction(y_true, y_pred)
    for i, loss_hist in enumerate(loss_records):
        plot_loss_curve(
            loss_hist,
            save_path=f"loss_imf_{i+1}.png",
            title=f"IMF-{i+1} 训练损失曲线"
        )
