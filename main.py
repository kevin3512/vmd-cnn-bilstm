import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from vmdpy import VMD
import torch
import torch.nn as nn
from cnn import CNN, CNN_LSTM, CNN_BiLSTM
import matplotlib.pyplot as plt
from config import Config
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os




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

def vmd_decompose(signal, K=Config.K):
    alpha = Config.alpha
    tau = Config.tau
    DC = Config.DC
    init = Config.init
    tol = Config.tol

    imfs, _, _ = VMD(
        signal,
        alpha=alpha,
        tau=tau,
        K=K,
        DC=DC,
        init=init,
        tol=tol
    )
    return imfs


def create_dataset(series, window=Config.window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    return np.array(X), np.array(y)


def select_model(imf, window):
    std = np.std(imf)

    if std > 0.15:
        return CNN_BiLSTM(window)
    elif std > 0.05:
        return CNN_LSTM(window)
    else:
        return CNN(window)

def train_and_predict(series, model, window, epochs=Config.epochs):
    X, y = create_dataset(series, window)

    split = int(len(X) * Config.train_percent)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    loss_fn = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X_train).squeeze()
        loss = loss_fn(output, y_train)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    preds = model(X_test).detach().numpy().flatten()

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


def vmd_cnn_bilstm_pipeline(file_path):
    series, scaler = load_data(file_path)

    imfs = vmd_decompose(series, K=Config.K)
    #vmd全部数据
    plot_vmd_imfs(series, imfs,save_path="whole_vmd_imfs.png")
    #预测的vmd分解数据
    split = int(len(imfs[1]) * Config.test_percent)
    plot_vmd_imfs(series[-split:], imfs[:, -split:],save_path="test_vmd_imfs.png")
    window = Config.window

    predictions = []
    loss_records = []

    for idx, imf in enumerate(imfs):
        model = select_model(imf, window)
        pred, loss_hist = train_and_predict(imf, model, window)
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
    y_true, y_pred, loss_records = vmd_cnn_bilstm_pipeline("安徽数据集.xlsx")

    # 计算并打印回归指标
    metrics = regression_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 绘制预测结果对比图
    plot_prediction(y_true, y_pred)
    for i, loss_hist in enumerate(loss_records):
        plot_loss_curve(
            loss_hist,
            save_path=f"loss_imf_{i+1}.png",
            title=f"IMF-{i+1} 训练损失曲线"
        )
