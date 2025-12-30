import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from cnn import CNN, CNN_LSTM, RNN, CNN_BiLSTM, TCN, LSTM
import matplotlib.pyplot as plt
from config import Config
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import my_vmd
import metrics
import logging




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


def select_model(imf, window, vmd_single_model=False, single_model="cnn"):

    if vmd_single_model:  #只使用单个模型进行预测
        if single_model == "cnn":
            print(f"选择单个 CNN 模型")
            return CNN(window)
        elif single_model == "rnn":
            print(f"选择单个 RNN 模型")
            return RNN(window)  
        elif single_model == "lstm":
            print(f"选择单个 LSTM 模型")
            return LSTM(window)
        elif single_model == "cnn_lstm":
            print(f"选择单个 CNN-LSTM 模型")
            return CNN_LSTM(window)
        elif single_model == "cnn_bilstm":
            print(f"选择单个 CNN-BiLSTM 模型")
            return CNN_BiLSTM(window)
        else:
            raise ValueError(f"Unsupported single_model: {single_model}")
    else:  #根据阈值选择模型
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


def vmd_cnn_bilstm_pipeline(file_path, vmd_single_model=False, single_model="cnn"):
    print(f"加载数据并进行 VMD 分解...是否单模型处理VMD分量：{vmd_single_model} , 单模型为:{single_model}")
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
            model = select_model(imf, window, vmd_single_model, single_model)

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
    print(f"加载数据...使用单模型:{select_model}")
    series, scaler = load_data(file_path)

    imfs = my_vmd.vmd_decompose(series)
    window = Config.window

    predictions = []
    loss_records = []

    if(select_model == "cnn"):
        model = CNN(window)
    elif(select_model == "tcn"):
        model = TCN(window)
    elif (select_model == "rnn"):
        model = RNN(window)
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


def evaluate_configs(configs, file_path, save_path="configs_comparison.png"):
    """Run multiple Config variants and plot their predictions on one figure.

    configs: list of dicts, each dict contains Config attributes to set and optional 'label' for legend
    """
    results = []

    # determine keys to restore
    keys = set()
    for c in configs:
        keys.update(k for k in c.keys() if k != 'label')

    original = {k: getattr(Config, k) if hasattr(Config, k) else None for k in keys}

    for c in configs:
        label = c.get('label', str(c))
        # apply temporary config
        for k, v in c.items():
            if k == 'label':
                continue
            setattr(Config, k, v)
        print(f"Running config: {label} -> { {k:v for k,v in c.items() if k!='label'} }")
        if c.get('vmd_enable', False):
            c_vmd_single_mode = c.get('vmd_single_model')
            c_single_model = c.get('single_model')
            y_true, y_pred, _ = vmd_cnn_bilstm_pipeline(file_path, c_vmd_single_mode, c_single_model)
        else:
            sel = c.get('single_model', "cnn")
            y_true, y_pred, _ = no_cmd_pipeline(file_path, select_model=sel)
        results.append((label, y_pred))

    # restore original config values
    for k, v in original.items():
        if v is None and hasattr(Config, k):
            delattr(Config, k)
        else:
            setattr(Config, k, v)


    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler('多模型对比实验性能结果.txt', encoding='utf-8'),  # 输出到文件
            logging.StreamHandler()  # 输出到控制台
        ]
    )
    # plot all on same figure
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='真实值', color='black', linewidth=2)
    for label, pred in results:
        plt.plot(pred, '--', label=label)
        logging.info(f"{label} 模型参数如下：{metrics.evaluate(y_true, pred)}")
        logging.info("\n--------------------------------------------------\n")
        
    plt.title('不同 Config 配置预测对比')
    plt.xlabel('时间步')
    plt.ylabel('碳排放值')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    #删除所有.png文件
    delete_all_png_files()

    evaluate_configs(Config.compare_models, Config.file_name, save_path="多模型对比结果图片.png")

