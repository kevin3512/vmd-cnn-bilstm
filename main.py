import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from cnn import CNN, CNN_LSTM, RNN, BiLSTM, CNN_BiLSTM, TCN, LSTM
import matplotlib.pyplot as plt
from config import Config
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os
import metrics
import my_vmd
from metrics import evaluate
import time
import json




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


def imf_spectral_entropy(imf, eps=1e-12):
    """
    Compute normalized spectral entropy of the IMF (approximately in range 0-1).
    Higher values indicate a more complex / less predictable series.
    Uses the power spectrum from the real FFT and normalizes entropy by log(n_bins).
    """
    imf = np.asarray(imf).astype(np.float64)
    if imf.size < 2:
        return 0.0
    ps = np.abs(np.fft.rfft(imf))**2
    ps_sum = ps.sum()
    if ps_sum <= 0:
        return 0.0
    p = ps / (ps_sum + eps)
    entropy = -np.sum(p * np.log(p + eps))
    max_entropy = np.log(len(p))
    return float(entropy / (max_entropy + eps))


def select_model(imf, window):

    if Config.vmd_single_model:  #只使用单个模型进行预测
        if Config.single_model == "CNN":
            print(f"选择单个 CNN 模型")
            return CNN(window), Config.single_model
        elif Config.single_model == "RNN":
            print(f"选择单个 RNN 模型")
            return RNN(window), Config.single_model
        elif Config.single_model == "LSTM":
            print(f"选择单个 LSTM 模型")
            return LSTM(window), Config.single_model
        elif Config.single_model == "BiLSTM":
            print(f"选择单个 BiLSTM 模型")
            return BiLSTM(window), Config.single_model
        elif Config.single_model == "CNN-LSTM":
            print(f"选择单个 CNN-LSTM 模型")
            return CNN_LSTM(window), Config.single_model
        elif Config.single_model == "CNN-BiLSTM":
            print(f"选择单个 CNN-BiLSTM 模型")
            return CNN_BiLSTM(window), Config.single_model
        else:
            raise ValueError(f"Unsupported single_model: {Config.single_model}")
    else:  # 根据 IMF 预测复杂度选择模型（谱熵）
        complexity = imf_spectral_entropy(imf)
        st = np.std(imf)
        if st > Config.std_bilstm_threshold:
            print(f"选择高频模型{Config.high_mode}，IMF 复杂度: {complexity:.4f}， 标准差: {st:.4f}")
            return CNN_BiLSTM(window), Config.high_mode.upper()
        elif st > Config.std_lstm_threshold:
            print(f"选择中频模型{Config.mide_mode}，IMF 复杂度: {complexity:.4f}， 标准差: {st:.4f}")
            return CNN_LSTM(window), Config.mide_mode.upper()
        else:
            print(f"选择低频模型{Config.low_mode}，IMF 复杂度: {complexity:.4f}， 标准差: {st:.4f}")
            return CNN(window), Config.low_mode.upper()

def get_model_by_name(name, window):
    if name == "CNN":
        return CNN(window)
    elif name == "RNN":
        return RNN(window)
    elif name == "LSTM":
        return LSTM(window)
    elif name == "BiLSTM":
        return BiLSTM(window)
    elif name == "CNN-LSTM":
        return CNN_LSTM(window)
    elif name == "CNN-BiLSTM":
        return CNN_BiLSTM(window)
    elif name == "TCN":
        return TCN(window)
    else:
        raise ValueError(f"Unsupported model name: {name}")

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
    train_start = time.time()
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

    train_end = time.time()
    train_time = train_end - train_start
    print(f"Training time: {train_time:.4f} seconds")

    inference_start = time.time()
    preds = model(X_test).detach().numpy().flatten()
    inference_end = time.time()
    inference_time = inference_end - inference_start
    print(f"Inference time: {inference_time:.6f} seconds")

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
    metrics.save_imf_evaluation(imf_true, imf_pred, imf_index, filename="分频性能指标保存.txt", out_dir="result")
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
    imf_preds = []
    imf_trues = []

    for idx, imf in enumerate(imfs):
        _, model_name = select_model(imf, window)
        # 对每个 IMF 先做去均值标准化再训练（防止不同 IMF 幅度差异导致偏差）
        # pred, _ = train_and_predict(imf, model, window, per_imf_normalize=True, batch_size=32, loss_type='huber')
        infer_start = time.time()
        pred = infer_imf_model(imf, model_zoo[(imf_idx, model_name)], window)
        infer_end = time.time()
        print(f"IMF-{idx + 1} 使用模型 {model_name} 推理时间: {infer_end - infer_start:.6f} 秒")
        # 保存推理时间
        metrics.save_imf_model_train_time(imf_idx=imf_idx, model_name=model_name, train_time=infer_start - infer_end, sheet_name="Infer_times", file_name=Config.model_predict_file)
        predictions.append(pred)
        imf_preds.append(pred)
        imf_trues.append(imf[-len(pred):])
        # 绘制每个 IMF 的预测结果对比图
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

    return y_true, final_pred, imf_preds, imf_trues

def no_cmd_pipeline(file_path, select_model="CNN"):
    series, scaler = load_data(file_path)
    # 对每个 IMF 先做去均值标准化再训练（防止不同 IMF 幅度差异导致偏差）
    # pred, _ = train_and_predict(series, model, window, per_imf_normalize=True, batch_size=32, loss_type='huber')
    pred = infer_imf_model(series, model_zoo[(-1, select_model)], Config.window)  #-1表示整体序列,没有进行vmd分解
    final_pred = np.sum(pred, axis=0)

    y_true = series[-len(final_pred):]

    final_pred = scaler.inverse_transform(final_pred.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

    return y_true, final_pred

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




def read_column_from_sheet(file_name, sheet_name, col_name):
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    if col_name not in df.columns:
        raise ValueError(f"Sheet [{sheet_name}] 中不存在列 [{col_name}]")
    return df[col_name].dropna().values


def save_pred_and_metrics_to_file(
        y_true,
        y_pred,
        model_name,
        file_name="模型运行结果.xlsx",
        pred_sheet="模型预测值",
        metrics_sheet="模型指标",
        y_true_col="TRUE_VALUE",
        imf_preds=None,
        imf_trues=None,
        imf_sheet_name="分频预测性能"
):
    """
    改造说明：
    - 当已有结果时不再抛错，而是覆盖原有数据 (替换列/行)
    - y_true / y_pred 只要有一个非 None，就保存到 `pred_sheet`
    - 只有当 y_true 和 y_pred 都非 None 时，才计算并保存指标到 `metrics_sheet`
    - 如果提供了 imf_preds（list of arrays） 和 imf_trues（list of arrays），
      会在 `imf_sheet_name` 中保存每个 IMF 的指标（MAPE），存在则覆盖
    """

    # ================== 1. 准备要写入的列 ==================
    if y_pred is not None:
        series_to_write = pd.Series(y_pred, name=model_name)
    elif y_true is not None:
        series_to_write = pd.Series(y_true, name=model_name)
    else:
        raise ValueError("y_true 和 y_pred 不能同时为 None")

    # ================== 2. 写入 / 覆盖预测值 Sheet ==================
    try:
        df_old = pd.read_excel(file_name, sheet_name=pred_sheet) if os.path.exists(file_name) else pd.DataFrame()
    except ValueError:
        # sheet 不存在
        df_old = pd.DataFrame()

    max_len = max(len(df_old), len(series_to_write))
    df_old = df_old.reindex(range(max_len))
    series_to_write = series_to_write.reindex(range(max_len))

    # 覆盖或新增列
    df_old[model_name] = series_to_write

    # 写回（替换整个 sheet，保证行为可预期）
    if os.path.exists(file_name):
        with pd.ExcelWriter(file_name, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_old.to_excel(writer, sheet_name=pred_sheet, index=False)
    else:
        with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
            df_old.to_excel(writer, sheet_name=pred_sheet, index=False)

    print(f"已写入/覆盖预测值列：{model_name}")

    # ================== 3. 指标计算（当且仅当 y_true 和 y_pred 都非 None） ==================
    if y_true is not None and y_pred is not None:

        if len(y_true) != len(y_pred):
            raise ValueError("y_true 与 y_pred 长度不一致")

        metrics = evaluate(y_true, y_pred)
        metrics_df_new = pd.DataFrame(metrics, index=[model_name])

        try:
            df_metrics_old = pd.read_excel(file_name, sheet_name=metrics_sheet, index_col=0)
        except ValueError:
            df_metrics_old = pd.DataFrame()

        # 覆盖或新增行：先移除已有同名行（若存在），再 concat 新行，避免在空 DataFrame 上直接赋值时报错
        df_metrics_old = df_metrics_old.drop(index=model_name, errors='ignore')
        df_metrics_old = pd.concat([df_metrics_old, metrics_df_new])

        # 写回（替换 sheet）
        if os.path.exists(file_name):
            with pd.ExcelWriter(file_name, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df_metrics_old.to_excel(writer, sheet_name=metrics_sheet)
        else:
            with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                df_metrics_old.to_excel(writer, sheet_name=metrics_sheet)

        print(f"模型 [{model_name}] 指标已计算并写入（覆盖/新增）")
    else:
        print(f"模型 [{model_name}] 未计算指标（y_true 或 y_pred 为 None）")

    # ================== 4. 分频预测性能（按 IMF 保存预测值与指标，存在则覆盖） ==================
    if imf_preds is not None and imf_trues is not None:
        if len(imf_preds) != len(imf_trues):
            raise ValueError("imf_preds 与 imf_trues 长度不一致")

        # 计算每个 IMF 的 MAPE（更稳健的定义，避免 y_true 中近零值导致百分比爆炸）：
        mape_vals = {}
        eps_local = 1e-8
        for i, (t, p) in enumerate(zip(imf_trues, imf_preds)):
            t_arr = np.array(t).flatten().astype(float)
            p_arr = np.array(p).flatten().astype(float)
            if t_arr.size == 0:
                mape = np.nan
            else:
                mae = np.mean(np.abs(t_arr - p_arr))
                denom = np.mean(np.abs(t_arr)) + eps_local
                mape = (mae / denom) * 100.0
            mape_vals[f"IMF{i+1}"] = float(mape)

        row = pd.Series(mape_vals, name=model_name)

        # 读取已存在的分频 sheet（如果有），并把本模型按行覆盖或追加
        try:
            df_imf_old = pd.read_excel(file_name, sheet_name=imf_sheet_name, index_col=0) if os.path.exists(file_name) else pd.DataFrame()
        except Exception:
            df_imf_old = pd.DataFrame()

        # 合并列（保证 IMF 列顺序为 IMF1, IMF2, ...）
        all_cols = sorted(set(df_imf_old.columns).union(row.index), key=lambda c: (int(c.replace('IMF','')) if c.startswith('IMF') and c[3:].isdigit() else c))
        df_imf_old = df_imf_old.reindex(columns=all_cols)
        row = row.reindex(all_cols)

        # 覆盖或新增行
        df_imf_old.loc[model_name] = row

        # 写回 sheet（替换原 sheet）
        if os.path.exists(file_name):
            with pd.ExcelWriter(file_name, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df_imf_old.to_excel(writer, sheet_name=imf_sheet_name)
        else:
            with pd.ExcelWriter(file_name, engine="openpyxl", mode="w") as writer:
                df_imf_old.to_excel(writer, sheet_name=imf_sheet_name)

        print(f"已写入/覆盖分频预测性能 (MAPE) 到 Sheet: {imf_sheet_name}")


def get_model_name_from_config():
    if Config.vmd_enable:
        if Config.vmd_single_model:
            return f"VMD_{Config.single_model.upper()}"
        else:
            return "本文模型"
    else:
        return Config.single_model.upper()
    
def build_model(model_name, window):
    model_name = model_name.upper()
    if model_name == "CNN":
        return CNN(window)
    elif model_name == "CNN-LSTM":
        return CNN_LSTM(window)
    elif model_name == "CNN-BILSTM":
        return CNN_BiLSTM(window)
    elif model_name == "LSTM":
        return LSTM(window)
    elif model_name == "BILSTM":
        return BiLSTM(window)
    elif model_name == "RNN":
        return RNN(window)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def train_imf_model(
        imf,
        imf_idx,
        model_name,
        window,
        epochs,
        batch_size,
        loss_type="huber",
        per_imf_normalize=True
):
    """
    针对 (IMF_idx, model_name) 这一组合进行一次完整训练
    """

    series = imf.copy()
    mu, sigma = 0.0, 1.0

    if per_imf_normalize:
        mu = np.mean(series)
        sigma = np.std(series) if np.std(series) > 0 else 1.0
        series = (series - mu) / sigma

    X, y = create_dataset(series, window)
    split = int(len(X) * Config.train_percent)

    X_train = torch.tensor(X[:split], dtype=torch.float32)
    y_train = torch.tensor(y[:split], dtype=torch.float32)

    model = build_model(model_name, window)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)

    loss_fn = {
        "mse": nn.MSELoss(),
        "mae": nn.L1Loss(),
        "huber": nn.SmoothL1Loss()
    }[loss_type]

    loss_history = []
    model.train()

    for epoch in range(epochs):
        losses = []
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i + batch_size]
            yb = y_train[i:i + batch_size]

            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        loss_history.append(float(np.mean(losses)))
        if epoch == 0 or (epoch + 1) % 100 == 0:
            print(
                f"[TRAIN] IMF-{imf_idx+1} | {model_name} | "
                f"Epoch {epoch+1}/{epochs} | Loss={loss_history[-1]:.6f}"
            )

    return {
        "model": model,
        "mu": mu,
        "sigma": sigma,
        "loss_history": loss_history,
        "imf_idx": imf_idx,
        "model_name": model_name
    }


def infer_imf_model(imf, trained_bundle, window):
    series = imf.copy()

    mu = trained_bundle["mu"]
    sigma = trained_bundle["sigma"]

    series = (series - mu) / sigma

    X, _ = create_dataset(series, window)
    split = int(len(X) * Config.train_percent)
    X_test = torch.tensor(X[split:], dtype=torch.float32)

    model = trained_bundle["model"]
    model.eval()

    with torch.no_grad():
        preds = model(X_test).cpu().numpy().flatten()

    preds = preds * sigma + mu
    return preds

def train_model(series):
    """
    直接传入完整的数据集进行训练，内部实现会自动根据Config.train_percent划分训练集和测试集
    返回训练好的模型字典： (imf_idx, model_name) -> trained_bundle
    """
    model_zoo = {}   # (imf_idx, model_name) -> trained_bundle
    loss_records = []
    if(Config.vmd_enable):
        imfs = my_vmd.vmd_decompose(series)
        for imf_idx, imf in enumerate(imfs):
            # “多个模型对应不同的imf都要训练”
            for model_name in Config.train_models:
                key = (imf_idx, model_name)
                print(f"训练 IMF-{imf_idx+1} 使用模型 {model_name}...")
                train_start = time.time()
                if key not in model_zoo:
                    bundel = train_imf_model(
                        imf=imf,
                        imf_idx=imf_idx,
                        model_name=model_name,
                        window=Config.window,
                        epochs=Config.epochs,
                        batch_size=32
                    )
                    model_zoo[key] = bundel
                    loss_records.append(bundel["loss_history"])
                    # 保存模型
                    save_imf_model(model_zoo[key])
                    train_end = time.time()
                    print(f"[TIME] IMF-{imf_idx+1} | {model_name} 训练耗时: {train_end - train_start:.2f} 秒")
                    metrics.save_imf_model_train_time(imf_idx=imf_idx, model_name=model_name, train_time=train_end - train_start, sheet_name="Train_times", file_name=Config.model_predict_file)
                    plot_loss_curve(
                        bundel["loss_history"],
                        save_path=f"train/loss_imf{imf_idx}_{model_name}.png",
                        title=f"IMF-{imf_idx}_{model_name} 训练损失曲线"
                    )
    else:  # 不进行 VMD 分解，整体序列训练单模型
        imf_idx = -1  #-1表示整体序列,没有进行vmd分解
        imf = series
        for model_name in Config.train_models:
            key = (imf_idx, model_name)
            print(f"训练 整体序列 使用模型 {model_name}...")
            train_start = time.time()
            if key not in model_zoo:
                bundel = train_imf_model(
                    imf=imf,
                    imf_idx=imf_idx,
                    model_name=model_name,
                    window=Config.window,
                    epochs=Config.epochs,
                    batch_size=32
                )
                model_zoo[key] = bundel
                loss_records.append(bundel["loss_history"])
                # 保存模型
                save_imf_model(model_zoo[key])
                train_end = time.time()
                print(f"[TIME] 整体序列 | {model_name} 训练耗时: {train_end - train_start:.2f} 秒")
                metrics.save_imf_model_train_time(imf_idx=imf_idx, model_name=model_name, train_time=train_end - train_start, sheet_name="Train_times", file_name=Config.model_predict_file)
                plot_loss_curve(
                    bundel["loss_history"],
                    save_path=f"train/loss_{model_name}.png",
                    title=f"整体序列_{model_name} 训练损失曲线"
                )
    return model_zoo, loss_records

def load_imf_model(imf_idx, model_name, window, save_root="checkpoints"):
    load_dir = os.path.join(save_root, f"IMF{imf_idx+1}_{model_name}")

    model_path = os.path.join(load_dir, "model.pth")
    meta_path = os.path.join(load_dir, "meta.json")

    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None  # 表示本地不存在，需要重新训练

    # 1. 读取 meta
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 2. 构建模型结构
    model = build_model(model_name, window)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"[LOAD] IMF-{imf_idx+1} | {model_name} loaded.")

    return {
        "model": model,
        "mu": meta["mu"],
        "sigma": meta["sigma"],
        "imf_idx": meta["imf_idx"],
        "model_name": meta["model_name"]
    }



def save_imf_model(trained_bundle, save_root="checkpoints"):
    imf_idx = trained_bundle["imf_idx"]
    model_name = trained_bundle["model_name"]

    save_dir = os.path.join(save_root, f"IMF{imf_idx+1}_{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. 保存模型参数
    torch.save(
        trained_bundle["model"].state_dict(),
        os.path.join(save_dir, "model.pth")
    )

    # 2. 保存元信息（JSON 可读、可复现）
    meta = {
        "imf_idx": imf_idx,
        "model_name": model_name,
        "mu": trained_bundle["mu"],
        "sigma": trained_bundle["sigma"],
        "window": Config.window,
        "train_percent": Config.train_percent
    }

    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SAVE] IMF-{imf_idx+1} | {model_name} saved.")


if __name__ == '__main__':
    #删除所有.png文件
    delete_all_png_files()
    model_zoo = {}
    #加载数据
    series, scaler = load_data(Config.file_name)
    # 重新训练模型
    if(Config.retrain_model):
        model_zoo, loss_records = train_model(series)
    else:  # 加载已有模型
        for imf_idx in range(Config.K):  #假设有3个imf分量
            for model_name in Config.train_models:
                bundel = load_imf_model(imf_idx=imf_idx, model_name=model_name, window=Config.window)
                if bundel is not None:
                    model_zoo[(imf_idx, model_name)] = bundel
    if model_zoo is not None and len(model_zoo) > 0:
        print(f"模型库加载完成，共有 {len(model_zoo)} 个训练好的 IMF-模型 组合。")
    else:
        print("模型库为空，请设置Config.retrain_model=True重新训练模型后再运行。")
        exit(1)


    # 运行 VMD-CNN-BiLSTM 模型获取预测结果
    if(Config.vmd_enable):
        print("使用 VMD-CNN-BiLSTM 组合模型进行预测...")
        y_true, y_pred, loss_records, imf_preds, imf_trues = vmd_cnn_bilstm_pipeline(Config.file_name)
    else:
        print("使用单模型进行预测...")
        y_true, y_pred, loss_records = no_cmd_pipeline(Config.file_name, Config.single_model)

    # 计算并打印保存回归指标
    # metrics.evaluate(y_true=y_true, y_pred=y_pred)
    metrics.save_evaluation(y_true, y_pred, filename="模型性能指标保存.txt", out_dir="result")
    
    if os.path.exists(Config.model_predict_file):
        print(f"文件 {Config.model_predict_file} 已存在, 跳过写入真实值")
    else:  # 文件创建时写入真实值
        save_pred_and_metrics_to_file(y_true=y_true, y_pred=None, file_name=Config.model_predict_file, model_name="TRUE_VALUE")


    # 追加写入当前模型预测结果
    if Config.vmd_enable:
        save_pred_and_metrics_to_file(y_true=y_true, y_pred=y_pred, file_name=Config.model_predict_file, model_name=get_model_name_from_config(), imf_preds=imf_preds, imf_trues=imf_trues)
    else:
        save_pred_and_metrics_to_file(y_true=y_true, y_pred=y_pred, file_name=Config.model_predict_file, model_name=get_model_name_from_config())

    # 绘制预测结果对比图
    plot_prediction(y_true, y_pred)
    

    
