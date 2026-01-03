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
import metrics
import my_vmd
from metrics import evaluate




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
        if Config.single_model == "cnn":
            print(f"选择单个 CNN 模型")
            return CNN(window)
        elif Config.single_model == "rnn":
            print(f"选择单个 RNN 模型")
            return RNN(window)  
        elif Config.single_model == "lstm":
            print(f"选择单个 LSTM 模型")
            return LSTM(window)
        elif Config.single_model == "cnn_lstm":
            print(f"选择单个 CNN-LSTM 模型")
            return CNN_LSTM(window)
        elif Config.single_model == "cnn_bilstm":
            print(f"选择单个 CNN-BiLSTM 模型")
            return CNN_BiLSTM(window)
        else:
            raise ValueError(f"Unsupported single_model: {Config.single_model}")
    else:  # 根据 IMF 预测复杂度选择模型（谱熵）
        complexity = imf_spectral_entropy(imf)
        st = np.std(imf)
        if st > Config.std_bilstm_threshold:
            print(f"选择 CNN-BiLSTM 模型，IMF 复杂度: {complexity:.4f}， 标准差: {st:.4f}")
            return CNN_BiLSTM(window)
        elif st > Config.std_lstm_threshold:
            print(f"选择 CNN-LSTM 模型，IMF 复杂度: {complexity:.4f}， 标准差: {st:.4f}")
            return CNN_LSTM(window)
        else:
            print(f"选择 CNN 模型，IMF 复杂度: {complexity:.4f}， 标准差: {st:.4f}")
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
    loss_records = []
    imf_preds = []
    imf_trues = []

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
        imf_preds.append(pred)
        imf_trues.append(imf[-len(pred):])
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

    return y_true, final_pred, loss_records, imf_preds, imf_trues

def no_cmd_pipeline(file_path, select_model="cnn"):
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
    with pd.ExcelWriter(file_name, engine="openpyxl", mode="a" if os.path.exists(file_name) else "w", if_sheet_exists="replace") as writer:
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

        # 覆盖或新增行
        df_metrics_old.loc[model_name] = metrics_df_new.loc[model_name]

        # 写回（替换 sheet）
        with pd.ExcelWriter(file_name, engine="openpyxl", mode="a" if os.path.exists(file_name) else "w", if_sheet_exists="replace") as writer:
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
        with pd.ExcelWriter(file_name, engine="openpyxl", mode="a" if os.path.exists(file_name) else "w", if_sheet_exists="replace") as writer:
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
    for i, loss_hist in enumerate(loss_records):
        plot_loss_curve(
            loss_hist,
            save_path=f"loss_imf_{i+1}.png",
            title=f"IMF-{i+1} 训练损失曲线"
        )
