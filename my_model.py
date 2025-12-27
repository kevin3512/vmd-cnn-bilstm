import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pywt
from vmdpy import VMD


def build_cnn_bilstm_model(self, seq_length, n_features=1):
    model = keras.Sequential([
        # CNN层提取局部特征
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', 
                    padding='same', input_shape=(seq_length, n_features)),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # BiLSTM层捕捉序列依赖
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        
        # 全连接层
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
        
def build_rnn_model(self, seq_length, n_features=1):
    """为缓慢变化数据构建简化的RNN"""
    model = keras.Sequential([
        # 简单的RNN层（LSTM可能过于复杂）
        layers.SimpleRNN(32, activation='tanh', 
                input_shape=(seq_length, n_features)),
        
        # 较小的全连接层
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # 输出层
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

    

def sarima_predict(self, data, forecast_steps=1):
    """
    SARIMA模型预测（用于中频分量）
    这里使用简化版本，实际应用中建议使用statsmodels的SARIMAX
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    try:
        # 自动选择参数或使用默认参数
        model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=forecast_steps)
        return forecast
    except:
        # 如果SARIMA失败，使用简单移动平均
        return np.mean(data[-12:]) if len(data) >= 12 else np.mean(data)