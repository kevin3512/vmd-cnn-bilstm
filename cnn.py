
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, kernel_size=3)
        self.fc = nn.Linear((window - 2) * 32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class RNN(nn.Module):
    def __init__(self, window, n_features=1, hidden_size=32, fc_hidden=16):
        super().__init__()
        # input shape expected: (batch, seq_len) or (batch, seq_len, n_features)
        self.rnn = nn.RNN(input_size=n_features, hidden_size=hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc1 = nn.Linear(hidden_size, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # ensure features dimension exists
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        _, h = self.rnn(x)
        h_last = h[-1]  # (batch, hidden_size)
        x = self.relu(self.fc1(h_last))
        return self.fc2(x)

class LSTM(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class CNN_LSTM(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, 3)
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
    
class CNN_BiLSTM(nn.Module):
    def __init__(self, window):
        super().__init__()
        self.conv = nn.Conv1d(1, 32, 3)
        self.bilstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        _, (h, _) = self.bilstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)


class TCN(nn.Module):
    """
    A small Temporal Convolutional Network (dilated conv) for sequence regression.
    Keeps sequence length via padding and flattens features for a final linear layer.
    """
    def __init__(self, window, in_channels=1, channels=(32, 64, 64), kernel_size=3):
        super().__init__()
        layers = []
        prev_ch = in_channels
        dilation = 1
        for ch in channels:
            pad = (kernel_size - 1) * dilation // 2
            layers.append(nn.Conv1d(prev_ch, ch, kernel_size, padding=pad, dilation=dilation))
            layers.append(nn.ReLU())
            prev_ch = ch
            dilation *= 2

        self.net = nn.Sequential(*layers)
        # final linear maps flattened features (channels * seq_len) to 1
        self.fc = nn.Linear(prev_ch * window, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.net(x)     # (batch, channels, seq_len)
        x = x.view(x.size(0), -1)
        return self.fc(x)
