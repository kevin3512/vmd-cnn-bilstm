
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
