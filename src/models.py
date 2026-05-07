import torch
import torch.nn as nn

INPUT_SIZE = 15      # 4 one-hot + 1 sigma + 10 noisy samples
OUTPUT_SIZE = 10     # 10 reconstructed clean samples
CONDITION_SIZE = 5   # one-hot (4) + sigma (1)
SEQ_LEN = 10         # number of noisy samples in the sequence


class MLPModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, OUTPUT_SIZE),
        )

    def forward(self, x):
        # x: (batch, 15)
        return self.net(x)


class RNNModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        # Each timestep receives: 1 noisy sample + 5 condition values.
        # Repeating condition at every step lets the RNN use frequency/sigma
        # context at each denoising decision rather than only at step 0.
        self.rnn = nn.RNN(
            input_size=CONDITION_SIZE + 1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, 15)
        condition = x[:, :CONDITION_SIZE]                          # (batch, 5)
        noisy = x[:, CONDITION_SIZE:].unsqueeze(-1)                # (batch, 10, 1)
        cond_seq = condition.unsqueeze(1).expand(-1, SEQ_LEN, -1)  # (batch, 10, 5)
        seq = torch.cat([noisy, cond_seq], dim=-1)                 # (batch, 10, 6)
        out, _ = self.rnn(seq)                                     # (batch, 10, hidden)
        return self.head(out).squeeze(-1)                          # (batch, 10)


class LSTMModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=CONDITION_SIZE + 1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, 15)
        condition = x[:, :CONDITION_SIZE]
        noisy = x[:, CONDITION_SIZE:].unsqueeze(-1)
        cond_seq = condition.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        seq = torch.cat([noisy, cond_seq], dim=-1)
        out, _ = self.lstm(seq)
        return self.head(out).squeeze(-1)
