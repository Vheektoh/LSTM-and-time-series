import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # by default LSTMs for pytorch accepts batch of tensors as (seq_len, batch, features
        # setting batch_first=True allows flexibility to shape my input as (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1) # the 1 can be num_classes if we are working with multiple classes then we would have to add the num_classes as an argument in the init

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        # out : batch_size, sequence_length, hidden_size
        out = out[:, -1, :]
        out = self.fc(out)
        return out
