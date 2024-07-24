import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_units):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(input_dim, attention_units)
        self.context_vector = nn.Linear(attention_units, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = torch.tanh(self.attention(lstm_output))
        attn_weights = self.context_vector(attn_weights).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        attn_output = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return attn_output


class DenseLayer(nn.Module):
    def __init__(self, input_dim):
        super(DenseLayer, self).__init__()
        self.dense = nn.Linear(input_dim, input_dim // 2)

    def forward(self, x):
        return self.dense(x)
