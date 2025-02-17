import torch.nn as nn
import torch
import math

class Generator(nn.Module):
    def __init__(self, dim, vocab_num):
        super(Generator,self).__init__()
        self.proj = nn.Linear(dim, vocab_num)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)

class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x+self.dropout(sublayer(x)))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear2(self.relu(self.linear1(x))))


class Embeddings(nn.Module):
    def __init__(self, vocab_num, dim):
        super(Embeddings,self).__init__()
        self.emb = nn.Embedding(vocab_num, dim)
        self.dim = dim
    def forward(self, x):
        return self.emb(x) * math.sqrt(self.dim)
    

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dim, dropout=0.1):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, dim)

        position = torch.arange(0, max_seq_len).unsqueeze(1)
        base = torch.full((dim // 2,), 10000.0)

        pow_term = torch.arange(0, dim, 2)/torch.tensor(dim, dtype=torch.float32)
        div_term = torch.pow(base, pow_term)

        pe[:, 0::2] = torch.sin(position/div_term)
        pe[:, 1::2] = torch.cos(position/div_term)

        pe = pe.unsqueeze(0)

        #pe를 학습되지 않는 변수로 등록록
        self.register_buffer('positional_encoding', pe)


    def forward(self, x):
        x = x + self.positional_encoding[:, :x.size(1), :].detach()
        return self.dropout(x)
