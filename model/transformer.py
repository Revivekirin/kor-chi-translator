from util import ResidualConnection, FeedForward, Embeddings, PositionalEncoding, Generator
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import math
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention,self).__init__()
        self.matmul = torch.matmul
        self.softmax = torch.softmax

    def forward(self, query, key, value, mask=None):
        key_transpose = torch.transpose(key, -2 -1)  #(batch, head_num, d_k, token_)
        matmul_result = self.matmul(query ,key_transpose)
        d_k = key.size()[-1]
        attention_score = matmul_result/math.sqrt(d_k)

        if mask is not None:
             attention_score = attention_score.masked_fill(mask==0, -1e20)
        
        softmax_attention_score = self.softmax(attention_score, dim=-1)
        result = self.matmul(softmax_attention_score, value)

        return result, softmax_attention_score
    

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num=8, dim=512, dropout=0.1):
        super(MultiHeadAttention,self).__init__()

        self.head_num = head_num
        self.dim = dim
        self.d_k = self.d_v = dim //head_num

        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.w_o = nn.Linear(dim, dim)

        self.self_attention = SelfAttention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_num = query.size(0)

        query = self.w_q(query).view(batch_num, -1, self.head_num, self.d_k).transpose(1, 2).contiguous()
        key = self.w_k(key).view(batch_num, -1, self.head_num, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_num, -1, self.head_num, self.d_k).transpose(1, 2)

        attention_result, attention_score = self.self_attention(query, key, value, mask)

        attention_result = attention_result.transpose(1, 2).contiguous().view(batch_num, -1, self.head_num * self.d_k)

        return self.w_o(attention_result)
    

class Encoder(nn.Module):
    def __init__(self, dim, head_num, dropout):
        super(Encoder,self).__init__()
        self.multi_head_attention =MultiHeadAttention(dim=dim, head_num=head_num)
        self.residual_1 = ResidualConnection(dim, dropout=dropout)

        self.feed_forward =FeedForward(dim)
        self.residual_2 = ResidualConnection(dim, dropout=dropout)
        
    def forward(self, input, mask):
        x = self.residual_1(input, lambda x: self.multi_head_attention(x, x, x, mask))
        x = self.residual_2(x, lambda x: self.feed_forward(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim, head_num, dropout):
        super(Decoder,self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(dim=dim, head_num=head_num)
        self.residual_1 = ResidualConnection(dim, dropout=dropout)
        
        self.encoder_decoder_attention = MultiHeadAttention(dim=dim, head_num=head_num)
        self.residual_2 = ResidualConnection(dim, dropout=dropout)

        self.feed_forward = FeedForward(dim)
        self.residual_3 = ResidualConnection(dim, dropout=dropout)

    def forward(self, target, encoder_output, target_mask, encoder_mask):
        # target, x, target_mask, input_mask
        x = self.residual_1(target, lambda x: self.masked_multi_head_attention(x, x, x, target_mask))
        x = self.residual_2(x, lambda x: self.encoder_decoder_attention(x, encoder_output, encoder_output, encoder_mask))
        x = self.residual_3(x, self.feed_forward)

        return x

class Transformer(nn.Module):
    def __init__(self, vocab_num, dim, max_seq_len, head_num, dropout, N):
        super(Transformer,self).__init__()
        self.embedding = Embeddings(vocab_num, dim)
        self.positional_encoding = PositionalEncoding(max_seq_len, dim)

        self.encoders = clones(Encoder(dim=dim, head_num=head_num, dropout=dropout))
        self.decoders = clones(Decoder(dim=dim, head_num=head_num, dropout=dropout))

        self.generator = Generator(dim,vocab_num)

    def forward(self, input, target, input_mask, target_mask, labels=None):
        x=self.positional_encoding(self.embedding(input))
        for encoder in self.encoders:
            x=encoder(x, input_mask)
        
        target = self.positional_encoding(self.embedding(target))
        for decoder in self.decoders:
            target = decoder(target, x, target_mask, input_mask)
        
        lm_logits = self.generator(target)
        loss=None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        loss = torch.tensor(0.0, device=lm_logits.device) if loss is None else loss
        
        return lm_logits, loss


    def encode(self, input, input_mask):
        x=self.positional_encoding(self.embedding(input))
        for encoder in self.encoders:
            x = encoder(x, input_mask)
        return x

    def decode(self, encode_output, encoder_mask, target, target_mask):
        target = self.positional_encoding(self.embedding(target))
        for decoder in self.decoders:
            target = decoder(target, encode_output, target_mask, encoder_mask)
        lm_logits = self.generator(target)
        return lm_logits 
