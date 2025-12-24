import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model #每个词向量的维度，[L,d_model]
        self.n_heads = n_heads #注意力头的数量
        self.d_k = d_model // n_heads  #每个头的维度，[L,d_k]

        #输入词向量[,d.model]，输出所有的头Q,K,V[,d_k * n_heads = d_model]
        self.W_q = nn.Linear(d_model,d_model,bias=False)
        self.W_k = nn.Linear(d_model,d_model,bias=False)
        self.W_v = nn.Linear(d_model,d_model,bias=False)
        #输出变换
        self.W_o = nn.Linear(d_model,d_model,bias=False)
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        #Q:[L1,d_model]，每行对应一个Q,[d_model]
        #K.t:[d_model,L2]，每列对应一个K,[d_model]
        #对应相乘得到，每个Q,K的相似度,[L1,L2]
        print(Q.size())
        print(K.size())
        attention = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        #softmax
        attention = torch.softmax(attention, dim=-1)
        #Q,K相似度：[L1,L2]，每行对应该词与其他词的相似度[,L]
        #V:[L2,d_model]，每列对应
        output = torch.matmul(attention, V)
        return output
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1,2)
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    def forward(self, Q, K, V, mask=None):
        #Q,K,V：(batch, heads, seq_len, d_k)
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        #计算注意力:[batch,heads,l,d_k]
        output_attention = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(output_attention))
        return output
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)##初始化
        ##position.shape:[max_len]-->[max_len,1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_freq = torch.exp(-math.log(10000.0) * torch.arange(0,d_model, 2).float() / d_model)
        #[max_len,1]*[d_model/2]--->广播：[max_len,d_model/2]*[max_len,d_model/2]
        #[position1,position1,....]*[div_freq1,div_freq2,...]
        pe[:, 0::2] = torch.sin(position * div_freq)
        if d_model % 2 ==0:
            pe[:, 1::2] = torch.cos(position * div_freq)
        else:
            pe[:, 1::2] = torch.cos(position * div_freq[0:-1])
        #pe.shape:[max_len,d_model]-->[1,max_len,d_model]
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x):
        #x.shape=[b,seq_len,d_model],pe.shape=[1,max_len,d_model]
        return x + self.pe[:, :x.size(1)]
class Transformer(nn.Module):
    def __init__(self,
                 latent_dim,
                 seq_len,
                 d_model,
                 n_head,
                 num_layers):
        super(Transformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.linear_map = nn.Linear(latent_dim, seq_len * d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model,
                                                   nhead=n_head,
                                                   batch_first=True,
                                                   norm_first=True
                                                   )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers = num_layers)
        self.output_head = nn.Linear(seq_len * d_model, latent_dim)
    def forward(self,x):

        x = self.linear_map(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "embed_dim 必须能被 num_heads 整除"

        # 定义 Q, K, V 的投影层
        # 注意：Cross Attention 中，Q 和 K,V 的来源不同
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, query, key, value, need_weights=False):
        # query: (B, L_q, D) -> 你的 x_wide_norm
        # key:   (B, L_kv, D) -> 你的 x_kv
        # value: (B, L_kv, D) -> 你的 x_kv

        B, L_q, _ = query.shape
        _, L_kv, _ = key.shape

        # 1. 投影 (Project)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. 拆分 Heads (Split Heads)
        # 变换为 (B, Heads, Seq_Len, Head_Dim) 以适配 Flash Attn
        q = q.view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_kv, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. 【强制】调用 Flash Attention
        # 这里的 is_causal=False (Cross Attention 不做掩码)
        # 这里的 dropout_p 在 MPS 上如果不兼容，PyTorch 可能会报错或警告，
        # 但如果是 M4 芯片，通常能支持。如果还崩，把这里手动改成 0.0
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )

        # 4. 还原形状 (Concat Heads)
        output = output.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        # 5. 输出投影
        return self.out_proj(output), None