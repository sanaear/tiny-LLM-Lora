import torch
import torch.nn as nn

# Multi-head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        B, T, D = x.size()
        Q = self.q_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        K = self.k_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        V = self.v_linear(x).view(B, T, self.n_heads, self.d_k).transpose(1,2)
        
        scores = torch.matmul(Q, K.transpose(-2,-1)) / (self.d_k**0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)
        
        context = context.transpose(1,2).contiguous().view(B, T, D)
        return self.out(context)

# Feed Forward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(x)
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x

# Tiny Transformer
class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)
