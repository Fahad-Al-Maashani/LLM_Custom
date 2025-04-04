# gpt_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTBlock(nn.Module):
    def __init__(self, emb_size, n_heads, dropout=0.1, forward_expansion=4):
        super(GPTBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * emb_size, emb_size)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, emb_size, num_layers, n_heads, max_seq_length, dropout=0.1):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(max_seq_length, emb_size)
        self.layers = nn.ModuleList([
            GPTBlock(emb_size, n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(emb_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_length = max_seq_length
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # [seq_len, batch_size, emb_size]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        for layer in self.layers:
            x = layer(x, mask)
        x = x.transpose(0, 1)  # [batch_size, seq_len, emb_size]
        logits = self.fc_out(x)
        return logits
