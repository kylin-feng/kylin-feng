"""
Transformer编码器层实现
包含自注意力和前馈网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # 自注意力机制
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        
        return src

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=5000):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        output = self.transformer_encoder(src, src_mask)
        return self.fc_out(output)

# 测试代码
def test_transformer():
    vocab_size = 10000
    model = SimpleTransformer(vocab_size)
    
    # 创建随机输入
    batch_size, seq_len = 32, 50
    input_ids = torch.randint(0, vocab_size, (seq_len, batch_size))
    
    output = model(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {output.shape}")
    print("Transformer模型测试完成！")

if __name__ == "__main__":
    test_transformer()
