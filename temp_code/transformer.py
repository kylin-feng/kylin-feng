"""
基于Transformer的文本分类器
实现情感分析功能
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 简单的词汇编码（实际项目中使用专业tokenizer）
        tokens = text.lower().split()[:self.max_length]
        token_ids = [hash(token) % 10000 for token in tokens]
        
        # 填充到固定长度
        while len(token_ids) < self.max_length:
            token_ids.append(0)
            
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, num_classes=2, max_length=128):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        
        # 词嵌入 + 位置编码
        embeddings = self.embedding(x)  # (batch_size, seq_len, d_model)
        embeddings += self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer需要 (seq_len, batch_size, d_model)
        embeddings = embeddings.transpose(0, 1)
        
        # 通过Transformer
        transformer_out = self.transformer(embeddings)  # (seq_len, batch_size, d_model)
        
        # 取第一个token的输出进行分类 (类似BERT的[CLS])
        cls_token = transformer_out[0]  # (batch_size, d_model)
        
        # 分类
        logits = self.classifier(cls_token)
        return logits

def train_step(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct_predictions += (pred == target).sum().item()
        total_samples += target.size(0)
        
        if batch_idx % 10 == 0:
            print(f'批次 {batch_idx}, 损失: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

# 示例训练函数
def example_training():
    # 模拟数据
    texts = ["这是一个很好的电影", "这部电影很糟糕", "我喜欢这个产品", "质量太差了"]
    labels = [1, 0, 1, 0]  # 1: 正面, 0: 负面
    
    # 创建数据集
    dataset = TextClassificationDataset(texts, labels, None)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 创建模型
    model = TransformerClassifier(vocab_size=10000, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("开始训练Transformer分类器...")
    for epoch in range(2):
        avg_loss, accuracy = train_step(model, dataloader, optimizer, criterion, device)
        print(f'Epoch {epoch+1}: 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}')

if __name__ == "__main__":
    example_training()
