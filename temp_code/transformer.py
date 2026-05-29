"""
Transformer模型优化和调参工具
包含学习率调度、梯度裁剪等技巧
"""
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """训练配置参数"""
    vocab_size: int = 10000
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    max_length: int = 512
    dropout: float = 0.1
    
    batch_size: int = 32
    learning_rate: float = 0.0001
    num_epochs: int = 10
    warmup_steps: int = 4000
    
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    
class LearningRateScheduler:
    """Transformer原论文中的学习率调度器"""
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return (self.d_model ** (-0.5)) * min(arg1, arg2)

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class ModelOptimizer:
    """模型优化器包装类"""
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # 学习率调度器
        self.lr_scheduler = LearningRateScheduler(config.d_model, config.warmup_steps)
        
        # 损失函数
        self.criterion = LabelSmoothingLoss(
            classes=config.vocab_size,
            smoothing=config.label_smoothing
        )
        
        self.step_count = 0
        
    def step(self, loss):
        """执行一步优化"""
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.gradient_clip_norm
        )
        
        # 更新参数
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # 更新学习率
        self.step_count += 1
        new_lr = self.lr_scheduler(self.step_count)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
            
        return new_lr

class TrainingMetrics:
    """训练指标记录器"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.total_tokens = 0
        self.start_time = time.time()
        
    def update(self, loss, num_tokens):
        self.total_loss += loss
        self.total_tokens += num_tokens
        
    def get_metrics(self):
        elapsed_time = time.time() - self.start_time
        avg_loss = self.total_loss / max(self.total_tokens, 1)
        tokens_per_sec = self.total_tokens / max(elapsed_time, 1)
        
        return {
            'avg_loss': avg_loss,
            'perplexity': math.exp(min(avg_loss, 10)),  # 防止溢出
            'tokens_per_sec': tokens_per_sec,
            'elapsed_time': elapsed_time
        }

def create_padding_mask(sequences, pad_token_id=0):
    """创建填充掩码"""
    return (sequences != pad_token_id).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    """创建前瞻掩码（用于解码器）"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

class AdvancedTransformerTrainer:
    """高级Transformer训练器"""
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = ModelOptimizer(model, config)
        self.metrics = TrainingMetrics()
        
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        self.metrics.reset()
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            # 创建掩码
            src_mask = create_padding_mask(src)
            tgt_mask = create_padding_mask(tgt)
            
            # 前向传播
            output = self.model(src, tgt[:, :-1], src_mask, tgt_mask)
            
            # 计算损失
            loss = self.optimizer.criterion(
                output.view(-1, self.config.vocab_size),
                tgt[:, 1:].contiguous().view(-1)
            )
            
            # 优化步骤
            current_lr = self.optimizer.step(loss)
            
            # 更新指标
            num_tokens = (tgt[:, 1:] != 0).sum().item()
            self.metrics.update(loss.item() * num_tokens, num_tokens)
            
            # 打印进度
            if batch_idx % 100 == 0:
                metrics = self.metrics.get_metrics()
                print(f"Epoch {epoch}, Batch {batch_idx}")
                print(f"  损失: {metrics['avg_loss']:.4f}")
                print(f"  困惑度: {metrics['perplexity']:.2f}")
                print(f"  学习率: {current_lr:.2e}")
                print(f"  速度: {metrics['tokens_per_sec']:.0f} tokens/sec")
                
        return self.metrics.get_metrics()

def demo_advanced_training():
    """演示高级训练功能"""
    print("=== Transformer高级训练演示 ===")
    
    config = TrainingConfig(
        vocab_size=5000,
        d_model=256,
        num_heads=8,
        num_layers=4
    )
    
    # 创建简单的Transformer模型（这里用一个占位符）
    class DummyTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.output_proj = nn.Linear(config.d_model, config.vocab_size)
            
        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            embedded = self.embedding(tgt)
            return self.output_proj(embedded)
    
    model = DummyTransformer(config)
    trainer = AdvancedTransformerTrainer(model, config)
    
    print(f"配置参数: {config}")
    print("训练器已准备就绪！")
    print("包含功能:")
    print("- 自适应学习率调度")
    print("- 标签平滑")
    print("- 梯度裁剪") 
    print("- 训练指标监控")

if __name__ == "__main__":
    demo_advanced_training()
