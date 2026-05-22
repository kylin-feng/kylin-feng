from flask import Flask, render_template, request, jsonify, redirect, url_for
import schedule
import time
import threading
import subprocess
import os
import json
from datetime import datetime
import logging

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 任务配置文件
CONFIG_FILE = 'config.json'

# 默认配置
DEFAULT_CONFIG = {
    'tasks': [],
    'github_repo_path': '',
    'github_remote': 'origin',
    'github_branch': 'main',
    'auto_push': True
}

def load_config():
    """加载配置文件"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """保存配置文件"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def execute_git_command(cmd, repo_path):
    """执行Git命令"""
    try:
        result = subprocess.run(cmd, cwd=repo_path, shell=True, 
                              capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def create_daily_log_entry():
    """创建日常日志条目"""
    config = load_config()
    repo_path = config.get('github_repo_path', '')
    
    if not repo_path or not os.path.exists(repo_path):
        logger.error(f"仓库路径不存在: {repo_path}")
        return False
    
    # 创建日志文件
    log_dir = os.path.join(repo_path, 'daily_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(log_dir, f'{today}.md')
    
    # 写入日志内容
    log_content = f"""# Daily Log - {today}

## 今日任务完成情况
- [ ] 代码review
- [ ] 学习新技术
- [ ] 文档更新

## 技术笔记
今日学习了新的编程概念和最佳实践。

## 代码提交统计
- 提交时间: {datetime.now().strftime('%H:%M:%S')}
- 自动化脚本生成

---
*此日志由自动化脚本生成*
"""
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    return commit_changes(repo_path, f"Add daily log for {today}")

def create_learning_note():
    """创建学习笔记"""
    config = load_config()
    repo_path = config.get('github_repo_path', '')
    
    if not repo_path or not os.path.exists(repo_path):
        logger.error(f"仓库路径不存在: {repo_path}")
        return False
    
    # 创建学习笔记目录
    notes_dir = os.path.join(repo_path, 'learning_notes')
    os.makedirs(notes_dir, exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    note_file = os.path.join(notes_dir, f'note_{today}.md')
    
    # 学习主题列表
    topics = [
        "Python最佳实践",
        "Web开发新技术",
        "数据结构与算法",
        "软件设计模式",
        "前端框架学习",
        "数据库优化",
        "API设计原则",
        "代码重构技巧"
    ]
    
    import random
    topic = random.choice(topics)
    
    note_content = f"""# {topic} - {today}

## 学习要点
1. 理解核心概念
2. 实践应用场景
3. 最佳实践总结

## 代码示例
```python
# 示例代码
def example_function():
    pass
```

## 总结
今日学习{topic}，收获颇丰。

---
*学习笔记 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(note_file, 'w', encoding='utf-8') as f:
        f.write(note_content)
    
    return commit_changes(repo_path, f"Add learning note: {topic}")

def update_readme():
    """更新README文件"""
    config = load_config()
    repo_path = config.get('github_repo_path', '')
    
    if not repo_path or not os.path.exists(repo_path):
        logger.error(f"仓库路径不存在: {repo_path}")
        return False
    
    readme_file = os.path.join(repo_path, 'README.md')
    
    # 个人介绍部分（固定内容）
    personal_intro = """### Hi there, I'm [Kylin Feng]. 👋

I am a learning student from China.
I like open source and all interesting things and want to try to do it.
I want to be an interesting person and create something that can be remembered by others.


- 🔭 I'm currently writing some amateur [open source projects].
- 🌱 I'm currently learning Computer Science & AI & Drawing, and want to learn everything interesting.
- 🤔 I want to make a ElectronBot. 
- ❤️ I enjoy coding 💻, exploring 🌍, listening to 🎵, practicing yoga 🧘‍♀️, and reading 📚
- 💬 Be free to ask me about anything [here](https://github.com/kylin-feng/kylin-feng/issues).

---

#### Languages


<img align="right" width="450" src="https://github-readme-stats.vercel.app/api?username=kylin-feng&show_icons=true&icon_color=0078e7&title_color=0078e7&include_all_commits=true"/>
<code><img height="20" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png" alt="python" /></code>

#### Frameworks and Tools


<code><img height="20" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/vue/vue.png" alt="vue" /></code>
<code><img height="20" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/react/react.png" alt="react" /></code>
<code><img height="20" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png" alt="git" /></code>


#### Interested


<code><img height="20" src="https://simpleicons.org/icons/blender.svg" alt="blender" /></code>
<code><img height="20" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/unity/unity.png" alt="unity" /></code>

---

"""
    
    # 自动化内容部分
    automation_content = f"""
## 📊 自动化学习记录

### 项目简介
这个仓库使用自动化工具维护，记录我的日常技术学习和实践轨迹。

### 📈 学习统计
- 📝 总提交数: {get_commit_count(repo_path)}
- 🕐 最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 🤖 自动化维护中...

### 📁 目录结构
```
├── daily_logs/     # 📚 日常学习日志
├── learning_notes/ # 🔬 技术学习笔记
└── README.md      # 📖 项目说明
```

### 🎯 学习目标
- 保持每日技术学习习惯
- 记录学习过程和心得
- 探索新技术和最佳实践
- 持续提升编程技能

---
*🤖 由自动化学习工具维护 - {datetime.now().strftime('%Y-%m-%d')}*
"""
    
    # 合并个人介绍和自动化内容
    readme_content = personal_intro + automation_content
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    return commit_changes(repo_path, "Update README with latest statistics")

def get_commit_count(repo_path):
    """获取提交数量"""
    success, output, _ = execute_git_command("git rev-list --count HEAD", repo_path)
    if success:
        return output.strip()
    return "N/A"

def commit_and_push_changes(repo_path, message):
    """提交更改到Git并推送到远程仓库"""
    config = load_config()
    remote = config.get('github_remote', 'origin')
    branch = config.get('github_branch', 'main')
    auto_push = config.get('auto_push', True)
    
    commands = [
        "git add .",
        f'git commit -m "{message}"'
    ]
    
    # 如果启用自动推送，添加push命令
    if auto_push:
        commands.append(f'git push {remote} {branch}')
    
    for cmd in commands:
        success, output, error = execute_git_command(cmd, repo_path)
        if not success:
            # 如果是没有变更要提交，不算错误
            if "nothing to commit" in error.lower():
                logger.info("没有新的变更需要提交")
                return True
            logger.error(f"命令执行失败: {cmd}, 错误: {error}")
            return False
        logger.info(f"命令执行成功: {cmd}")
    
    return True

def commit_changes(repo_path, message):
    """提交更改到Git（保持向后兼容）"""
    return commit_and_push_changes(repo_path, message)

def create_and_delete_python_file():
    """创建并删除Python文件（模拟代码开发过程）"""
    config = load_config()
    repo_path = config.get('github_repo_path', '')
    
    if not repo_path or not os.path.exists(repo_path):
        logger.error(f"仓库路径不存在: {repo_path}")
        return False
    
    # 创建临时代码目录
    temp_dir = os.path.join(repo_path, 'temp_code')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Python文件路径
    py_file = os.path.join(temp_dir, 'transformer.py')
    
    # 生成Transformer相关的Python代码
    transformer_codes = [
        '''"""
Transformer模型实现 - 自注意力机制
用于学习和实验目的
"""
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query)
        K = self.W_k(key)  
        V = self.W_v(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.W_o(attn_output)

# 示例使用
if __name__ == "__main__":
    model = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(32, 10, 512)  # (batch_size, seq_len, d_model)
    output = model(x, x, x)
    print(f"输出形状: {output.shape}")
''',
        '''"""
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
''',
        '''"""
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
''',
        '''"""
Transformer模型的注意力可视化工具
帮助理解模型的注意力机制
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子函数来捕获注意力权重"""
        def hook_fn(module, input, output):
            if hasattr(module, 'attention_weights'):
                layer_name = f"{module.__class__.__name__}_{len(self.attention_weights)}"
                self.attention_weights[layer_name] = output[1].detach().cpu()
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
    
    def visualize_attention(self, tokens, layer_idx=0, head_idx=0):
        """可视化指定层和头的注意力权重"""
        if not self.attention_weights:
            print("没有找到注意力权重，请先运行模型前向传播")
            return
        
        layer_name = list(self.attention_weights.keys())[layer_idx]
        attn_weights = self.attention_weights[layer_name]
        
        # 选择第一个样本和指定的头
        attn_matrix = attn_weights[0, head_idx].numpy()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(attn_matrix, 
                   xticklabels=tokens, 
                   yticklabels=tokens,
                   cmap='Blues', 
                   cbar=True,
                   square=True)
        plt.title(f'注意力权重热力图 - 层{layer_idx} 头{head_idx}')
        plt.xlabel('Key位置')
        plt.ylabel('Query位置')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def attention_summary(self):
        """输出注意力权重的统计信息"""
        print("=== 注意力权重统计 ===")
        for layer_name, weights in self.attention_weights.items():
            print(f"层 {layer_name}:")
            print(f"  形状: {weights.shape}")
            print(f"  均值: {weights.mean().item():.4f}")
            print(f"  标准差: {weights.std().item():.4f}")
            print(f"  最大值: {weights.max().item():.4f}")
            print(f"  最小值: {weights.min().item():.4f}")
            print("-" * 30)
    
    def remove_hooks(self):
        """移除所有钩子函数"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class SimpleAttentionModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=2):
        super(SimpleAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(100, d_model))
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # 词嵌入和位置编码
        seq_len = x.size(1)
        embeddings = self.embedding(x) + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # 通过多层注意力
        for attention, norm in zip(self.attention_layers, self.norm_layers):
            attn_output, attn_weights = attention(embeddings, embeddings, embeddings)
            embeddings = norm(embeddings + attn_output)
            
            # 保存注意力权重用于可视化
            attention.attention_weights = attn_weights
        
        return embeddings

def demo_attention_visualization():
    """演示注意力可视化功能"""
    # 创建模型和示例数据
    vocab_size = 1000
    model = SimpleAttentionModel(vocab_size)
    
    # 示例句子 (用数字表示token)
    tokens = ["我", "喜欢", "机器", "学习", "和", "深度", "学习"]
    token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    
    # 创建可视化器
    visualizer = AttentionVisualizer(model)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(token_ids)
    
    # 输出统计信息
    visualizer.attention_summary()
    
    # 可视化注意力（注意：实际运行需要matplotlib）
    print("注意力可视化功能已准备就绪")
    print("tokens:", tokens)
    
    # 清理
    visualizer.remove_hooks()

if __name__ == "__main__":
    demo_attention_visualization()
''',
        '''"""
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
'''
    ]
    
    import random
    code_content = random.choice(transformer_codes)
    
    # 写入Python文件
    with open(py_file, 'w', encoding='utf-8') as f:
        f.write(code_content)
    
    # 提交添加文件
    success1 = commit_and_push_changes(repo_path, "Add transformer.py - Exploring Transformer architecture")
    
    if success1:
        # 删除文件
        if os.path.exists(py_file):
            os.remove(py_file)
            
        # 如果目录为空，也删除目录
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            
        # 提交删除
        success2 = commit_and_push_changes(repo_path, "Remove transformer.py - Code experiment completed")
        return success2
    
    return False

def scheduled_task():
    """定时任务执行函数"""
    logger.info("开始执行定时任务...")
    
    tasks = [
        ("创建日志条目", create_daily_log_entry),
        ("创建学习笔记", create_learning_note),
        ("更新README", update_readme),
        ("Python代码实验", create_and_delete_python_file)
    ]
    
    import random
    # 随机选择2-3个任务执行，确保包含Python代码实验
    base_tasks = random.sample(tasks[:-1], k=random.randint(1, 2))  # 从前3个任务中选择
    selected_tasks = base_tasks + [tasks[-1]]  # 总是包含Python代码实验
    
    for task_name, task_func in selected_tasks:
        logger.info(f"执行任务: {task_name}")
        try:
            result = task_func()
            if result:
                logger.info(f"任务 {task_name} 执行成功")
            else:
                logger.error(f"任务 {task_name} 执行失败")
        except Exception as e:
            logger.error(f"任务 {task_name} 执行异常: {str(e)}")

# 路由定义
@app.route('/')
def index():
    """主页"""
    config = load_config()
    return render_template('index.html', config=config)

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    """配置页面"""
    if request.method == 'POST':
        config = load_config()
        
        # 更新基本配置
        config['github_repo_path'] = request.form.get('repo_path', '')
        config['github_remote'] = request.form.get('remote', 'origin')
        config['github_branch'] = request.form.get('branch', 'main')
        config['auto_push'] = request.form.get('auto_push') == 'on'
        
        save_config(config)
        return redirect(url_for('index'))
    
    config = load_config()
    return render_template('config.html', config=config)

@app.route('/api/run_task', methods=['POST'])
def run_task():
    """手动执行任务"""
    try:
        scheduled_task()
        return jsonify({'success': True, 'message': '任务执行成功'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'任务执行失败: {str(e)}'})

@app.route('/api/test_repo', methods=['POST'])
def test_repo():
    """测试仓库连接"""
    repo_path = request.json.get('repo_path', '')
    
    if not os.path.exists(repo_path):
        return jsonify({'success': False, 'message': '路径不存在'})
    
    success, output, error = execute_git_command("git status", repo_path)
    
    if success:
        return jsonify({'success': True, 'message': '仓库连接正常'})
    else:
        return jsonify({'success': False, 'message': f'仓库连接失败: {error}'})

@app.route('/api/test_python_experiment', methods=['POST'])
def test_python_experiment():
    """专门测试Python代码实验功能"""
    try:
        logger.info("开始测试Python代码实验功能...")
        result = create_and_delete_python_file()
        if result:
            return jsonify({'success': True, 'message': 'Python代码实验执行成功'})
        else:
            return jsonify({'success': False, 'message': 'Python代码实验执行失败'})
    except Exception as e:
        logger.error(f"Python代码实验异常: {str(e)}")
        return jsonify({'success': False, 'message': f'执行异常: {str(e)}'})

def setup_scheduler():
    """设置定时任务"""
    # 每天晚上8点执行
    schedule.every().day.at("20:00").do(scheduled_task)
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    
    # 在后台线程中运行调度器
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("定时任务调度器已启动，每天晚上8点执行任务")

if __name__ == '__main__':
    # 启动定时任务调度器
    setup_scheduler()
    
    # 启动Flask应用
    logger.info("GitHub自动化工具启动成功")
    app.run(host='0.0.0.0', port=58899, debug=False)