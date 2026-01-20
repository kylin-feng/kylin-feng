"""
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
