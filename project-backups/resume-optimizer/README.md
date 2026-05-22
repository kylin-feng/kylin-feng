# 职得 - AI简历优化助手

基于Streamlit的简历优化工具，使用硅基流动AI根据岗位描述(JD)智能优化简历内容。

## 功能特点

- 📝 **多格式支持**: 完美支持LaTeX、Markdown、Word/纯文本格式
- 🎯 **智能简历优化**: 基于JD要求优化简历内容，保持原格式不变
- 📊 **可视化对比**: 原简历vs优化后简历对比展示
- 💡 **优化建议**: 提供具体的改进建议和理由
- ⚡ **本地部署**: 无需数据库，本地快速运行

## 技术栈

- **前端框架**: Streamlit
- **AI服务**: 硅基流动 (SiliconFlow) API  
- **编程语言**: Python 3.13+
- **HTTP请求**: requests

## 项目结构

```
resume-optimizer/
├── app.py                 # 主应用文件
├── requirements.txt       # 项目依赖
├── .streamlit/
│   ├── secrets.example.toml # API密钥配置示例
│   └── secrets.toml         # 本地 API 密钥配置，不提交仓库
└── README.md             # 项目说明
```

## 安装和运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行应用

```bash
streamlit run app.py
```

应用将在 http://localhost:8501 启动

## 使用方法

1. **选择格式**: 选择你的简历原始格式（LaTeX/Markdown/Word纯文本）
2. **输入JD**: 在左侧文本框粘贴完整的岗位描述
3. **输入简历**: 在右侧文本框粘贴你的简历内容
4. **点击优化**: 点击"开始优化简历"按钮
5. **查看结果**: 在三个标签页中查看:
   - 💡 优化建议
   - ✨ 优化后简历（保持原格式）
   - 🔄 对比查看

## API配置

项目使用硅基流动AI服务。请复制 `.streamlit/secrets.example.toml` 为 `.streamlit/secrets.toml`，再填入自己的 API 密钥：

```toml
SILICONFLOW_API_KEY = "your-api-key"
```

## 优化亮点

- 🎯 **精准匹配**: 根据JD关键词优化简历
- 📝 **格式保持**: 完美保持LaTeX、Markdown、纯文本原始格式
- 📈 **提升通过率**: 突出相关技能和经验  
- ✨ **专业润色**: 改善表达和格式
- 🔍 **关键词优化**: 提高ATS系统识别度

## 格式支持详情

### LaTeX格式
- 保持所有LaTeX命令和环境（`\section{}`、`\textbf{}`、`\\`等）
- 适合学术界和研究领域的简历
- 完美支持复杂的排版结构

### Markdown格式  
- 保持Markdown语法（`#`、`**`、`-`、`*`等）
- 适合技术岗位和程序员简历
- 易于版本控制和在线展示

### Word/纯文本格式
- 保持简洁的文本结构
- 使用缩进和空行组织内容
- 适合传统行业和一般岗位

## 支持的AI模型

当前使用 `deepseek-ai/DeepSeek-R1` 模型，也可以配置其他硅基流动支持的模型：

- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Meta-Llama-3-8B-Instruct  
- 01-ai/Yi-1.5-9B-Chat-16K

## 注意事项

- 确保网络连接正常以访问AI API
- 简历内容请保持真实，AI只会优化表达不会编造经历
- 首次运行可能需要一些时间来下载依赖

## License

MIT License
