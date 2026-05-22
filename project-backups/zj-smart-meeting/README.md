# 之江智会——让浙江企业的每一次会议都产生价值

## 项目简介

基于6个专业AI智能体协作的会议全流程管理平台。采用通义千问、DeepSeek等国产大模型，实现实时转录、智能分析、纪要生成、任务提取、知识沉淀的全自动化。为企业节省80%会议管理时间，提升40%信息完整度，将会议决策转化为可追踪的行动计划。

### 核心创新
- 多智能体分工协作
- 实时会议理解
- 跨会议知识图谱
- 全栈国产化技术路线

### 立足余杭
依托阿里生态（通义千问）、服务3000+余杭高新企业，打造浙江会议管理标杆。

## 技术架构

### 多智能体系统
1. **记录员Agent** - 阿里云ASR实时转录
2. **分析师Agent** - 通义千问qwen-max深度分析
3. **总结者Agent** - 通义千问qwen-plus生成纪要
4. **任务官Agent** - DeepSeek提取任务
5. **知识管家Agent** - Qdrant向量检索
6. **协调员Agent** - LangGraph工作流编排

### 技术栈（全栈国产化）
- **大模型**: 通义千问（主力）+ DeepSeek（辅助）
- **框架**: 阿里魔搭多智能体协作
- **后端**: Python + FastAPI
- **前端**: Next.js
- **数据库**: SQLite
- **基础设施**: 阿里云 + 魔搭社区算力

## 产品功能

### 核心功能
- **实时转录**: 95%+准确率，自动识别发言人
- **智能纪要**: 3分钟生成快速/标准/详细三版本
- **任务管理**: 自动提取待办、分配责任人、跟进提醒
- **会议洞察**: 效率评分、参与度分析、改进建议
- **知识沉淀**: 向量检索、智能问答、跨会议关联

### 技术指标
- 转录准确率: 95%+
- 纪要生成: <3分钟
- 任务识别率: 90%+
- 系统响应: <500ms

### 用户价值
- 节省80%时间
- 提升40%质量
- 年为百人企业节省400万成本

## 目录结构

```
zj-smart-meeting/
├── README.md                  # 项目说明文档
├── requirements.txt           # Python依赖包
├── backend/                   # 后端服务
│   ├── app.py                # FastAPI应用主文件
│   ├── agents/               # 智能体模块
│   │   ├── __init__.py
│   │   ├── recorder_agent.py      # 记录员Agent
│   │   ├── analyst_agent.py       # 分析师Agent
│   │   ├── summarizer_agent.py    # 总结者Agent
│   │   ├── task_agent.py          # 任务官Agent
│   │   ├── knowledge_agent.py     # 知识管家Agent
│   │   └── coordinator_agent.py   # 协调员Agent
│   ├── models/               # 数据模型
│   ├── utils/                # 工具函数
│   └── config/               # 配置文件
├── frontend/                 # 前端应用
│   ├── package.json
│   ├── src/
│   │   ├── components/       # React组件
│   │   ├── pages/           # 页面
│   │   └── utils/           # 前端工具
│   └── public/              # 静态资源
├── docs/                    # 项目文档
├── tests/                   # 测试文件
└── scripts/                 # 部署脚本
```

## 快速开始

1. 克隆项目到本地
2. 安装依赖: `pip install -r requirements.txt`
3. 配置环境变量
4. 启动后端服务: `python backend/app.py`
5. 启动前端应用: `cd frontend && npm run dev`

## 项目状态
- MVP已完成，核心功能可用
- 正在余杭本地企业试点
- 已获得阿里生态支持

---

**联系方式**: 项目团队致力于为浙江企业提供最专业的AI会议管理解决方案