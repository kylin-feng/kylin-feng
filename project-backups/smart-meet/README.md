# SmartMeet AI - 多智能体协作智能会议助手

> 基于多智能体协作架构的智能会议管理系统，提供实时会议记录、智能分析和多版本纪要生成功能。

![SmartMeet AI](https://img.shields.io/badge/SmartMeet-AI-blue)
![React](https://img.shields.io/badge/React-18.2-61dafb)
![Node.js](https://img.shields.io/badge/Node.js-18+-green)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)

## ✨ 功能特色

### 🤖 多智能体协作
- **记录员智能体**: 实时语音转文字和发言人识别
- **分析师智能体**: 提取关键信息和决策要点  
- **秘书智能体**: 整理待办事项和责任分配
- **编辑智能体**: 优化语言表达和格式规范
- **质检智能体**: 验证信息准确性和逻辑检查

### 📝 智能会议管理
- 实时会议录制和转录
- 多版本会议纪要生成（高管版、执行版、技术版、客户版）
- 自动任务提取和分配
- 会议效率分析和ROI评估

### 🎨 现代化设计
- 极简主义设计理念
- 沉浸式交互体验
- 响应式布局，支持多设备
- 基于 shadcn/ui 的现代组件库

## 🚀 快速开始

### 环境要求
- Node.js 18+
- npm 或 yarn

### 安装依赖
```bash
# 克隆项目
git clone <repository-url>
cd smart-meet

# 安装所有依赖
npm run install:all
```

### 配置环境变量

#### 后端配置 (backend/.env)
```bash
# 大模型API配置
QIANWEN_API_KEY=your_qianwen_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# 其他配置...
```

#### 前端配置 (frontend/.env)
```bash
VITE_API_URL=http://localhost:5000/api
```

### 启动项目
```bash
# 同时启动前后端
npm run dev

# 或者分别启动
npm run dev:frontend  # 前端 (http://localhost:3000)
npm run dev:backend   # 后端 (http://localhost:5000)
```

## 📁 项目结构

```
smart-meet/
├── frontend/                 # React 前端
│   ├── src/
│   │   ├── components/       # 组件
│   │   │   ├── ui/          # shadcn/ui 基础组件
│   │   │   └── features/    # 业务组件
│   │   ├── pages/           # 页面
│   │   ├── services/        # API 服务
│   │   ├── types/           # TypeScript 类型
│   │   └── utils/           # 工具函数
│   ├── public/              # 静态资源
│   └── package.json
├── backend/                  # Node.js 后端
│   ├── src/
│   │   ├── routes/          # 路由
│   │   ├── controllers/     # 控制器
│   │   ├── services/        # 业务服务
│   │   ├── models/          # 数据模型
│   │   ├── middleware/      # 中间件
│   │   ├── config/          # 配置
│   │   └── utils/           # 工具函数
│   └── package.json
└── README.md
```

## 🔧 技术栈

### 前端
- **React 18** - 用户界面框架
- **TypeScript** - 类型安全
- **Vite** - 快速构建工具
- **Tailwind CSS** - 样式框架
- **shadcn/ui** - 现代组件库
- **Lucide React** - 图标库
- **Axios** - HTTP 客户端

### 后端  
- **Node.js** - 运行时环境
- **Express** - Web 框架
- **通义千问** - 内容分析和处理
- **DeepSeek** - 逻辑验证和质检
- **WebSocket** - 实时通信
- **Multer** - 文件上传

## 🤖 智能体架构

### 协作流程
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   会议录制启动   │────▶│  多智能体调度中心  │────▶│   结果整合输出   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                ┌─────────────▼─────────────┐
                │     智能体协作网络          │
                │ ┌───┐ ┌───┐ ┌───┐ ┌───┐ │
                │ │记录│ │分析│ │秘书│ │编辑│ │
                │ └───┘ └───┘ └───┘ └───┘ │
                │         ┌───┐             │
                │         │质检│             │
                │         └───┘             │
                └───────────────────────────┘
```

### 大模型分工
- **通义千问**: 记录员、分析师、秘书、编辑智能体
- **DeepSeek**: 质检智能体，负责逻辑验证和准确性检查

## 📋 API 文档

### 智能体相关
```bash
GET    /api/agents                    # 获取所有智能体
GET    /api/agents/:id               # 获取单个智能体
POST   /api/agents/collaborate       # 启动协作
GET    /api/agents/collaborate/:id   # 获取协作状态
```

### 健康检查
```bash
GET    /api/health                   # 服务健康检查
GET    /api/info                     # API 信息
```

## 🔨 开发指南

### 添加新智能体
1. 在 `backend/src/services/AgentService.js` 中定义智能体
2. 实现对应的处理方法
3. 更新前端 `Agent` 类型定义
4. 在前端添加对应的UI组件

### 自定义主题
修改 `frontend/src/index.css` 中的 CSS 变量：
```css
:root {
  --primary: 262.1 83.3% 57.8%;
  --background: 0 0% 100%;
  /* ... */
}
```

## 🛠️ 部署

### Docker 部署
```bash
# 构建镜像
docker build -t smart-meet .

# 运行容器
docker run -p 3000:3000 -p 5000:5000 smart-meet
```

### 生产环境
```bash
# 构建前端
cd frontend && npm run build

# 启动后端
cd backend && npm start
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [shadcn/ui](https://ui.shadcn.com/) - 优秀的组件库
- [Lucide](https://lucide.dev/) - 精美的图标
- [通义千问](https://tongyi.aliyun.com/) - 强大的大语言模型
- [DeepSeek](https://www.deepseek.com/) - 高质量的推理模型

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

- 项目 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

---

⭐ 如果这个项目对您有帮助，请给它一个 Star！