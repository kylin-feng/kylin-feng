# QAI智能助手

<div align="center">
  <img src="src/images/robot.png" alt="QAI Logo" width="120">
  <h1>QAI智能助手</h1>
  <p>您的专属AI对话伙伴 - 智能、便捷、贴心</p>
</div>

## 📱 项目简介

QAI智能助手是一款基于人工智能技术开发的微信小程序，为用户提供智能对话、问答咨询、生活助手等多种AI服务。采用现代化的技术栈，提供流畅的用户体验。

## ✨ 核心功能

- 🤖 **智能对话** - 基于先进AI模型的自然语言对话
- 💬 **多轮对话** - 支持上下文理解的连续对话
- 🔐 **用户系统** - 完整的用户登录注册体系
- 💳 **充值系统** - 灵活的积分充值机制
- 📱 **响应式设计** - 适配各种屏幕尺寸
- 🎨 **现代化UI** - 简洁美观的用户界面

## 🛠 技术栈

- **框架**: Taro 3.6+ (支持多端开发)
- **语言**: TypeScript
- **UI库**: Taro Components
- **样式**: Sass/SCSS
- **状态管理**: React Context
- **构建工具**: Webpack 5
- **代码规范**: ESLint + Prettier

## 📦 项目结构

```
Q_AI/
├── src/                    # 源代码目录
│   ├── pages/             # 页面目录
│   │   ├── chat/          # AI对话页面
│   │   ├── home/          # 个人中心页面
│   │   ├── login/         # 登录注册页面
│   │   └── recharge/      # 充值页面
│   ├── images/            # 图片资源
│   ├── app.config.ts      # 应用配置
│   ├── app.tsx           # 应用入口
│   └── app.scss          # 全局样式
├── config/                # 构建配置
├── types/                 # TypeScript类型定义
└── __tests__/            # 测试文件
```

## 🚀 快速开始

### 环境要求

- Node.js >= 14.0
- npm >= 6.0 或 yarn >= 1.0
- 微信开发者工具

### 安装依赖

```bash
# 使用npm
npm install

# 或使用yarn
yarn install
```

### 开发运行

```bash
# 启动微信小程序开发模式
npm run dev:weapp

# 启动H5开发模式
npm run dev:h5

# 启动支付宝小程序开发模式
npm run dev:alipay
```

### 构建发布

```bash
# 构建微信小程序
npm run build:weapp

# 构建H5版本
npm run build:h5

# 构建支付宝小程序
npm run build:alipay
```

## 🎯 页面说明

### AI对话页面 (`/pages/chat`)
- 智能对话界面
- 消息收发功能
- 对话历史记录
- 实时响应状态

### 个人中心 (`/pages/home`)
- 用户信息展示
- 使用统计
- 设置选项
- 账户管理

### 登录注册 (`/pages/login`)
- 用户登录
- 新用户注册
- 密码找回
- 第三方授权登录

### 充值页面 (`/pages/recharge`)
- 积分充值
- 套餐选择
- 支付功能
- 充值记录

## ⚙️ 配置说明

### 环境配置
- `env.development` - 开发环境配置
- `env.production` - 生产环境配置

### 小程序配置
- `project.config.json` - 微信小程序项目配置
- `project.tt.json` - 字节跳动小程序配置

## 🔧 开发指南

### 代码规范
项目使用ESLint进行代码规范检查，请确保代码符合规范：

```bash
# 检查代码规范
npm run lint

# 自动修复规范问题
npm run lint:fix
```

### 提交规范
请使用规范的commit message格式：
- `feat: 新功能`
- `fix: 修复bug`
- `docs: 文档更新`
- `style: 样式修改`
- `refactor: 代码重构`

## 📝 更新日志

### v1.0.0 (2024-12-20)
- 🎉 项目初始化
- ✨ 完成基础架构搭建
- 🔧 优化项目配置
- 📱 实现核心页面结构

## 🤝 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目！

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的修改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系我们

- 项目地址: [GitHub Repository](https://github.com/your-username/q-ai-assistant)
- 问题反馈: [Issues](https://github.com/your-username/q-ai-assistant/issues)
- 邮箱: contact@qai-assistant.com

---

<div align="center">
  Made with ❤️ by QAI Team
</div> 