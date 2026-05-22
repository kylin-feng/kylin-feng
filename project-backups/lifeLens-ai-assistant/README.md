# LifeLens AI生活助手

基于Rokid AR眼镜的智能生活决策支持应用

## 项目概述

LifeLens是一款基于Rokid AR眼镜的AI生活助手，通过第一视角视觉识别+语音交互，为用户提供实时、智能的生活决策支持。

### 核心价值
让AI真正"看懂"你的生活，解放双手的同时解放大脑。

## 核心功能

### MVP功能 (P0)
1. **智能识物+信息查询** - 识别任何物体并提供相关信息
2. **即时餐饮推荐** - 实时餐厅评价和推荐
3. **购物决策助手** - 商品比价和购买建议
4. **导航问路助手** - AR导航和路线指引
5. **语音备忘录** - 语音记录和智能分类

### 增强功能 (P1)
- 实时翻译
- 健康助手
- 社交话题生成

## 技术架构

### 前端
- **框架**: Flutter
- **平台**: Rokid AR眼镜
- **语言**: Dart

### 后端服务
- **AI平台**: 灵珠Agent平台
- **视觉识别**: 百度AI/阿里云视觉API
- **地图服务**: 高德地图API
- **餐饮数据**: 大众点评API
- **商品数据**: 京东/淘宝API

### 项目结构
```
lib/
├── core/                 # 核心配置和工具
├── features/             # 功能模块
│   ├── object_recognition/    # 智能识物
│   ├── restaurant_recommendation/ # 餐饮推荐
│   ├── shopping_assistant/    # 购物助手
│   ├── navigation_helper/     # 导航助手
│   └── voice_memo/           # 语音备忘录
├── services/             # 服务层
│   ├── ai_vision/            # 视觉识别服务
│   ├── search/               # 搜索服务
│   ├── map/                  # 地图服务
│   ├── restaurant/           # 餐饮服务
│   ├── commerce/             # 电商服务
│   ├── voice/                # 语音服务
│   └── ar_display/           # AR显示服务
├── models/               # 数据模型
└── widgets/              # 通用组件
```

## 开发指南

### 环境要求
- Flutter 3.10+
- Dart 3.0+
- Rokid AR SDK
- Android Studio / VS Code

### 安装依赖
```bash
flutter pub get
```

### 运行项目
```bash
flutter run
```

### 构建项目
```bash
flutter build apk --release
```

## API密钥配置

在 `config/api_keys.dart` 中配置以下API密钥：
- 百度AI API Key
- 高德地图API Key
- 大众点评API Key
- 其他第三方服务密钥

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 许可证

MIT License

## 联系方式

- 项目维护者: 冯麒鸣
- 邮箱: your-email@example.com
- 项目地址: https://github.com/your-username/lifeLens-ai-assistant