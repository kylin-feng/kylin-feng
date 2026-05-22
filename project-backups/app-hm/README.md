# AI小说生成器 - 鸿蒙应用

这是一个基于华为鸿蒙系统开发的AI小说生成应用，使用最新的ArkTS语言和ArkUI框架构建。

## 功能特性

- 🤖 AI智能小说生成
- 📖 小说阅读界面
- 💰 算力消耗和付费解锁机制
- 🎨 现代化UI设计
- 📱 支持手机和平板设备

## 技术栈

- **开发语言**: ArkTS (基于TypeScript)
- **UI框架**: ArkUI (声明式UI)
- **目标系统**: HarmonyOS 5.0+
- **开发工具**: DevEco Studio

## 项目结构

```
app-hm/
├── src/main/ets/
│   ├── entryability/
│   │   └── EntryAbility.ets          # 应用入口
│   ├── pages/
│   │   ├── Index.ets                 # 主页面
│   │   └── ReaderPage.ets            # 阅读页面
│   └── services/
│       └── AIWriterService.ets       # AI服务
├── src/main/resources/
│   ├── base/
│   │   ├── element/
│   │   │   └── string.json           # 字符串资源
│   │   └── profile/
│   │       └── main_pages.json       # 页面配置
├── app.json5                         # 应用配置
├── build-profile.json5               # 构建配置
└── oh-package.json5                  # 依赖配置
```

## 主要功能

### 1. 主界面 (Index.ets)
- 显示当前小说标题
- "开启AI写作之旅"按钮
- 算力消耗提示

### 2. 阅读界面 (ReaderPage.ets)
- 小说内容展示
- 滑动阅读功能
- 付费解锁机制
- 返回导航

### 3. AI服务 (AIWriterService.ets)
- 模拟AI小说生成
- 算力计算
- 内容分页处理

## 开发环境要求

1. **DevEco Studio**: 最新版本
2. **HarmonyOS SDK**: API 12+
3. **Node.js**: 18.0+
4. **TypeScript**: 4.9+

## 安装和运行

1. 使用DevEco Studio打开项目
2. 配置HarmonyOS SDK
3. 连接设备或启动模拟器
4. 点击运行按钮

## 配置说明

### 应用权限
应用需要以下权限：
- `ohos.permission.INTERNET`: 网络访问权限

### AI服务配置
在`AIWriterService.ets`中配置实际的AI服务API：
```typescript
private baseUrl: string = 'https://your-ai-service.com/api';
```

## 自定义配置

### 修改AI生成参数
在`AIWriterService.ets`中修改生成逻辑：
```typescript
private generateContent(request: NovelRequest): string {
  // 自定义生成逻辑
}
```

### 修改UI样式
在各个页面组件中修改样式：
```typescript
.fontSize(16)
.fontColor('#333333')
.backgroundColor('#FFFFFF')
```

## 注意事项

1. 确保使用最新的鸿蒙开发工具和SDK
2. 网络请求需要配置相应的权限
3. AI服务需要配置实际的API端点
4. 付费功能需要集成相应的支付SDK

## 许可证

MIT License
