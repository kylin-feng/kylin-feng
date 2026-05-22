# SmartMeet AI 故障排除指南

## 问题诊断

如果无法访问 http://localhost:3000，请按以下步骤排查：

### 1. 快速诊断
打开 `test-frontend.html` 文件在浏览器中查看系统状态：
```bash
open /Users/shixianping/smart-meet/test-frontend.html
```

### 2. 手动启动服务

#### 启动后端服务：
```bash
cd /Users/shixianping/smart-meet/backend
npm install
npm run dev
```

#### 启动前端服务：
```bash
cd /Users/shixianping/smart-meet/frontend  
npm install
npm run dev
```

### 3. 常见问题修复

#### 问题1: 端口被占用
```bash
# 清理占用的端口
lsof -ti:3000 | xargs kill -9
lsof -ti:5001 | xargs kill -9
```

#### 问题2: 依赖缺失
```bash
# 重新安装依赖
cd /Users/shixianping/smart-meet/frontend
rm -rf node_modules package-lock.json
npm install

cd /Users/shixianping/smart-meet/backend
rm -rf node_modules package-lock.json
npm install
```

#### 问题3: API端口不匹配
确保以下文件中的端口配置正确：
- `/frontend/src/services/api.ts` - API_BASE_URL 应该是 `http://localhost:5001/api`
- `/frontend/vite.config.ts` - proxy target 应该是 `http://localhost:5001`

#### 问题4: 环境变量
创建 `/frontend/.env.local` 文件：
```
VITE_API_URL=http://localhost:5001/api
VITE_WS_URL=ws://localhost:5001
```

### 4. 一键启动脚本

使用自动化脚本：
```bash
cd /Users/shixianping/smart-meet
node debug-start.js
```

### 5. 验证服务状态

检查服务是否正常运行：
```bash
# 检查后端API
curl http://localhost:5001/api/health

# 检查前端服务
curl http://localhost:3000
```

### 6. 浏览器访问

确保在浏览器中访问：
- 前端应用: http://localhost:3000
- 后端API: http://localhost:5001/api/health

## 成功启动标志

当看到以下输出时，表示启动成功：

**后端输出：**
```
🚀 SmartMeet AI API 启动成功!
📊 服务信息:
   - 端口: 5001
   - 环境: development
```

**前端输出：**
```
VITE v4.5.14  ready in 213 ms
➜  Local:   http://localhost:3000/
```

## 联系支持

如果问题仍然存在，请检查：
1. Node.js 版本 (建议 18+)
2. npm 版本 (建议 9+)
3. 网络防火墙设置
4. 系统权限问题