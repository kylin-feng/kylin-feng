import express from 'express';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import config from './config/index.js';
import corsMiddleware from './middleware/cors.js';
import { errorHandler, notFoundHandler } from './middleware/errorHandler.js';
import apiRoutes from './routes/index.js';
import WebSocketService from './services/WebSocketService.js';

const app = express();

// 安全中间件
app.use(helmet({
  crossOriginResourcePolicy: { policy: "cross-origin" }
}));

// CORS中间件
app.use(corsMiddleware);

// 压缩中间件
app.use(compression());

// 日志中间件
if (config.nodeEnv === 'development') {
  app.use(morgan('dev'));
} else {
  app.use(morgan('combined'));
}

// 解析中间件
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// 静态文件服务
app.use('/uploads', express.static('uploads'));

// API路由
app.use('/api', apiRoutes);

// 根路径
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: '欢迎使用 SmartMeet AI API',
    data: {
      name: 'SmartMeet AI',
      version: '1.0.0',
      description: '多智能体协作的智能会议管理助手',
      documentation: '/api/info',
      health: '/api/health'
    },
    timestamp: new Date().toISOString()
  });
});

// 404处理
app.use(notFoundHandler);

// 全局错误处理
app.use(errorHandler);

// 启动服务器
const PORT = config.port;
const server = app.listen(PORT, () => {
  console.log(`
🚀 SmartMeet AI API 启动成功!

📊 服务信息:
   - 端口: ${PORT}
   - 环境: ${config.nodeEnv}
   - API基础地址: http://localhost:${PORT}/api

🤖 智能体服务:
   - 智能体管理: /api/agents
   - 协作启动: /api/agents/collaborate
   - 健康检查: /api/health

🔧 开发工具:
   - API文档: http://localhost:${PORT}/api/info
   - 健康检查: http://localhost:${PORT}/api/health

⚠️  注意: 
   - 请确保在 .env 文件中配置大模型API密钥
   - 当前为开发模式，使用模拟数据
  `);
});

// 初始化WebSocket服务
const wsService = new WebSocketService();
wsService.initialize(server);

// 导出WebSocket服务供其他模块使用
export { wsService };

// 优雅关闭
process.on('SIGTERM', () => {
  console.log('收到 SIGTERM 信号，正在优雅关闭...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('收到 SIGINT 信号，正在优雅关闭...');
  process.exit(0);
});

export default app;