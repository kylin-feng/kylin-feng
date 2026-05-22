import express from 'express';
import agentRoutes from './agents.js';
import transcriptionRoutes from './transcription.js';
import realtimeRoutes from './realtime.js';

const router = express.Router();

// API路由配置
router.use('/agents', agentRoutes);
router.use('/transcription', transcriptionRoutes);
router.use('/realtime', realtimeRoutes);

// 健康检查
router.get('/health', (req, res) => {
  res.json({
    success: true,
    message: 'SmartMeet AI API 运行正常',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// API信息
router.get('/info', (req, res) => {
  res.json({
    success: true,
    data: {
      name: 'SmartMeet AI API',
      version: '1.0.0',
      description: '多智能体协作的智能会议管理助手 API',
      features: [
        '多智能体协作',
        '实时会议记录',
        '智能纪要生成',
        '语音识别转换',
        '内容分析处理'
      ],
      endpoints: {
        agents: '/api/agents',
        collaboration: '/api/agents/collaborate',
        transcription: '/api/transcription',
        realtime: '/api/realtime',
        health: '/api/health'
      }
    },
    timestamp: new Date().toISOString()
  });
});

export default router;