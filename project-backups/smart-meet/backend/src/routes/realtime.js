import express from 'express';
import multer from 'multer';
import RealAgentService from '../services/RealAgentService.js';
import RealTranscriptionService from '../services/RealTranscriptionService.js';

const router = express.Router();
const realAgentService = new RealAgentService();
const realTranscriptionService = new RealTranscriptionService();

// 配置文件上传
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB
  }
});

// 启动真实协作会话
router.post('/collaboration/start', async (req, res) => {
  try {
    const { meetingData } = req.body;
    
    const session = await realAgentService.startRealCollaboration(meetingData);
    
    res.json({
      success: true,
      message: '真实多智能体协作已启动',
      data: {
        sessionId: session.id,
        agents: session.agents,
        currentPhase: session.currentPhase,
        startTime: session.startTime
      }
    });
  } catch (error) {
    console.error('启动协作失败:', error);
    res.status(500).json({
      success: false,
      message: '启动协作失败',
      error: error.message
    });
  }
});

// 实时音频转录和智能体分析
router.post('/transcribe/:sessionId', upload.single('audio'), async (req, res) => {
  try {
    const { sessionId } = req.params;
    const audioFile = req.file;
    
    if (!audioFile) {
      return res.status(400).json({
        success: false,
        message: '音频文件不能为空'
      });
    }

    console.log(`🎙️ 收到音频转录请求，会话: ${sessionId}, 大小: ${audioFile.size}字节`);

    // 1. 真实音频转录
    const transcriptionResult = await realTranscriptionService.transcribeAudio(
      audioFile.buffer,
      {
        language: 'zh-CN',
        sessionId: sessionId
      }
    );

    if (!transcriptionResult.success) {
      throw new Error('音频转录失败');
    }

    // 2. 创建转录片段
    const transcriptionSegment = {
      id: `${sessionId}_${Date.now()}`,
      sessionId,
      text: transcriptionResult.text,
      speaker: transcriptionResult.speaker,
      confidence: transcriptionResult.confidence,
      timestamp: new Date(),
      duration: transcriptionResult.duration || 0,
      language: transcriptionResult.language || 'zh-CN'
    };

    // 3. 触发智能体分析
    const updatedSession = await realAgentService.processTranscriptionData(
      sessionId, 
      transcriptionSegment
    );

    res.json({
      success: true,
      message: transcriptionResult.fallback ? '使用模拟转录结果' : '转录和分析成功',
      data: {
        transcription: transcriptionSegment,
        session: {
          currentPhase: updatedSession.currentPhase,
          progress: updatedSession.progress,
          agents: updatedSession.agents,
          transcriptionCount: updatedSession.transcriptionData.length
        },
        fallback: transcriptionResult.fallback || false
      }
    });

  } catch (error) {
    console.error('实时转录分析失败:', error);
    res.status(500).json({
      success: false,
      message: '实时转录分析失败',
      error: error.message
    });
  }
});

// 获取会话状态
router.get('/collaboration/:sessionId/status', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const status = realAgentService.getSessionStatus(sessionId);
    
    res.json({
      success: true,
      data: status
    });
  } catch (error) {
    console.error('获取会话状态失败:', error);
    res.status(404).json({
      success: false,
      message: '获取会话状态失败',
      error: error.message
    });
  }
});

// 获取智能体分析结果
router.get('/collaboration/:sessionId/results', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const results = realAgentService.getAnalysisResults(sessionId);
    
    res.json({
      success: true,
      data: results
    });
  } catch (error) {
    console.error('获取分析结果失败:', error);
    res.status(404).json({
      success: false,
      message: '获取分析结果失败',
      error: error.message
    });
  }
});

// 生成最终会议纪要
router.post('/collaboration/:sessionId/minutes', async (req, res) => {
  try {
    const { sessionId } = req.params;
    
    const minutes = realAgentService.generateFinalMinutes(sessionId);
    
    if (!minutes.success) {
      return res.status(400).json(minutes);
    }
    
    res.json({
      success: true,
      message: '会议纪要生成成功',
      data: minutes
    });
  } catch (error) {
    console.error('生成会议纪要失败:', error);
    res.status(500).json({
      success: false,
      message: '生成会议纪要失败',
      error: error.message
    });
  }
});

// 检查服务状态
router.get('/status', async (req, res) => {
  try {
    const transcriptionStatus = await realTranscriptionService.checkServiceAvailability();
    const llmStatus = await realAgentService.llmService.getModelStatus();
    
    res.json({
      success: true,
      data: {
        transcription: transcriptionStatus,
        llm: llmStatus,
        agents: {
          available: true,
          count: realAgentService.agents.size,
          types: Array.from(realAgentService.agents.keys())
        },
        timestamp: new Date()
      }
    });
  } catch (error) {
    console.error('检查服务状态失败:', error);
    res.status(500).json({
      success: false,
      message: '检查服务状态失败',
      error: error.message
    });
  }
});

// 测试端点 - 模拟完整流程
router.post('/test/full-flow', async (req, res) => {
  try {
    const testMeetingData = {
      title: '测试会议',
      participants: ['张三', '李四', '王五'],
      date: new Date().toISOString()
    };

    // 1. 启动协作
    const session = await realAgentService.startRealCollaboration(testMeetingData);
    
    // 2. 模拟转录数据
    const mockTranscriptions = [
      { text: '大家好，今天我们讨论产品功能优化', speaker: '张三' },
      { text: '我认为用户体验是最重要的', speaker: '李四' },
      { text: '预算控制在50万以内', speaker: '王五' }
    ];

    for (const mock of mockTranscriptions) {
      const segment = {
        id: `${session.id}_${Date.now()}_${Math.random()}`,
        sessionId: session.id,
        text: mock.text,
        speaker: mock.speaker,
        confidence: 0.9,
        timestamp: new Date(),
        duration: 3000,
        language: 'zh-CN'
      };
      
      await realAgentService.processTranscriptionData(session.id, segment);
    }

    // 3. 等待分析完成
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 4. 生成纪要
    const minutes = realAgentService.generateFinalMinutes(session.id);

    res.json({
      success: true,
      message: '完整流程测试成功',
      data: {
        session: realAgentService.getSessionStatus(session.id),
        minutes: minutes
      }
    });

  } catch (error) {
    console.error('测试完整流程失败:', error);
    res.status(500).json({
      success: false,
      message: '测试完整流程失败',
      error: error.message
    });
  }
});

export default router;