import express from 'express';
import multer from 'multer';
import TranscriptionService from '../services/TranscriptionService.js';
import RealTranscriptionService from '../services/RealTranscriptionService.js';

const router = express.Router();
const transcriptionService = new TranscriptionService();
const realTranscriptionService = new RealTranscriptionService();

// 配置文件上传
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB
  },
  fileFilter: (req, file, cb) => {
    // 允许的音频格式
    const allowedMimes = [
      'audio/wav',
      'audio/mp3',
      'audio/mpeg',
      'audio/webm',
      'audio/ogg'
    ];
    
    if (allowedMimes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('不支持的音频格式'), false);
    }
  }
});

// 开始转录会话
router.post('/sessions', async (req, res) => {
  try {
    const { sessionId, options = {} } = req.body;
    
    if (!sessionId) {
      return res.status(400).json({
        success: false,
        message: '会话ID不能为空'
      });
    }

    const session = transcriptionService.startTranscriptionSession(sessionId, options);
    
    res.json({
      success: true,
      message: '转录会话已开始',
      data: {
        sessionId: session.id,
        startTime: session.startTime,
        options: session.options,
        status: session.status
      }
    });
  } catch (error) {
    console.error('开始转录会话失败:', error);
    res.status(500).json({
      success: false,
      message: '开始转录会话失败',
      error: error.message
    });
  }
});

// 停止转录会话
router.delete('/sessions/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const session = transcriptionService.stopTranscriptionSession(sessionId);
    
    res.json({
      success: true,
      message: '转录会话已停止',
      data: {
        sessionId: session.id,
        startTime: session.startTime,
        endTime: session.endTime,
        duration: session.duration,
        totalSegments: session.segments.length,
        speakers: Array.from(session.speakers)
      }
    });
  } catch (error) {
    console.error('停止转录会话失败:', error);
    res.status(404).json({
      success: false,
      message: '停止转录会话失败',
      error: error.message
    });
  }
});

// 上传音频进行转录
router.post('/sessions/:sessionId/transcribe', upload.single('audio'), async (req, res) => {
  try {
    const { sessionId } = req.params;
    const audioFile = req.file;
    
    if (!audioFile) {
      return res.status(400).json({
        success: false,
        message: '音频文件不能为空'
      });
    }

    const metadata = {
      originalName: audioFile.originalname,
      mimeType: audioFile.mimetype,
      size: audioFile.size,
      uploadTime: new Date()
    };

    // 使用真实转录服务
    const transcriptionResult = await realTranscriptionService.transcribeAudio(
      audioFile.buffer,
      {
        language: 'zh-CN',
        sessionId: sessionId
      }
    );

    if (transcriptionResult.success) {
      // 创建转录片段
      const segment = {
        id: `${sessionId}_${Date.now()}`,
        sessionId,
        text: transcriptionResult.text,
        speaker: transcriptionResult.speaker,
        confidence: transcriptionResult.confidence,
        timestamp: new Date(),
        duration: transcriptionResult.duration || 0,
        language: transcriptionResult.language || 'zh-CN',
        metadata: {
          ...metadata,
          fallback: transcriptionResult.fallback || false,
          provider: transcriptionResult.provider || 'unknown'
        }
      };

      // 同时保存到原有的转录服务
      await transcriptionService.processAudioData(sessionId, audioFile.buffer, metadata);
      
      res.json({
        success: true,
        message: transcriptionResult.fallback ? '使用模拟转录结果' : '音频转录成功',
        data: segment
      });
    } else {
      throw new Error('转录失败');
    }
  } catch (error) {
    console.error('音频转录失败:', error);
    res.status(500).json({
      success: false,
      message: '音频转录失败',
      error: error.message
    });
  }
});

// 获取转录结果
router.get('/sessions/:sessionId', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const transcription = transcriptionService.getSessionTranscription(sessionId);
    
    res.json({
      success: true,
      message: '获取转录结果成功',
      data: transcription
    });
  } catch (error) {
    console.error('获取转录结果失败:', error);
    res.status(404).json({
      success: false,
      message: '获取转录结果失败',
      error: error.message
    });
  }
});

// 导出转录结果
router.get('/sessions/:sessionId/export', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { format = 'txt' } = req.query;
    
    const exportedData = transcriptionService.exportTranscription(sessionId, format);
    
    // 设置响应头
    const contentTypes = {
      txt: 'text/plain',
      json: 'application/json',
      srt: 'application/x-subrip'
    };
    
    const fileExtensions = {
      txt: 'txt',
      json: 'json',
      srt: 'srt'
    };
    
    res.setHeader('Content-Type', contentTypes[format] || 'text/plain');
    res.setHeader('Content-Disposition', `attachment; filename="transcription-${sessionId}.${fileExtensions[format]}"`);
    
    res.send(exportedData);
  } catch (error) {
    console.error('导出转录结果失败:', error);
    res.status(500).json({
      success: false,
      message: '导出转录结果失败',
      error: error.message
    });
  }
});

// 搜索转录内容
router.post('/sessions/:sessionId/search', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const { query, options = {} } = req.body;
    
    if (!query) {
      return res.status(400).json({
        success: false,
        message: '搜索关键词不能为空'
      });
    }

    const results = transcriptionService.searchTranscription(sessionId, query, options);
    
    res.json({
      success: true,
      message: '搜索完成',
      data: results
    });
  } catch (error) {
    console.error('搜索转录内容失败:', error);
    res.status(500).json({
      success: false,
      message: '搜索转录内容失败',
      error: error.message
    });
  }
});

// 获取发言人统计
router.get('/sessions/:sessionId/speakers', async (req, res) => {
  try {
    const { sessionId } = req.params;
    const stats = transcriptionService.getSpeakerStatistics(sessionId);
    
    res.json({
      success: true,
      message: '获取发言人统计成功',
      data: stats
    });
  } catch (error) {
    console.error('获取发言人统计失败:', error);
    res.status(500).json({
      success: false,
      message: '获取发言人统计失败',
      error: error.message
    });
  }
});

// 实时转录WebSocket事件处理
transcriptionService.on('transcription_update', (data) => {
  // 这里可以通过WebSocket推送实时更新
  // wsService可以从主应用导入
  console.log('转录更新:', data.segment.text);
});

transcriptionService.on('session_started', (data) => {
  console.log('转录会话开始:', data.sessionId);
});

transcriptionService.on('session_stopped', (data) => {
  console.log('转录会话结束:', data.sessionId);
});

transcriptionService.on('transcription_error', (data) => {
  console.error('转录错误:', data.error);
});

export default router;