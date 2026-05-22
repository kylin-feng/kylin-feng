// 语音转录服务
import EventEmitter from 'events';

class TranscriptionService extends EventEmitter {
  constructor() {
    super();
    this.isProcessing = false;
    this.sessions = new Map(); // 存储转录会话
    this.speakers = new Map(); // 存储发言人信息
  }

  // 开始转录会话
  startTranscriptionSession(sessionId, options = {}) {
    const session = {
      id: sessionId,
      startTime: new Date(),
      options: {
        language: options.language || 'zh-CN',
        enableSpeakerDiarization: options.enableSpeakerDiarization !== false,
        enablePunctuation: options.enablePunctuation !== false,
        enableWordTimestamps: options.enableWordTimestamps || false,
        ...options
      },
      segments: [],
      speakers: new Set(),
      status: 'active'
    };

    this.sessions.set(sessionId, session);
    
    console.log(`转录会话已开始: ${sessionId}`);
    this.emit('session_started', { sessionId, session });
    
    return session;
  }

  // 停止转录会话
  stopTranscriptionSession(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`转录会话不存在: ${sessionId}`);
    }

    session.status = 'completed';
    session.endTime = new Date();
    session.duration = session.endTime - session.startTime;

    console.log(`转录会话已停止: ${sessionId}`);
    this.emit('session_stopped', { sessionId, session });

    return session;
  }

  // 处理音频数据
  async processAudioData(sessionId, audioData, metadata = {}) {
    const session = this.sessions.get(sessionId);
    if (!session || session.status !== 'active') {
      throw new Error(`无效的转录会话: ${sessionId}`);
    }

    try {
      this.isProcessing = true;

      // 模拟语音转录处理
      const transcriptionResult = await this.simulateTranscription(audioData, session.options);
      
      // 创建转录片段
      const segment = {
        id: `${sessionId}_${Date.now()}`,
        sessionId,
        text: transcriptionResult.text,
        speaker: transcriptionResult.speaker,
        startTime: transcriptionResult.startTime,
        endTime: transcriptionResult.endTime,
        confidence: transcriptionResult.confidence,
        language: transcriptionResult.language,
        words: transcriptionResult.words || [],
        metadata: {
          audioLength: metadata.audioLength || 0,
          sampleRate: metadata.sampleRate || 16000,
          channels: metadata.channels || 1,
          ...metadata
        },
        timestamp: new Date()
      };

      // 添加到会话
      session.segments.push(segment);
      session.speakers.add(transcriptionResult.speaker);

      // 发送实时更新
      this.emit('transcription_update', {
        sessionId,
        segment,
        totalSegments: session.segments.length
      });

      console.log(`转录片段已生成: ${segment.id}`);
      return segment;

    } catch (error) {
      console.error('转录处理失败:', error);
      this.emit('transcription_error', { sessionId, error: error.message });
      throw error;
    } finally {
      this.isProcessing = false;
    }
  }

  // 模拟语音转录 (在实际项目中，这里会调用真实的语音转录API)
  async simulateTranscription(audioData, options) {
    // 模拟处理延迟
    await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

    const mockTexts = [
      '大家好，今天我们来讨论项目的最新进展情况。',
      '关于这个功能模块，我认为我们需要进一步优化用户体验。',
      '从数据分析的角度来看，用户反馈整体是积极的。',
      '我建议我们在下个迭代中加入更多的智能化特性。',
      '这个方案的技术实现难度相对较低，可以优先考虑。',
      '我们需要确保系统的稳定性和可扩展性。',
      '用户界面设计方面，建议采用更加简洁的风格。',
      '安全性是我们必须重点关注的问题。',
      '测试覆盖率还需要进一步提升。',
      '项目时间线整体符合预期，质量也达到了要求。'
    ];

    const mockSpeakers = [
      '张经理', '李工程师', '王设计师', '刘分析师', '陈产品经理'
    ];

    const selectedText = mockTexts[Math.floor(Math.random() * mockTexts.length)];
    const selectedSpeaker = mockSpeakers[Math.floor(Math.random() * mockSpeakers.length)];
    
    const duration = Math.floor(Math.random() * 8000) + 2000; // 2-10秒
    const startTime = Date.now();
    const endTime = startTime + duration;

    // 模拟单词级别的时间戳
    const words = options.enableWordTimestamps ? this.generateWordTimestamps(selectedText, startTime, duration) : [];

    return {
      text: selectedText,
      speaker: selectedSpeaker,
      startTime,
      endTime,
      confidence: Math.random() * 0.3 + 0.7, // 70%-100%置信度
      language: options.language,
      words
    };
  }

  // 生成单词级时间戳
  generateWordTimestamps(text, startTime, duration) {
    const words = text.split(/[\s，。、！？；：]/);
    const wordDuration = duration / words.length;
    
    return words.filter(word => word.trim()).map((word, index) => ({
      word: word.trim(),
      startTime: startTime + (index * wordDuration),
      endTime: startTime + ((index + 1) * wordDuration),
      confidence: Math.random() * 0.2 + 0.8
    }));
  }

  // 获取会话转录结果
  getSessionTranscription(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`转录会话不存在: ${sessionId}`);
    }

    return {
      sessionId: session.id,
      startTime: session.startTime,
      endTime: session.endTime,
      duration: session.duration,
      status: session.status,
      totalSegments: session.segments.length,
      speakers: Array.from(session.speakers),
      segments: session.segments,
      options: session.options
    };
  }

  // 导出转录结果
  exportTranscription(sessionId, format = 'txt') {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`转录会话不存在: ${sessionId}`);
    }

    switch (format.toLowerCase()) {
      case 'txt':
        return this.exportToText(session);
      case 'json':
        return this.exportToJSON(session);
      case 'srt':
        return this.exportToSRT(session);
      default:
        throw new Error(`不支持的导出格式: ${format}`);
    }
  }

  // 导出为文本格式
  exportToText(session) {
    const header = `会议转录记录\n时间: ${session.startTime.toLocaleString()}\n总时长: ${Math.round(session.duration / 1000)}秒\n参与者: ${Array.from(session.speakers).join(', ')}\n\n`;
    
    const content = session.segments.map((segment, index) => {
      const time = new Date(segment.startTime).toLocaleTimeString();
      return `[${time}] ${segment.speaker}: ${segment.text}`;
    }).join('\n\n');

    return header + content;
  }

  // 导出为JSON格式
  exportToJSON(session) {
    return JSON.stringify({
      session: {
        id: session.id,
        startTime: session.startTime,
        endTime: session.endTime,
        duration: session.duration,
        status: session.status,
        options: session.options
      },
      speakers: Array.from(session.speakers),
      segments: session.segments
    }, null, 2);
  }

  // 导出为SRT字幕格式
  exportToSRT(session) {
    return session.segments.map((segment, index) => {
      const startTime = this.formatSRTTime(segment.startTime);
      const endTime = this.formatSRTTime(segment.endTime);
      
      return `${index + 1}\n${startTime} --> ${endTime}\n${segment.speaker}: ${segment.text}\n`;
    }).join('\n');
  }

  // 格式化SRT时间
  formatSRTTime(timestamp) {
    const date = new Date(timestamp);
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');
    const milliseconds = date.getMilliseconds().toString().padStart(3, '0');
    
    return `${hours}:${minutes}:${seconds},${milliseconds}`;
  }

  // 搜索转录内容
  searchTranscription(sessionId, query, options = {}) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`转录会话不存在: ${sessionId}`);
    }

    const {
      caseSensitive = false,
      wholeWord = false,
      speaker = null
    } = options;

    let searchQuery = caseSensitive ? query : query.toLowerCase();
    
    const results = session.segments.filter(segment => {
      // 发言人过滤
      if (speaker && segment.speaker !== speaker) {
        return false;
      }

      let text = caseSensitive ? segment.text : segment.text.toLowerCase();
      
      if (wholeWord) {
        const regex = new RegExp(`\\b${searchQuery}\\b`, caseSensitive ? 'g' : 'gi');
        return regex.test(text);
      } else {
        return text.includes(searchQuery);
      }
    });

    return {
      query,
      totalResults: results.length,
      segments: results
    };
  }

  // 获取发言人统计
  getSpeakerStatistics(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`转录会话不存在: ${sessionId}`);
    }

    const stats = {};
    
    session.segments.forEach(segment => {
      if (!stats[segment.speaker]) {
        stats[segment.speaker] = {
          name: segment.speaker,
          segmentCount: 0,
          totalDuration: 0,
          wordCount: 0,
          averageConfidence: 0
        };
      }

      const speakerStats = stats[segment.speaker];
      speakerStats.segmentCount++;
      speakerStats.totalDuration += (segment.endTime - segment.startTime);
      speakerStats.wordCount += segment.text.split(/\s+/).length;
      speakerStats.averageConfidence = (speakerStats.averageConfidence + segment.confidence) / speakerStats.segmentCount;
    });

    return Object.values(stats);
  }

  // 清理过期会话
  cleanupExpiredSessions(maxAge = 24 * 60 * 60 * 1000) { // 默认24小时
    const now = Date.now();
    let cleanedCount = 0;

    for (const [sessionId, session] of this.sessions.entries()) {
      if (now - session.startTime.getTime() > maxAge) {
        this.sessions.delete(sessionId);
        cleanedCount++;
        console.log(`已清理过期转录会话: ${sessionId}`);
      }
    }

    return cleanedCount;
  }
}

export default TranscriptionService;