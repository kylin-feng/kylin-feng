import { v4 as uuidv4 } from 'uuid';
import LLMService from './LLMService.js';

class RealAgentService {
  constructor() {
    this.llmService = new LLMService();
    this.sessions = new Map();
    this.agents = new Map();
    
    // 初始化智能体
    this.initializeAgents();
  }

  initializeAgents() {
    const agentConfigs = [
      {
        id: 'recorder-agent',
        name: '记录员',
        role: 'Recorder Agent',
        description: '专治记录恐惧症！全程自动听写，再也不用边听边记到手抽筋，彻底解放双手',
        capabilities: ['多人语音识别', '说话人分离', '实时转录', '语音降噪', '语言模型优化'],
        prompt: `你是SmartMeet AI的记录员智能体。你的任务是：
1. 整理和格式化会议转录内容
2. 标注发言人和时间戳
3. 修正语音识别错误
4. 提供清晰的会议记录

请将以下转录内容整理成规范的会议记录格式：`
      },
      {
        id: 'analyst-agent',
        name: '分析师',
        role: 'Analyst Agent',
        description: '专治思考懒惰症！AI大脑24小时不休息，帮你提炼重点，再懒也能抓住关键信息',
        capabilities: ['内容分析', '关键信息提取', '情感分析', '趋势识别', '数据挖掘'],
        prompt: `你是SmartMeet AI的分析师智能体。你的任务是：
1. 分析以下会议内容，提取关键信息
2. 识别重要决策和行动点
3. 总结主要讨论议题
4. 分析参与者观点和情感倾向

请分析以下会议内容：`
      },
      {
        id: 'secretary-agent',
        name: '秘书',
        role: 'Secretary Agent',
        description: '专治任务拖延症！自动安排待办事项，强制分配责任人，让拖延症无处可逃',
        capabilities: ['任务管理', '时间规划', '责任分配', '日程协调', '提醒服务'],
        prompt: `你是SmartMeet AI的秘书智能体。你的任务是：
1. 从会议内容中提取待办事项
2. 识别责任人和截止时间
3. 制定跟进计划
4. 安排后续会议

请基于以下会议内容制定待办事项清单：`
      },
      {
        id: 'editor-agent',
        name: '编辑',
        role: 'Editor Agent',
        description: '专治整理懒惰症！自动润色文字、统一格式，懒得修改也能输出完美文档',
        capabilities: ['文本优化', '格式标准化', '语言润色', '风格统一', '可读性提升'],
        prompt: `你是SmartMeet AI的编辑智能体。你的任务是：
1. 优化会议纪要的语言表达
2. 统一文档格式和风格
3. 提高文本可读性
4. 确保专业性和准确性

请对以下会议内容进行语言优化和格式整理：`
      },
      {
        id: 'qa-agent',
        name: '质检',
        role: 'QA Agent',
        description: '专治检查懒惰症！AI火眼金睛找错误，比强迫症还仔细，确保质量万无一失',
        capabilities: ['逻辑验证', '准确性检查', '质量控制', '错误检测', '一致性审核'],
        prompt: `你是SmartMeet AI的质检智能体。你的任务是：
1. 检查会议纪要的逻辑一致性
2. 验证信息的准确性
3. 识别潜在的错误或遗漏
4. 提供质量改进建议

请对以下会议内容进行质量检查：`
      }
    ];

    agentConfigs.forEach(config => {
      this.agents.set(config.id, {
        ...config,
        status: 'idle',
        progress: 0,
        lastUpdate: new Date(),
        results: []
      });
    });
  }

  // 启动真实的智能体协作
  async startRealCollaboration(meetingData) {
    const sessionId = uuidv4();
    
    console.log(`🚀 启动真实多智能体协作会话: ${sessionId}`);
    
    const session = {
      id: sessionId,
      meetingData,
      agents: Array.from(this.agents.values()).map(agent => ({
        ...agent,
        status: 'idle',
        progress: 0
      })),
      transcriptionData: [],
      results: {},
      currentPhase: 'preparation',
      progress: 0,
      startTime: new Date(),
      estimatedEndTime: new Date(Date.now() + 60 * 60 * 1000)
    };

    this.sessions.set(sessionId, session);
    
    return session;
  }

  // 处理转录数据并触发智能体分析
  async processTranscriptionData(sessionId, transcriptionSegment) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`会话不存在: ${sessionId}`);
    }

    console.log(`📝 处理转录数据: ${transcriptionSegment.text.substring(0, 50)}...`);
    
    // 添加转录数据
    session.transcriptionData.push(transcriptionSegment);
    session.currentPhase = 'recording';

    // 当累积足够的转录数据时，启动智能体分析
    if (session.transcriptionData.length >= 3 || this.shouldTriggerAnalysis(session)) {
      await this.triggerAgentAnalysis(sessionId);
    }

    return session;
  }

  // 判断是否应该触发分析
  shouldTriggerAnalysis(session) {
    const lastTrigger = session.lastAnalysisTrigger || session.startTime;
    const timeSinceLastTrigger = Date.now() - lastTrigger.getTime();
    
    // 每30秒或累积5个转录片段时触发一次分析
    return timeSinceLastTrigger > 30000 || session.transcriptionData.length >= 5;
  }

  // 触发智能体分析
  async triggerAgentAnalysis(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) return;

    console.log(`🤖 启动智能体分析，会话: ${sessionId}`);
    
    session.currentPhase = 'processing';
    session.lastAnalysisTrigger = new Date();

    // 准备会议内容
    const meetingContent = this.prepareMeetingContent(session.transcriptionData);
    
    // 并行执行所有智能体
    const agentPromises = session.agents.map(agent => 
      this.executeAgent(sessionId, agent.id, meetingContent)
    );

    try {
      const results = await Promise.allSettled(agentPromises);
      
      // 处理结果
      results.forEach((result, index) => {
        const agent = session.agents[index];
        if (result.status === 'fulfilled') {
          agent.status = 'completed';
          agent.progress = 100;
          session.results[agent.id] = result.value;
          console.log(`✅ ${agent.name}完成任务`);
        } else {
          agent.status = 'error';
          console.error(`❌ ${agent.name}执行失败:`, result.reason);
        }
      });

      session.currentPhase = 'review';
      session.progress = this.calculateOverallProgress(session.agents);
      
      // 如果所有智能体都完成了，进入完成阶段
      if (session.agents.every(agent => agent.status === 'completed')) {
        session.currentPhase = 'completed';
        session.progress = 100;
        session.endTime = new Date();
        console.log(`🎉 智能体协作完成，会话: ${sessionId}`);
      }

    } catch (error) {
      console.error('智能体分析执行失败:', error);
      session.currentPhase = 'error';
    }

    return session;
  }

  // 执行单个智能体
  async executeAgent(sessionId, agentId, meetingContent) {
    const agent = this.agents.get(agentId);
    if (!agent) {
      throw new Error(`智能体不存在: ${agentId}`);
    }

    console.log(`🔄 执行智能体: ${agent.name}`);
    
    // 更新状态
    const session = this.sessions.get(sessionId);
    const sessionAgent = session.agents.find(a => a.id === agentId);
    if (sessionAgent) {
      sessionAgent.status = 'working';
      sessionAgent.progress = 0;
    }

    try {
      // 模拟进度更新
      const progressInterval = setInterval(() => {
        if (sessionAgent && sessionAgent.status === 'working') {
          sessionAgent.progress = Math.min(sessionAgent.progress + Math.random() * 20, 90);
        }
      }, 1000);

      // 调用LLM服务
      const prompt = agent.prompt + '\n\n' + meetingContent;
      let result;

      if (agentId === 'qa-agent') {
        // QA智能体使用DeepSeek
        result = await this.llmService.callDeepSeek(prompt, {
          temperature: 0.1,
          maxTokens: 1500
        });
      } else {
        // 其他智能体使用通义千问
        result = await this.llmService.callQianwen(prompt, {
          temperature: 0.3,
          maxTokens: 2000
        });
      }

      clearInterval(progressInterval);

      if (sessionAgent) {
        sessionAgent.progress = 100;
        sessionAgent.status = 'completed';
        sessionAgent.lastUpdate = new Date();
      }

      return {
        agentId,
        agentName: agent.name,
        success: result.success,
        content: result.content,
        model: result.model,
        timestamp: new Date(),
        processingTime: Date.now() - session.lastAnalysisTrigger.getTime()
      };

    } catch (error) {
      if (sessionAgent) {
        sessionAgent.status = 'error';
        sessionAgent.progress = 0;
      }
      throw error;
    }
  }

  // 准备会议内容
  prepareMeetingContent(transcriptionData) {
    if (!transcriptionData || transcriptionData.length === 0) {
      return '暂无会议内容。';
    }

    return transcriptionData
      .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
      .map((segment, index) => {
        const time = new Date(segment.timestamp).toLocaleTimeString();
        return `[${time}] ${segment.speaker || '发言人'}: ${segment.text}`;
      })
      .join('\n\n');
  }

  // 计算整体进度
  calculateOverallProgress(agents) {
    if (!agents || agents.length === 0) return 0;
    
    const totalProgress = agents.reduce((sum, agent) => sum + agent.progress, 0);
    return Math.round(totalProgress / agents.length);
  }

  // 获取会话状态
  getSessionStatus(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`会话不存在: ${sessionId}`);
    }

    return {
      sessionId: session.id,
      currentPhase: session.currentPhase,
      progress: session.progress,
      agents: session.agents,
      transcriptionCount: session.transcriptionData.length,
      startTime: session.startTime,
      endTime: session.endTime,
      estimatedEndTime: session.estimatedEndTime
    };
  }

  // 获取分析结果
  getAnalysisResults(sessionId) {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`会话不存在: ${sessionId}`);
    }

    return {
      sessionId: session.id,
      status: session.currentPhase,
      results: session.results,
      meetingContent: this.prepareMeetingContent(session.transcriptionData),
      generatedAt: new Date()
    };
  }

  // 生成最终会议纪要
  generateFinalMinutes(sessionId) {
    const results = this.getAnalysisResults(sessionId);
    
    if (!results.results || Object.keys(results.results).length === 0) {
      return {
        success: false,
        message: '智能体分析结果不完整，无法生成会议纪要'
      };
    }

    // 组合所有智能体的结果
    const minutes = {
      executive: this.generateExecutiveVersion(results),
      technical: this.generateTechnicalVersion(results),
      management: this.generateManagementVersion(results),
      client: this.generateClientVersion(results)
    };

    return {
      success: true,
      sessionId,
      minutes,
      generatedAt: new Date(),
      agentsUsed: Object.keys(results.results)
    };
  }

  // 生成高管版本纪要
  generateExecutiveVersion(results) {
    const analyst = results.results['analyst-agent'];
    const secretary = results.results['secretary-agent'];
    
    return {
      title: '高管决策摘要',
      content: analyst?.content || '分析结果不可用',
      actionItems: secretary?.content || '待办事项不可用',
      keyDecisions: this.extractKeyDecisions(results),
      summary: '基于AI智能体分析的高管决策摘要'
    };
  }

  // 生成技术版本纪要
  generateTechnicalVersion(results) {
    const recorder = results.results['recorder-agent'];
    const editor = results.results['editor-agent'];
    
    return {
      title: '技术实现详情',
      content: recorder?.content || '会议记录不可用',
      editedContent: editor?.content || '编辑内容不可用',
      technicalDetails: this.extractTechnicalDetails(results),
      summary: '面向技术团队的详细会议记录'
    };
  }

  // 生成管理版本纪要
  generateManagementVersion(results) {
    const secretary = results.results['secretary-agent'];
    const qa = results.results['qa-agent'];
    
    return {
      title: '项目管理要点',
      content: secretary?.content || '管理内容不可用',
      qualityCheck: qa?.content || '质量检查不可用',
      riskAssessment: this.extractRisks(results),
      summary: '面向项目管理的执行要点'
    };
  }

  // 生成客户版本纪要
  generateClientVersion(results) {
    const editor = results.results['editor-agent'];
    const analyst = results.results['analyst-agent'];
    
    return {
      title: '客户沟通摘要',
      content: editor?.content || '客户内容不可用',
      businessValue: this.extractBusinessValue(results),
      nextSteps: this.extractNextSteps(results),
      summary: '面向客户的商业价值总结'
    };
  }

  // 辅助方法
  extractKeyDecisions(results) {
    // 从分析结果中提取关键决策
    return '基于AI分析提取的关键决策点';
  }

  extractTechnicalDetails(results) {
    return '基于AI分析提取的技术实现细节';
  }

  extractRisks(results) {
    return '基于AI分析识别的项目风险';
  }

  extractBusinessValue(results) {
    return '基于AI分析总结的商业价值';
  }

  extractNextSteps(results) {
    return '基于AI分析制定的后续步骤';
  }
}

export default RealAgentService;