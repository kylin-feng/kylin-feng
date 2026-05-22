import Agent from '../models/Agent.js';
import LLMService from './LLMService.js';
import { v4 as uuidv4 } from 'uuid';

// 智能体服务
export class AgentService {
  constructor() {
    this.agents = new Map();
    this.collaborations = new Map();
    this.llmService = new LLMService();
    this.initializeAgents();
  }

  // 初始化默认智能体
  initializeAgents() {
    const defaultAgents = [
      {
        id: 'recorder-agent',
        name: '记录员',
        role: 'Recorder Agent',
        description: '专注实时语音转文字和发言人识别',
        capabilities: ['语音识别', '说话人分离', '实时转录'],
        modelType: 'qianwen',
        config: {
          temperature: 0.1,
          maxTokens: 2000
        }
      },
      {
        id: 'analyst-agent', 
        name: '分析师',
        role: 'Analyst Agent',
        description: '提取关键信息和决策要点',
        capabilities: ['内容分析', '关键信息提取', '情感分析'],
        modelType: 'qianwen',
        config: {
          temperature: 0.3,
          maxTokens: 3000
        }
      },
      {
        id: 'secretary-agent',
        name: '秘书',
        role: 'Secretary Agent',
        description: '整理待办事项和责任分配',
        capabilities: ['任务管理', '时间规划', '责任分配'],
        modelType: 'qianwen',
        config: {
          temperature: 0.2,
          maxTokens: 2000
        }
      },
      {
        id: 'editor-agent',
        name: '编辑',
        role: 'Editor Agent',
        description: '优化语言表达和格式规范',
        capabilities: ['文本优化', '格式标准化', '语言润色'],
        modelType: 'qianwen',
        config: {
          temperature: 0.4,
          maxTokens: 2500
        }
      },
      {
        id: 'qa-agent',
        name: '质检',
        role: 'QA Agent',
        description: '验证信息准确性和逻辑检查',
        capabilities: ['逻辑验证', '准确性检查', '质量控制'],
        modelType: 'deepseek',
        config: {
          temperature: 0.1,
          maxTokens: 2000
        }
      }
    ];

    defaultAgents.forEach(agentData => {
      const agent = new Agent(agentData);
      this.agents.set(agent.id, agent);
    });
  }

  // 获取所有智能体
  getAllAgents() {
    return Array.from(this.agents.values()).map(agent => agent.toJSON());
  }

  // 获取单个智能体
  getAgent(agentId) {
    const agent = this.agents.get(agentId);
    return agent ? agent.toJSON() : null;
  }

  // 更新智能体状态
  updateAgentStatus(agentId, status, progress) {
    const agent = this.agents.get(agentId);
    if (agent) {
      agent.updateStatus(status, progress);
      return agent.toJSON();
    }
    return null;
  }

  // 启动智能体协作
  async startCollaboration(meetingData) {
    const sessionId = uuidv4();
    const agents = Array.from(this.agents.values()).map(agent => {
      agent.updateStatus('working', 0);
      return agent.toJSON();
    });

    const collaboration = {
      sessionId,
      agents,
      currentPhase: 'preparation',
      progress: 0,
      startTime: new Date().toISOString(),
      estimatedEndTime: new Date(Date.now() + 60 * 60 * 1000).toISOString(), // 1小时后
      meetingData,
      results: {}
    };

    this.collaborations.set(sessionId, collaboration);

    // 启动协作流程
    this.processCollaboration(sessionId);

    return collaboration;
  }

  // 处理协作流程
  async processCollaboration(sessionId) {
    const collaboration = this.collaborations.get(sessionId);
    if (!collaboration) return;

    try {
      // 阶段1: 准备阶段
      collaboration.currentPhase = 'preparation';
      await this.updateCollaborationProgress(sessionId, 10);

      // 阶段2: 录制阶段  
      collaboration.currentPhase = 'recording';
      await this.updateCollaborationProgress(sessionId, 30);

      // 阶段3: 处理阶段
      collaboration.currentPhase = 'processing';
      await this.processTranscription(sessionId);
      await this.updateCollaborationProgress(sessionId, 60);

      // 阶段4: 审核阶段
      collaboration.currentPhase = 'review';
      await this.reviewResults(sessionId);
      await this.updateCollaborationProgress(sessionId, 90);

      // 阶段5: 完成阶段
      collaboration.currentPhase = 'completed';
      await this.updateCollaborationProgress(sessionId, 100);

    } catch (error) {
      console.error('协作处理失败:', error);
      collaboration.currentPhase = 'error';
      collaboration.error = error.message;
    }
  }

  // 处理转录内容
  async processTranscription(sessionId) {
    const collaboration = this.collaborations.get(sessionId);
    if (!collaboration) return;

    // 模拟处理转录内容
    const mockTranscription = [
      { speaker: '张三', text: '今天我们讨论的主要议题是产品功能优化', timestamp: Date.now() - 60000 },
      { speaker: '李四', text: '我认为我们应该重点关注用户体验方面的改进', timestamp: Date.now() - 30000 },
      { speaker: '王五', text: '预算方面我们需要控制在50万以内', timestamp: Date.now() }
    ];

    // 各智能体并行处理
    const agentTasks = collaboration.agents.map(async (agentData) => {
      const agent = this.agents.get(agentData.id);
      if (!agent) return;

      agent.updateStatus('working', Math.random() * 50 + 25);

      try {
        let result;
        switch (agent.role) {
          case 'Recorder Agent':
            result = await this.recorderProcess(mockTranscription);
            break;
          case 'Analyst Agent':
            result = await this.analystProcess(mockTranscription);
            break;
          case 'Secretary Agent':
            result = await this.secretaryProcess(mockTranscription);
            break;
          case 'Editor Agent':
            result = await this.editorProcess(mockTranscription);
            break;
          case 'QA Agent':
            result = await this.qaProcess(mockTranscription);
            break;
        }

        collaboration.results[agent.id] = result;
        agent.updateStatus('completed', 100);

      } catch (error) {
        console.error(`${agent.name}处理失败:`, error);
        agent.updateStatus('error', 0);
      }
    });

    await Promise.all(agentTasks);
  }

  // 记录员智能体处理
  async recorderProcess(transcription) {
    const prompt = `请将以下会议转录内容进行整理和格式化：
${transcription.map(t => `${t.speaker}: ${t.text}`).join('\n')}

要求：
1. 保持原始内容的准确性
2. 添加时间戳
3. 标注发言人
4. 修正明显的语音识别错误`;

    return await this.llmService.callQianwen(prompt);
  }

  // 分析师智能体处理
  async analystProcess(transcription) {
    const prompt = `请分析以下会议内容，提取关键信息：
${transcription.map(t => `${t.speaker}: ${t.text}`).join('\n')}

请从以下维度分析：
1. 主要议题和讨论重点
2. 重要决策和结论
3. 关键数据和指标
4. 风险点和机会
5. 参与者观点和态度`;

    return await this.llmService.callQianwen(prompt);
  }

  // 秘书智能体处理
  async secretaryProcess(transcription) {
    const prompt = `请从以下会议内容中提取待办事项和任务分配：
${transcription.map(t => `${t.speaker}: ${t.text}`).join('\n')}

请整理出：
1. 明确的行动项目
2. 责任人分配
3. 完成时间要求
4. 优先级评估
5. 相关依赖关系`;

    return await this.llmService.callQianwen(prompt);
  }

  // 编辑智能体处理
  async editorProcess(transcription) {
    const prompt = `请对以下会议内容进行语言优化和格式规范：
${transcription.map(t => `${t.speaker}: ${t.text}`).join('\n')}

要求：
1. 优化语言表达，使其更加正式和专业
2. 统一格式和术语
3. 改善可读性
4. 保持内容完整性
5. 按照标准会议纪要格式整理`;

    return await this.llmService.callQianwen(prompt);
  }

  // 质检智能体处理
  async qaProcess(transcription) {
    const prompt = `请对以下会议内容进行质量检查和逻辑验证：
${transcription.map(t => `${t.speaker}: ${t.text}`).join('\n')}

检查内容：
1. 信息的逻辑一致性
2. 数据的准确性
3. 决策的合理性
4. 遗漏的重要信息
5. 可能的理解偏差`;

    return await this.llmService.callDeepSeek(prompt);
  }

  // 审核结果
  async reviewResults(sessionId) {
    const collaboration = this.collaborations.get(sessionId);
    if (!collaboration) return;

    // 模拟审核过程
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // 更新智能体状态为审核完成
    collaboration.agents.forEach(agentData => {
      const agent = this.agents.get(agentData.id);
      if (agent && agent.status !== 'error') {
        agent.updateStatus('completed', 100);
      }
    });
  }

  // 更新协作进度
  async updateCollaborationProgress(sessionId, progress) {
    const collaboration = this.collaborations.get(sessionId);
    if (collaboration) {
      collaboration.progress = progress;
      
      // 模拟处理时间
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }

  // 获取协作状态
  getCollaborationStatus(sessionId) {
    const collaboration = this.collaborations.get(sessionId);
    if (!collaboration) return null;

    // 更新智能体状态
    collaboration.agents = collaboration.agents.map(agentData => {
      const agent = this.agents.get(agentData.id);
      return agent ? agent.toJSON() : agentData;
    });

    return collaboration;
  }

  // 停止协作
  stopCollaboration(sessionId) {
    const collaboration = this.collaborations.get(sessionId);
    if (collaboration) {
      collaboration.currentPhase = 'stopped';
      collaboration.endTime = new Date().toISOString();
      
      // 重置智能体状态
      collaboration.agents.forEach(agentData => {
        const agent = this.agents.get(agentData.id);
        if (agent) {
          agent.updateStatus('idle', 0);
        }
      });
    }
    return collaboration;
  }
}

export default AgentService;