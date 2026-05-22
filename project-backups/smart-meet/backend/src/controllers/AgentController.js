import AgentService from '../services/AgentService.js';

const agentService = new AgentService();

// 智能体控制器
export class AgentController {
  // 获取所有智能体
  static async getAgents(req, res) {
    try {
      const agents = agentService.getAllAgents();
      
      res.json({
        success: true,
        data: agents,
        message: '获取智能体列表成功',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('获取智能体失败:', error);
      res.status(500).json({
        success: false,
        data: null,
        message: '获取智能体列表失败',
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }

  // 获取单个智能体
  static async getAgent(req, res) {
    try {
      const { agentId } = req.params;
      const agent = agentService.getAgent(agentId);
      
      if (!agent) {
        return res.status(404).json({
          success: false,
          data: null,
          message: '智能体不存在',
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        data: agent,
        message: '获取智能体信息成功',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('获取智能体失败:', error);
      res.status(500).json({
        success: false,
        data: null,
        message: '获取智能体信息失败',
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }

  // 启动智能体协作
  static async startCollaboration(req, res) {
    try {
      const meetingData = req.body;
      
      // 验证请求数据
      if (!meetingData.title) {
        return res.status(400).json({
          success: false,
          data: null,
          message: '会议标题不能为空',
          timestamp: new Date().toISOString()
        });
      }

      const collaboration = await agentService.startCollaboration(meetingData);
      
      res.json({
        success: true,
        data: collaboration,
        message: '智能体协作启动成功',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('启动协作失败:', error);
      res.status(500).json({
        success: false,
        data: null,
        message: '启动智能体协作失败',
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }

  // 获取协作状态
  static async getCollaborationStatus(req, res) {
    try {
      const { sessionId } = req.params;
      const collaboration = agentService.getCollaborationStatus(sessionId);
      
      if (!collaboration) {
        return res.status(404).json({
          success: false,
          data: null,
          message: '协作会话不存在',
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        data: collaboration,
        message: '获取协作状态成功',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('获取协作状态失败:', error);
      res.status(500).json({
        success: false,
        data: null,
        message: '获取协作状态失败',
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }

  // 停止协作
  static async stopCollaboration(req, res) {
    try {
      const { sessionId } = req.params;
      const collaboration = agentService.stopCollaboration(sessionId);
      
      if (!collaboration) {
        return res.status(404).json({
          success: false,
          data: null,
          message: '协作会话不存在',
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        data: collaboration,
        message: '停止协作成功',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('停止协作失败:', error);
      res.status(500).json({
        success: false,
        data: null,
        message: '停止协作失败',
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }

  // 更新智能体状态
  static async updateAgentStatus(req, res) {
    try {
      const { agentId } = req.params;
      const { status, progress } = req.body;
      
      const agent = agentService.updateAgentStatus(agentId, status, progress);
      
      if (!agent) {
        return res.status(404).json({
          success: false,
          data: null,
          message: '智能体不存在',
          timestamp: new Date().toISOString()
        });
      }

      res.json({
        success: true,
        data: agent,
        message: '更新智能体状态成功',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('更新智能体状态失败:', error);
      res.status(500).json({
        success: false,
        data: null,
        message: '更新智能体状态失败',
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }
}

export default AgentController;