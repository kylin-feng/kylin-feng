import { WebSocket, WebSocketServer } from 'ws';
import { v4 as uuidv4 } from 'uuid';

// WebSocket服务
export class WebSocketService {
  constructor() {
    this.wss = null;
    this.clients = new Map(); // 存储客户端连接
    this.sessions = new Map(); // 存储会话信息
  }

  // 初始化WebSocket服务器
  initialize(server) {
    this.wss = new WebSocketServer({ server });
    
    this.wss.on('connection', (ws, req) => {
      const clientId = uuidv4();
      console.log(`新客户端连接: ${clientId}`);
      
      // 存储客户端信息
      this.clients.set(clientId, {
        id: clientId,
        ws,
        sessionId: null,
        connectedAt: new Date()
      });

      // 发送连接确认
      this.sendToClient(clientId, {
        type: 'connection',
        data: {
          clientId,
          message: '连接成功',
          timestamp: new Date().toISOString()
        }
      });

      // 处理消息
      ws.on('message', (message) => {
        try {
          const data = JSON.parse(message.toString());
          this.handleMessage(clientId, data);
        } catch (error) {
          console.error('WebSocket消息解析失败:', error);
          this.sendError(clientId, '消息格式错误');
        }
      });

      // 处理连接关闭
      ws.on('close', () => {
        console.log(`客户端断开连接: ${clientId}`);
        this.handleDisconnect(clientId);
      });

      // 处理错误
      ws.on('error', (error) => {
        console.error(`WebSocket错误 ${clientId}:`, error);
      });
    });

    console.log('WebSocket服务器初始化完成');
  }

  // 处理客户端消息
  handleMessage(clientId, data) {
    const client = this.clients.get(clientId);
    if (!client) return;

    console.log(`收到客户端 ${clientId} 消息:`, data.type);

    switch (data.type) {
      case 'join_session':
        this.joinSession(clientId, data.sessionId);
        break;
      
      case 'leave_session':
        this.leaveSession(clientId);
        break;
      
      case 'agent_update':
        this.broadcastAgentUpdate(clientId, data.data);
        break;
      
      case 'collaboration_update':
        this.broadcastCollaborationUpdate(clientId, data.data);
        break;
      
      case 'ping':
        this.sendToClient(clientId, { type: 'pong', timestamp: new Date().toISOString() });
        break;
      
      default:
        console.log(`未知消息类型: ${data.type}`);
    }
  }

  // 加入会话
  joinSession(clientId, sessionId) {
    const client = this.clients.get(clientId);
    if (!client) return;

    // 离开之前的会话
    if (client.sessionId) {
      this.leaveSession(clientId);
    }

    // 加入新会话
    client.sessionId = sessionId;
    
    if (!this.sessions.has(sessionId)) {
      this.sessions.set(sessionId, new Set());
    }
    
    this.sessions.get(sessionId).add(clientId);

    console.log(`客户端 ${clientId} 加入会话 ${sessionId}`);
    
    this.sendToClient(clientId, {
      type: 'session_joined',
      data: {
        sessionId,
        message: '已加入会话',
        timestamp: new Date().toISOString()
      }
    });

    // 通知会话中的其他客户端
    this.broadcastToSession(sessionId, {
      type: 'client_joined',
      data: {
        clientId,
        timestamp: new Date().toISOString()
      }
    }, clientId);
  }

  // 离开会话
  leaveSession(clientId) {
    const client = this.clients.get(clientId);
    if (!client || !client.sessionId) return;

    const sessionId = client.sessionId;
    const sessionClients = this.sessions.get(sessionId);
    
    if (sessionClients) {
      sessionClients.delete(clientId);
      
      // 如果会话中没有客户端了，删除会话
      if (sessionClients.size === 0) {
        this.sessions.delete(sessionId);
      }
    }

    client.sessionId = null;

    console.log(`客户端 ${clientId} 离开会话 ${sessionId}`);

    // 通知会话中的其他客户端
    this.broadcastToSession(sessionId, {
      type: 'client_left',
      data: {
        clientId,
        timestamp: new Date().toISOString()
      }
    });
  }

  // 处理断开连接
  handleDisconnect(clientId) {
    const client = this.clients.get(clientId);
    if (client && client.sessionId) {
      this.leaveSession(clientId);
    }
    this.clients.delete(clientId);
  }

  // 广播智能体更新
  broadcastAgentUpdate(fromClientId, agentData) {
    const client = this.clients.get(fromClientId);
    if (!client || !client.sessionId) return;

    this.broadcastToSession(client.sessionId, {
      type: 'agent_updated',
      data: agentData,
      timestamp: new Date().toISOString()
    });
  }

  // 广播协作更新
  broadcastCollaborationUpdate(fromClientId, collaborationData) {
    const client = this.clients.get(fromClientId);
    if (!client || !client.sessionId) return;

    this.broadcastToSession(client.sessionId, {
      type: 'collaboration_updated',
      data: collaborationData,
      timestamp: new Date().toISOString()
    });
  }

  // 向特定客户端发送消息
  sendToClient(clientId, message) {
    const client = this.clients.get(clientId);
    if (client && client.ws.readyState === WebSocket.OPEN) {
      try {
        client.ws.send(JSON.stringify(message));
      } catch (error) {
        console.error(`发送消息到客户端 ${clientId} 失败:`, error);
      }
    }
  }

  // 向会话中的所有客户端广播消息
  broadcastToSession(sessionId, message, excludeClientId = null) {
    const sessionClients = this.sessions.get(sessionId);
    if (!sessionClients) return;

    sessionClients.forEach(clientId => {
      if (clientId !== excludeClientId) {
        this.sendToClient(clientId, message);
      }
    });
  }

  // 向所有客户端广播消息
  broadcast(message, excludeClientId = null) {
    this.clients.forEach((client, clientId) => {
      if (clientId !== excludeClientId) {
        this.sendToClient(clientId, message);
      }
    });
  }

  // 发送错误消息
  sendError(clientId, errorMessage) {
    this.sendToClient(clientId, {
      type: 'error',
      data: {
        message: errorMessage,
        timestamp: new Date().toISOString()
      }
    });
  }

  // 获取服务器状态
  getStatus() {
    return {
      totalClients: this.clients.size,
      totalSessions: this.sessions.size,
      sessions: Array.from(this.sessions.entries()).map(([sessionId, clients]) => ({
        sessionId,
        clientCount: clients.size
      }))
    };
  }

  // 通知智能体状态更新
  notifyAgentStatusUpdate(sessionId, agentId, status, progress) {
    this.broadcastToSession(sessionId, {
      type: 'agent_status_update',
      data: {
        agentId,
        status,
        progress,
        timestamp: new Date().toISOString()
      }
    });
  }

  // 通知协作阶段更新
  notifyCollaborationPhaseUpdate(sessionId, phase, progress) {
    this.broadcastToSession(sessionId, {
      type: 'collaboration_phase_update',
      data: {
        phase,
        progress,
        timestamp: new Date().toISOString()
      }
    });
  }

  // 通知会议纪要生成完成
  notifyMinutesGenerated(sessionId, minutesData) {
    this.broadcastToSession(sessionId, {
      type: 'minutes_generated',
      data: minutesData,
      timestamp: new Date().toISOString()
    });
  }

  // 清理过期连接
  cleanupConnections() {
    const now = new Date();
    const maxAge = 30 * 60 * 1000; // 30分钟

    this.clients.forEach((client, clientId) => {
      const age = now - client.connectedAt;
      if (age > maxAge && client.ws.readyState !== WebSocket.OPEN) {
        console.log(`清理过期连接: ${clientId}`);
        this.handleDisconnect(clientId);
      }
    });
  }
}

export default WebSocketService;