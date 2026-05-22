// WebSocket 客户端服务
export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectInterval: number = 5000;
  private maxReconnectAttempts: number = 5;
  private reconnectAttempts: number = 0;
  private listeners: Map<string, Set<Function>> = new Map();
  private sessionId: string | null = null;
  private isConnecting: boolean = false;

  constructor(url: string) {
    this.url = url;
  }

  // 连接WebSocket
  connect(): Promise<void> {
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return Promise.resolve();
    }

    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket连接已建立');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.emit('connected');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('WebSocket消息解析失败:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket连接已关闭', event.code, event.reason);
          this.isConnecting = false;
          this.emit('disconnected', { code: event.code, reason: event.reason });
          
          // 自动重连
          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket错误:', error);
          this.isConnecting = false;
          this.emit('error', error);
          reject(error);
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  // 断开连接
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.sessionId = null;
  }

  // 发送消息
  send(message: any): boolean {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('发送WebSocket消息失败:', error);
        return false;
      }
    }
    console.warn('WebSocket未连接，无法发送消息');
    return false;
  }

  // 加入会话
  joinSession(sessionId: string) {
    this.sessionId = sessionId;
    this.send({
      type: 'join_session',
      sessionId: sessionId
    });
  }

  // 离开会话
  leaveSession() {
    if (this.sessionId) {
      this.send({
        type: 'leave_session'
      });
      this.sessionId = null;
    }
  }

  // 更新智能体状态
  updateAgentStatus(agentData: any) {
    this.send({
      type: 'agent_update',
      data: agentData
    });
  }

  // 更新协作状态
  updateCollaboration(collaborationData: any) {
    this.send({
      type: 'collaboration_update',
      data: collaborationData
    });
  }

  // 心跳检测
  ping() {
    this.send({ type: 'ping' });
  }

  // 处理收到的消息
  private handleMessage(message: any) {
    console.log('收到WebSocket消息:', message.type);
    
    switch (message.type) {
      case 'connection':
        this.emit('connection_confirmed', message.data);
        break;
      
      case 'session_joined':
        this.emit('session_joined', message.data);
        break;
      
      case 'agent_updated':
        this.emit('agent_updated', message.data);
        break;
      
      case 'collaboration_updated':
        this.emit('collaboration_updated', message.data);
        break;
      
      case 'agent_status_update':
        this.emit('agent_status_update', message.data);
        break;
      
      case 'collaboration_phase_update':
        this.emit('collaboration_phase_update', message.data);
        break;
      
      case 'minutes_generated':
        this.emit('minutes_generated', message.data);
        break;
      
      case 'client_joined':
        this.emit('client_joined', message.data);
        break;
      
      case 'client_left':
        this.emit('client_left', message.data);
        break;
      
      case 'pong':
        this.emit('pong', message);
        break;
      
      case 'error':
        this.emit('error', message.data);
        break;
      
      default:
        console.log('未知消息类型:', message.type);
    }
  }

  // 计划重连
  private scheduleReconnect() {
    this.reconnectAttempts++;
    console.log(`WebSocket重连尝试 ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
    
    setTimeout(() => {
      this.connect().catch(error => {
        console.error('WebSocket重连失败:', error);
      });
    }, this.reconnectInterval);
  }

  // 事件监听
  on(event: string, listener: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);
  }

  // 移除事件监听
  off(event: string, listener: Function) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(listener);
    }
  }

  // 触发事件
  private emit(event: string, data?: any) {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error('事件监听器执行失败:', error);
        }
      });
    }
  }

  // 获取连接状态
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  // 获取当前会话ID
  get currentSessionId(): string | null {
    return this.sessionId;
  }
}

// 创建全局WebSocket客户端实例
const WS_URL = import.meta.env.VITE_WS_URL || 
  `ws://localhost:${window.location.hostname === 'localhost' ? '5001' : window.location.port}`;

export const wsClient = new WebSocketClient(WS_URL);

// 自动连接
wsClient.connect().catch(error => {
  console.error('初始WebSocket连接失败:', error);
});

export default wsClient;