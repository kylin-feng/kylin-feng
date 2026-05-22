import { useEffect, useCallback, useRef } from 'react';
import { wsClient } from '@/services/websocket';

// WebSocket Hook
export const useWebSocket = () => {
  const listenersRef = useRef<Map<string, Function>>(new Map());

  // 添加事件监听
  const subscribe = useCallback((event: string, callback: Function) => {
    // 移除之前的监听器（如果存在）
    const oldCallback = listenersRef.current.get(event);
    if (oldCallback) {
      wsClient.off(event, oldCallback);
    }

    // 添加新的监听器
    wsClient.on(event, callback);
    listenersRef.current.set(event, callback);
  }, []);

  // 移除事件监听
  const unsubscribe = useCallback((event: string) => {
    const callback = listenersRef.current.get(event);
    if (callback) {
      wsClient.off(event, callback);
      listenersRef.current.delete(event);
    }
  }, []);

  // 发送消息
  const sendMessage = useCallback((message: any) => {
    return wsClient.send(message);
  }, []);

  // 加入会话
  const joinSession = useCallback((sessionId: string) => {
    wsClient.joinSession(sessionId);
  }, []);

  // 离开会话
  const leaveSession = useCallback(() => {
    wsClient.leaveSession();
  }, []);

  // 更新智能体状态
  const updateAgentStatus = useCallback((agentData: any) => {
    wsClient.updateAgentStatus(agentData);
  }, []);

  // 更新协作状态
  const updateCollaboration = useCallback((collaborationData: any) => {
    wsClient.updateCollaboration(collaborationData);
  }, []);

  // 组件卸载时清理监听器
  useEffect(() => {
    return () => {
      listenersRef.current.forEach((callback, event) => {
        wsClient.off(event, callback);
      });
      listenersRef.current.clear();
    };
  }, []);

  return {
    subscribe,
    unsubscribe,
    sendMessage,
    joinSession,
    leaveSession,
    updateAgentStatus,
    updateCollaboration,
    isConnected: wsClient.isConnected,
    currentSessionId: wsClient.currentSessionId
  };
};

// 智能体实时更新Hook
export const useAgentUpdates = (onAgentUpdate: (agentData: any) => void) => {
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    subscribe('agent_status_update', onAgentUpdate);
    subscribe('agent_updated', onAgentUpdate);

    return () => {
      unsubscribe('agent_status_update');
      unsubscribe('agent_updated');
    };
  }, [subscribe, unsubscribe, onAgentUpdate]);
};

// 协作实时更新Hook
export const useCollaborationUpdates = (onCollaborationUpdate: (data: any) => void) => {
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    subscribe('collaboration_phase_update', onCollaborationUpdate);
    subscribe('collaboration_updated', onCollaborationUpdate);

    return () => {
      unsubscribe('collaboration_phase_update');
      unsubscribe('collaboration_updated');
    };
  }, [subscribe, unsubscribe, onCollaborationUpdate]);
};

// 会议纪要生成Hook
export const useMinutesUpdates = (onMinutesGenerated: (data: any) => void) => {
  const { subscribe, unsubscribe } = useWebSocket();

  useEffect(() => {
    subscribe('minutes_generated', onMinutesGenerated);

    return () => {
      unsubscribe('minutes_generated');
    };
  }, [subscribe, unsubscribe, onMinutesGenerated]);
};

export default useWebSocket;