import React, { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useWebSocket } from '@/hooks/useWebSocket';
import { Bell, CheckCircle, AlertCircle, Info, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Notification {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  title: string;
  message: string;
  timestamp: Date;
  autoClose?: boolean;
}

const RealTimeNotifications: React.FC = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const { subscribe, isConnected } = useWebSocket();

  useEffect(() => {
    // 监听各种WebSocket事件
    subscribe('connected', () => {
      addNotification({
        type: 'success',
        title: '连接成功',
        message: '实时通信已建立',
        autoClose: true
      });
    });

    subscribe('disconnected', () => {
      addNotification({
        type: 'error',
        title: '连接断开',
        message: '正在尝试重新连接...',
        autoClose: false
      });
    });

    subscribe('session_joined', (data: any) => {
      addNotification({
        type: 'info',
        title: '加入会话',
        message: `已加入会话 ${data.sessionId.slice(0, 8)}...`,
        autoClose: true
      });
    });

    subscribe('agent_status_update', (data: any) => {
      const statusText = {
        'working': '开始工作',
        'completed': '任务完成',
        'error': '发生错误'
      }[data.status] || '状态更新';

      addNotification({
        type: data.status === 'completed' ? 'success' : 
              data.status === 'error' ? 'error' : 'info',
        title: '智能体更新',
        message: `${data.agentId} ${statusText}`,
        autoClose: true
      });
    });

    subscribe('collaboration_phase_update', (data: any) => {
      const phaseText = {
        'preparation': '准备阶段',
        'recording': '录制阶段',
        'processing': '处理阶段',
        'review': '审核阶段',
        'completed': '协作完成'
      }[data.phase] || data.phase;

      addNotification({
        type: data.phase === 'completed' ? 'success' : 'info',
        title: '协作进度',
        message: `进入${phaseText} (${data.progress}%)`,
        autoClose: true
      });
    });

    subscribe('minutes_generated', () => {
      addNotification({
        type: 'success',
        title: '纪要生成完成',
        message: '多版本会议纪要已准备就绪',
        autoClose: false
      });
    });

    subscribe('error', (data: any) => {
      addNotification({
        type: 'error',
        title: '系统错误',
        message: data.message || '发生未知错误',
        autoClose: false
      });
    });

  }, [subscribe]);

  const addNotification = (notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const newNotification: Notification = {
      id: Date.now().toString(),
      timestamp: new Date(),
      ...notification
    };

    setNotifications(prev => [newNotification, ...prev.slice(0, 4)]); // 最多保留5个通知

    // 自动关闭
    if (notification.autoClose) {
      setTimeout(() => {
        removeNotification(newNotification.id);
      }, 5000);
    }
  };

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const getNotificationIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      case 'warning':
        return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      default:
        return <Info className="w-4 h-4 text-blue-500" />;
    }
  };

  const getNotificationStyles = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return 'border-green-500/50 bg-green-500/10';
      case 'error':
        return 'border-red-500/50 bg-red-500/10';
      case 'warning':
        return 'border-yellow-500/50 bg-yellow-500/10';
      default:
        return 'border-blue-500/50 bg-blue-500/10';
    }
  };

  if (notifications.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
      {/* 连接状态指示器 */}
      <div className="flex justify-end mb-2">
        <Badge
          variant={isConnected ? 'default' : 'destructive'}
          className={cn(
            'flex items-center gap-1 text-xs',
            isConnected ? 'bg-green-500/20 text-green-300 border-green-500/50' : ''
          )}
        >
          <div className={cn(
            'w-2 h-2 rounded-full',
            isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'
          )} />
          {isConnected ? '实时连接' : '连接断开'}
        </Badge>
      </div>

      {/* 通知列表 */}
      {notifications.map((notification, index) => (
        <Card
          key={notification.id}
          className={cn(
            'glass-effect border transition-all duration-300 animate-slide-in-right',
            getNotificationStyles(notification.type)
          )}
          style={{ animationDelay: `${index * 0.1}s` }}
        >
          <CardContent className="p-3">
            <div className="flex items-start justify-between">
              <div className="flex items-start space-x-2 flex-1">
                {getNotificationIcon(notification.type)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">
                    {notification.title}
                  </p>
                  <p className="text-xs text-gray-300 mt-1">
                    {notification.message}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {notification.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
              <button
                onClick={() => removeNotification(notification.id)}
                className="text-gray-400 hover:text-white transition-colors ml-2"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default RealTimeNotifications;