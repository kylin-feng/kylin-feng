import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Agent } from '@/types';
import { cn } from '@/lib/utils';
import { Eye, Zap, CheckCircle, AlertCircle, Clock } from 'lucide-react';

interface EnhancedAgentCardProps {
  agent: Agent;
  className?: string;
  index?: number;
}

const EnhancedAgentCard: React.FC<EnhancedAgentCardProps> = ({ agent, className, index = 0 }) => {
  const [showDetails, setShowDetails] = useState(false);

  const getStatusIcon = (status: Agent['status']) => {
    switch (status) {
      case 'idle':
        return <Clock className="w-4 h-4 text-gray-400" />;
      case 'working':
        return <Zap className="w-4 h-4 text-blue-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: Agent['status']) => {
    switch (status) {
      case 'idle':
        return 'bg-gray-500';
      case 'working':
        return 'bg-blue-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusText = (status: Agent['status']) => {
    switch (status) {
      case 'idle':
        return '待机中';
      case 'working':
        return '工作中';
      case 'completed':
        return '已完成';
      case 'error':
        return '错误';
      default:
        return '未知';
    }
  };

  const animationDelay = `${index * 0.1}s`;

  return (
    <>
      <Card 
        className={cn(
          'agent-card card-hover cursor-pointer',
          agent.status === 'working' && 'working',
          agent.status === 'completed' && 'completed',
          className
        )}
        style={{ animationDelay }}
        onClick={() => setShowDetails(true)}
      >
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
              {getStatusIcon(agent.status)}
              {agent.name}
            </CardTitle>
            <div className="flex items-center space-x-2">
              <div 
                className={cn(
                  'w-3 h-3 rounded-full status-indicator',
                  getStatusColor(agent.status)
                )} 
              />
              <Badge variant="outline" className="text-xs bg-white/10 border-white/20 text-white">
                {getStatusText(agent.status)}
              </Badge>
            </div>
          </div>
          <p className="text-sm text-gray-300 animate-fade-in-up" style={{ animationDelay }}>
            {agent.role}
          </p>
        </CardHeader>
        
        <CardContent className="space-y-4">
          <p className="text-sm text-gray-400 line-clamp-2">{agent.description}</p>
          
          {agent.status === 'working' && (
            <div className="space-y-3">
              <div className="flex justify-between text-xs text-gray-400">
                <span>工作进度</span>
                <span className="font-mono">{agent.progress}%</span>
              </div>
              <Progress 
                value={agent.progress} 
                className="h-2"
              />
              <div className="flex items-center gap-2 text-xs text-blue-300">
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span>正在处理中...</span>
              </div>
            </div>
          )}

          {agent.status === 'completed' && (
            <div className="flex items-center gap-2 text-sm text-green-400 animate-fade-in-up">
              <CheckCircle className="w-4 h-4" />
              <span>任务完成</span>
            </div>
          )}
          
          <div className="space-y-2">
            <p className="text-xs text-gray-400 font-medium">核心能力</p>
            <div className="flex flex-wrap gap-1">
              {agent.capabilities.slice(0, 3).map((capability, capIndex) => (
                <Badge 
                  key={capIndex} 
                  variant="secondary" 
                  className="text-xs bg-white/5 text-white border-white/10 hover:bg-white/10 transition-colors"
                >
                  {capability}
                </Badge>
              ))}
              {agent.capabilities.length > 3 && (
                <Badge 
                  variant="secondary" 
                  className="text-xs bg-white/5 text-white border-white/10"
                >
                  +{agent.capabilities.length - 3}
                </Badge>
              )}
            </div>
          </div>

          <div className="flex justify-between items-center pt-2">
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <Eye className="w-3 h-3" />
              <span>点击查看详情</span>
            </div>
            {agent.status === 'working' && (
              <div className="flex items-center gap-1 text-xs text-blue-400">
                <Zap className="w-3 h-3 animate-pulse" />
                <span>活跃中</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 详情弹窗 */}
      <Dialog open={showDetails} onOpenChange={setShowDetails}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {getStatusIcon(agent.status)}
              {agent.name} - 详细信息
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            {/* 基本信息 */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-gray-600">角色定位</h4>
                <p className="text-sm">{agent.role}</p>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-gray-600">当前状态</h4>
                <div className="flex items-center gap-2">
                  {getStatusIcon(agent.status)}
                  <span className="text-sm">{getStatusText(agent.status)}</span>
                </div>
              </div>
            </div>

            {/* 工作进度 */}
            {agent.status === 'working' && (
              <div className="space-y-2">
                <h4 className="font-medium text-sm text-gray-600">工作进度</h4>
                <Progress value={agent.progress} className="h-3" />
                <p className="text-xs text-gray-500">{agent.progress}% 已完成</p>
              </div>
            )}

            {/* 功能描述 */}
            <div className="space-y-2">
              <h4 className="font-medium text-sm text-gray-600">功能描述</h4>
              <p className="text-sm text-gray-700">{agent.description}</p>
            </div>

            {/* 核心能力 */}
            <div className="space-y-2">
              <h4 className="font-medium text-sm text-gray-600">核心能力</h4>
              <div className="flex flex-wrap gap-2">
                {agent.capabilities.map((capability, index) => (
                  <Badge key={index} variant="outline" className="text-xs">
                    {capability}
                  </Badge>
                ))}
              </div>
            </div>

            {/* 技术规格 */}
            <div className="space-y-2">
              <h4 className="font-medium text-sm text-gray-600">技术规格</h4>
              <div className="bg-gray-50 rounded-lg p-3 text-xs space-y-1">
                <div className="flex justify-between">
                  <span>智能体ID:</span>
                  <span className="font-mono">{agent.id}</span>
                </div>
                <div className="flex justify-between">
                  <span>创建时间:</span>
                  <span>{new Date().toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>大模型:</span>
                  <span>{agent.id.includes('qa') ? 'DeepSeek' : '通义千问'}</span>
                </div>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default EnhancedAgentCard;