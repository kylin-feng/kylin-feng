import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Agent } from '@/types';
import { cn } from '@/lib/utils';

interface AgentCardProps {
  agent: Agent;
  className?: string;
}

const AgentCard: React.FC<AgentCardProps> = ({ agent, className }) => {
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
        return '待机';
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

  return (
    <Card className={cn('agent-card', className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold text-white">
            {agent.name}
          </CardTitle>
          <div className="flex items-center space-x-2">
            <div className={cn('w-2 h-2 rounded-full', getStatusColor(agent.status))} />
            <Badge variant="outline" className="text-xs">
              {getStatusText(agent.status)}
            </Badge>
          </div>
        </div>
        <p className="text-sm text-gray-300">{agent.role}</p>
      </CardHeader>
      
      <CardContent className="space-y-4">
        <p className="text-sm text-gray-400">{agent.description}</p>
        
        {agent.status === 'working' && (
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-gray-400">
              <span>进度</span>
              <span>{agent.progress}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${agent.progress}%` }}
              />
            </div>
          </div>
        )}
        
        <div className="space-y-2">
          <p className="text-xs text-gray-400 font-medium">核心能力</p>
          <div className="flex flex-wrap gap-1">
            {agent.capabilities.map((capability, index) => (
              <Badge 
                key={index} 
                variant="secondary" 
                className="text-xs bg-white/10 text-white border-white/20"
              >
                {capability}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default AgentCard;