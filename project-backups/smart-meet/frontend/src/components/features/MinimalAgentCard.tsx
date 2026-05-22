import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Agent } from '@/types';
import { 
  Brain, 
  Mic, 
  FileText, 
  Users, 
  CheckCircle,
  Activity
} from 'lucide-react';

interface MinimalAgentCardProps {
  agent: Agent;
  compact?: boolean;
}

const MinimalAgentCard: React.FC<MinimalAgentCardProps> = ({ 
  agent, 
  compact = false 
}) => {
  const getAgentIcon = (type: string) => {
    const iconProps = { className: "w-4 h-4" };
    switch (type) {
      case 'recorder': return <Mic {...iconProps} />;
      case 'analyzer': return <Brain {...iconProps} />;
      case 'secretary': return <FileText {...iconProps} />;
      case 'editor': return <Users {...iconProps} />;
      case 'qa': return <CheckCircle {...iconProps} />;
      default: return <Activity {...iconProps} />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'working': return 'bg-green-500';
      case 'analyzing': return 'bg-blue-500';
      case 'completed': return 'bg-slate-400';
      default: return 'bg-slate-300';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'idle': return '待机';
      case 'working': return '工作中';
      case 'analyzing': return '分析中';
      case 'completed': return '已完成';
      default: return '未知';
    }
  };

  if (compact) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-lg border border-slate-100 bg-white hover:border-slate-200 transition-colors">
        <div className="w-8 h-8 bg-slate-100 rounded-full flex items-center justify-center relative">
          {getAgentIcon(agent.type)}
          <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-white ${getStatusColor(agent.status)}`} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-slate-900 truncate">
            {agent.name}
          </div>
          <div className="text-xs text-slate-500 truncate">
            {getStatusLabel(agent.status)}
          </div>
        </div>
        {agent.progress !== undefined && agent.status === 'working' && (
          <div className="w-16">
            <Progress 
              value={agent.progress} 
              className="h-1" 
            />
          </div>
        )}
      </div>
    );
  }

  return (
    <Card className="border-0 shadow-sm hover:shadow-md transition-shadow">
      <CardContent className="p-6">
        <div className="flex items-start gap-4">
          {/* Icon */}
          <div className="w-10 h-10 bg-slate-100 rounded-lg flex items-center justify-center relative">
            {getAgentIcon(agent.type)}
            {/* Status Indicator */}
            <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-white ${getStatusColor(agent.status)}`} />
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-medium text-slate-900 truncate">
                {agent.name}
              </h3>
              <Badge 
                variant="secondary" 
                className="text-xs rounded-full bg-slate-100 text-slate-600"
              >
                {getStatusLabel(agent.status)}
              </Badge>
            </div>

            <p className="text-sm text-slate-600 leading-relaxed mb-3">
              {agent.description}
            </p>

            {/* Progress */}
            {agent.progress !== undefined && agent.status === 'working' && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-slate-500">进度</span>
                  <span className="text-xs text-slate-700 font-medium">
                    {agent.progress}%
                  </span>
                </div>
                <Progress 
                  value={agent.progress} 
                  className="h-1.5"
                />
              </div>
            )}

            {/* Current Task */}
            {agent.currentTask && (
              <div className="mt-3 p-2 bg-slate-50 rounded-lg">
                <div className="text-xs text-slate-500 mb-1">当前任务</div>
                <div className="text-xs text-slate-700">
                  {agent.currentTask}
                </div>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default MinimalAgentCard;