import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Agent } from '@/types';
import { 
  CheckCircle, 
  Clock, 
  Zap, 
  AlertCircle,
  ArrowRight,
  PlayCircle
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface AgentWorkflowCardProps {
  agent: Agent;
  index: number;
  showWorkflow?: boolean;
}

const AgentWorkflowCard: React.FC<AgentWorkflowCardProps> = ({ 
  agent, 
  index,
  showWorkflow = true 
}) => {
  const getStatusIcon = (status: Agent['status']) => {
    switch (status) {
      case 'working':
        return <Zap className="w-4 h-4 text-blue-500 animate-pulse" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: Agent['status']) => {
    switch (status) {
      case 'working': return 'border-blue-500/50 bg-blue-500/10';
      case 'completed': return 'border-green-500/50 bg-green-500/10';
      case 'error': return 'border-red-500/50 bg-red-500/10';
      default: return 'border-gray-500/50 bg-gray-500/10';
    }
  };

  const getWorkflowSteps = (agentId: string) => {
    const workflows = {
      'recorder-agent': [
        { step: '音频接收', status: agent.progress > 0 ? 'completed' : 'pending' },
        { step: '语音识别', status: agent.progress > 25 ? 'completed' : agent.progress > 0 ? 'working' : 'pending' },
        { step: '说话人分离', status: agent.progress > 50 ? 'completed' : agent.progress > 25 ? 'working' : 'pending' },
        { step: '文本输出', status: agent.progress > 75 ? 'completed' : agent.progress > 50 ? 'working' : 'pending' }
      ],
      'analyst-agent': [
        { step: '内容分析', status: agent.progress > 0 ? 'completed' : 'pending' },
        { step: '关键信息提取', status: agent.progress > 30 ? 'completed' : agent.progress > 0 ? 'working' : 'pending' },
        { step: '情感分析', status: agent.progress > 60 ? 'completed' : agent.progress > 30 ? 'working' : 'pending' },
        { step: '数据整合', status: agent.progress > 90 ? 'completed' : agent.progress > 60 ? 'working' : 'pending' }
      ],
      'secretary-agent': [
        { step: '任务识别', status: agent.progress > 0 ? 'completed' : 'pending' },
        { step: '责任人分配', status: agent.progress > 40 ? 'completed' : agent.progress > 0 ? 'working' : 'pending' },
        { step: '时间规划', status: agent.progress > 70 ? 'completed' : agent.progress > 40 ? 'working' : 'pending' },
        { step: '待办生成', status: agent.progress > 95 ? 'completed' : agent.progress > 70 ? 'working' : 'pending' }
      ],
      'editor-agent': [
        { step: '文本优化', status: agent.progress > 0 ? 'completed' : 'pending' },
        { step: '格式统一', status: agent.progress > 35 ? 'completed' : agent.progress > 0 ? 'working' : 'pending' },
        { step: '语言润色', status: agent.progress > 65 ? 'completed' : agent.progress > 35 ? 'working' : 'pending' },
        { step: '最终校对', status: agent.progress > 90 ? 'completed' : agent.progress > 65 ? 'working' : 'pending' }
      ],
      'qa-agent': [
        { step: '逻辑检查', status: agent.progress > 0 ? 'completed' : 'pending' },
        { step: '准确性验证', status: agent.progress > 30 ? 'completed' : agent.progress > 0 ? 'working' : 'pending' },
        { step: '一致性审核', status: agent.progress > 70 ? 'completed' : agent.progress > 30 ? 'working' : 'pending' },
        { step: '质量确认', status: agent.progress > 95 ? 'completed' : agent.progress > 70 ? 'working' : 'pending' }
      ]
    };

    return workflows[agentId as keyof typeof workflows] || [];
  };

  const workflowSteps = getWorkflowSteps(agent.id);

  return (
    <Card className={cn(
      "glass-effect transition-all duration-300 hover:scale-105 border",
      getStatusColor(agent.status)
    )}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getStatusIcon(agent.status)}
            <CardTitle className="text-lg text-white">{agent.name}</CardTitle>
          </div>
          <Badge variant="outline" className="text-xs border-gray-600 text-gray-300">
            {agent.role}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* 进度显示 */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-300">工作进度</span>
            <span className="text-white font-mono">{agent.progress}%</span>
          </div>
          <Progress value={agent.progress} className="h-2" />
        </div>

        {/* 工作流程步骤 */}
        {showWorkflow && workflowSteps.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
              <PlayCircle className="w-4 h-4" />
              工作流程
            </h4>
            <div className="space-y-2">
              {workflowSteps.map((step, stepIndex) => (
                <div key={stepIndex} className="flex items-center gap-3 text-sm">
                  <div className={cn(
                    "w-3 h-3 rounded-full flex-shrink-0",
                    step.status === 'completed' ? 'bg-green-500' :
                    step.status === 'working' ? 'bg-blue-500 animate-pulse' :
                    'bg-gray-600'
                  )} />
                  <span className={cn(
                    "flex-1",
                    step.status === 'completed' ? 'text-green-300' :
                    step.status === 'working' ? 'text-blue-300' :
                    'text-gray-400'
                  )}>
                    {step.step}
                  </span>
                  {step.status === 'working' && (
                    <div className="flex items-center gap-1">
                      <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" />
                      <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                      <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                    </div>
                  )}
                  {stepIndex < workflowSteps.length - 1 && step.status === 'completed' && (
                    <ArrowRight className="w-3 h-3 text-green-400" />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 智能体描述 */}
        <div className="pt-2 border-t border-gray-700">
          <p className="text-xs text-gray-400 leading-relaxed">
            {agent.description}
          </p>
        </div>

        {/* 状态指示 */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">状态</span>
          <div className="flex items-center gap-2">
            <div className={cn(
              "w-2 h-2 rounded-full",
              agent.status === 'working' ? 'bg-blue-500 animate-pulse' :
              agent.status === 'completed' ? 'bg-green-500' :
              agent.status === 'error' ? 'bg-red-500' :
              'bg-gray-500'
            )} />
            <span className="text-gray-300 capitalize">
              {agent.status === 'working' ? '工作中' :
               agent.status === 'completed' ? '已完成' :
               agent.status === 'error' ? '错误' : '待机'}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default AgentWorkflowCard;