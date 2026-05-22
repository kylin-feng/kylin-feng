import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import AgentCard from '@/components/features/AgentCard';
import { Agent, Meeting, AgentCollaboration } from '@/types';
import { agentApi, meetingApi } from '@/services/api';
import { 
  Play, 
  Square, 
  Users, 
  Brain, 
  FileText, 
  Calendar,
  Mic,
  MicOff
} from 'lucide-react';

const Dashboard: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [currentMeeting, setCurrentMeeting] = useState<Meeting | null>(null);
  const [collaboration, setCollaboration] = useState<AgentCollaboration | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);

  // 模拟智能体数据
  const mockAgents: Agent[] = [
    {
      id: '1',
      name: '记录员',
      role: 'Recorder Agent',
      status: 'idle',
      progress: 0,
      description: '专注实时语音转文字和发言人识别',
      capabilities: ['语音识别', '说话人分离', '实时转录']
    },
    {
      id: '2', 
      name: '分析师',
      role: 'Analyst Agent',
      status: 'idle',
      progress: 0,
      description: '提取关键信息和决策要点',
      capabilities: ['内容分析', '关键信息提取', '情感分析']
    },
    {
      id: '3',
      name: '秘书',
      role: 'Secretary Agent', 
      status: 'idle',
      progress: 0,
      description: '整理待办事项和责任分配',
      capabilities: ['任务管理', '时间规划', '责任分配']
    },
    {
      id: '4',
      name: '编辑',
      role: 'Editor Agent',
      status: 'idle', 
      progress: 0,
      description: '优化语言表达和格式规范',
      capabilities: ['文本优化', '格式标准化', '语言润色']
    },
    {
      id: '5',
      name: '质检',
      role: 'QA Agent',
      status: 'idle',
      progress: 0,
      description: '验证信息准确性和逻辑检查',
      capabilities: ['逻辑验证', '准确性检查', '质量控制']
    }
  ];

  useEffect(() => {
    setAgents(mockAgents);
    fetchMeetings();
  }, []);

  const fetchMeetings = async () => {
    try {
      // const response = await meetingApi.getMeetings();
      // 暂时使用模拟数据
    } catch (error) {
      console.error('获取会议列表失败:', error);
    }
  };

  const startMeeting = async () => {
    setLoading(true);
    try {
      // 创建新会议
      const meetingData = {
        title: `会议 - ${new Date().toLocaleString()}`,
        date: new Date().toISOString(),
        participants: ['用户'],
        status: 'live' as const
      };

      // const meeting = await meetingApi.createMeeting(meetingData);
      const mockMeeting: Meeting = {
        id: Date.now().toString(),
        ...meetingData
      };
      
      setCurrentMeeting(mockMeeting);

      // 启动智能体协作
      // const collaboration = await agentApi.startCollaboration(mockMeeting);
      const mockCollaboration: AgentCollaboration = {
        sessionId: Date.now().toString(),
        agents: mockAgents.map(agent => ({ ...agent, status: 'working' as const })),
        currentPhase: 'recording',
        progress: 0,
        startTime: new Date().toISOString(),
        estimatedEndTime: new Date(Date.now() + 60 * 60 * 1000).toISOString()
      };

      setCollaboration(mockCollaboration);
      setAgents(mockCollaboration.agents);
      setIsRecording(true);

      // 模拟智能体进度更新
      simulateAgentProgress();

    } catch (error) {
      console.error('启动会议失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const stopMeeting = async () => {
    setLoading(true);
    try {
      if (currentMeeting) {
        // await meetingApi.stopRecording(currentMeeting.id);
        setIsRecording(false);
        setAgents(agents.map(agent => ({ 
          ...agent, 
          status: agent.progress === 100 ? 'completed' : 'idle' as const 
        })));
        setCurrentMeeting(null);
        setCollaboration(null);
      }
    } catch (error) {
      console.error('停止会议失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const simulateAgentProgress = () => {
    const interval = setInterval(() => {
      setAgents(prevAgents => {
        const updatedAgents = prevAgents.map(agent => {
          if (agent.status === 'working' && agent.progress < 100) {
            const newProgress = Math.min(agent.progress + Math.random() * 10, 100);
            return {
              ...agent,
              progress: Math.round(newProgress),
              status: newProgress === 100 ? 'completed' as const : 'working' as const
            };
          }
          return agent;
        });

        // 如果所有智能体都完成了，清除定时器
        if (updatedAgents.every(agent => agent.status === 'completed')) {
          clearInterval(interval);
        }

        return updatedAgents;
      });
    }, 1000);

    return () => clearInterval(interval);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <div className="container mx-auto p-6 space-y-6">
        {/* 头部 */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-white">
            SmartMeet AI
          </h1>
          <p className="text-lg text-gray-300">
            多智能体协作的智能会议管理助手
          </p>
        </div>

        {/* 控制面板 */}
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Brain className="w-6 h-6" />
              <span>智能体协作中心</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  {isRecording ? (
                    <Mic className="w-5 h-5 text-red-400" />
                  ) : (
                    <MicOff className="w-5 h-5 text-gray-400" />
                  )}
                  <span className="text-white">
                    {isRecording ? '会议进行中' : '待机状态'}
                  </span>
                </div>
                
                {collaboration && (
                  <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/50">
                    {collaboration.currentPhase === 'recording' ? '录制中' : '处理中'}
                  </Badge>
                )}
              </div>

              <div className="flex space-x-2">
                {!isRecording ? (
                  <Button 
                    onClick={startMeeting} 
                    disabled={loading}
                    className="bg-green-600 hover:bg-green-700"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    开始会议
                  </Button>
                ) : (
                  <Button 
                    onClick={stopMeeting} 
                    disabled={loading}
                    variant="destructive"
                  >
                    <Square className="w-4 h-4 mr-2" />
                    结束会议
                  </Button>
                )}
              </div>
            </div>

            {currentMeeting && (
              <div className="text-sm text-gray-300 space-y-1">
                <p>会议: {currentMeeting.title}</p>
                <p>开始时间: {new Date(currentMeeting.date).toLocaleString()}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 智能体状态网格 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((agent) => (
            <AgentCard key={agent.id} agent={agent} />
          ))}
        </div>

        {/* 协作进度总览 */}
        {collaboration && (
          <Card className="glass-effect">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2 text-white">
                <Users className="w-6 h-6" />
                <span>协作进度</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between text-sm text-gray-300">
                  <span>整体进度</span>
                  <span>
                    {Math.round(agents.reduce((acc, agent) => acc + agent.progress, 0) / agents.length)}%
                  </span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-3">
                  <div 
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${Math.round(agents.reduce((acc, agent) => acc + agent.progress, 0) / agents.length)}%` 
                    }}
                  />
                </div>
                <div className="grid grid-cols-3 gap-4 text-center text-sm">
                  <div>
                    <p className="text-gray-400">活跃智能体</p>
                    <p className="text-white font-semibold">
                      {agents.filter(a => a.status === 'working').length}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400">已完成</p>
                    <p className="text-white font-semibold">
                      {agents.filter(a => a.status === 'completed').length}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400">预计完成</p>
                    <p className="text-white font-semibold">
                      {collaboration.estimatedEndTime ? 
                        new Date(collaboration.estimatedEndTime).toLocaleTimeString() : 
                        '--'
                      }
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 快速操作 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="glass-effect cursor-pointer hover:bg-white/10 transition-all">
            <CardContent className="flex items-center space-x-4 p-6">
              <Calendar className="w-8 h-8 text-blue-400" />
              <div>
                <h3 className="text-white font-semibold">历史会议</h3>
                <p className="text-gray-400 text-sm">查看过往记录</p>
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-effect cursor-pointer hover:bg-white/10 transition-all">
            <CardContent className="flex items-center space-x-4 p-6">
              <FileText className="w-8 h-8 text-green-400" />
              <div>
                <h3 className="text-white font-semibold">会议纪要</h3>
                <p className="text-gray-400 text-sm">生成多版本纪要</p>
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-effect cursor-pointer hover:bg-white/10 transition-all">
            <CardContent className="flex items-center space-x-4 p-6">
              <Brain className="w-8 h-8 text-purple-400" />
              <div>
                <h3 className="text-white font-semibold">智能分析</h3>
                <p className="text-gray-400 text-sm">深度数据洞察</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;