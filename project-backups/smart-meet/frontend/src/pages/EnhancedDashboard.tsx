import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import EnhancedAgentCard from '@/components/features/EnhancedAgentCard';
import AgentWorkflowCard from '@/components/features/AgentWorkflowCard';
import MeetingMinutesModal from '@/components/features/MeetingMinutesModal';
import RealTimeNotifications from '@/components/features/RealTimeNotifications';
import VoiceRecorder from '@/components/features/VoiceRecorder';
import TranscriptionPanel from '@/components/features/TranscriptionPanel';
import Navigation from '@/components/layout/Navigation';
import { useWebSocket, useAgentUpdates, useCollaborationUpdates } from '@/hooks/useWebSocket';
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
  MicOff,
  BarChart3,
  Settings,
  Download,
  Clock,
  Zap,
  Activity
} from 'lucide-react';

interface TranscriptionSegment {
  id: string;
  speaker: string;
  text: string;
  timestamp: Date;
  confidence: number;
  duration: number;
}

const EnhancedDashboard: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [currentMeeting, setCurrentMeeting] = useState<Meeting | null>(null);
  const [collaboration, setCollaboration] = useState<AgentCollaboration | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showMinutes, setShowMinutes] = useState(false);
  const [transcriptionSegments, setTranscriptionSegments] = useState<TranscriptionSegment[]>([]);
  const [stats, setStats] = useState({
    totalMeetings: 12,
    totalMinutes: 1450,
    avgEfficiency: 87,
    savedTime: 240
  });

  // WebSocket连接
  const { joinSession, leaveSession, updateAgentStatus, updateCollaboration } = useWebSocket();

  // 模拟智能体数据
  const mockAgents: Agent[] = [
    {
      id: 'recorder-agent',
      name: '记录员',
      role: 'Recorder Agent',
      status: 'idle',
      progress: 0,
      description: '专治记录恐惧症！全程自动听写，再也不用边听边记到手抽筋，彻底解放双手',
      capabilities: ['多人语音识别', '说话人分离', '实时转录', '语音降噪', '语言模型优化']
    },
    {
      id: 'analyst-agent', 
      name: '分析师',
      role: 'Analyst Agent',
      status: 'idle',
      progress: 0,
      description: '专治思考懒惰症！AI大脑24小时不休息，帮你提炼重点，再懒也能抓住关键信息',
      capabilities: ['内容分析', '关键信息提取', '情感分析', '趋势识别', '数据挖掘']
    },
    {
      id: 'secretary-agent',
      name: '秘书',
      role: 'Secretary Agent', 
      status: 'idle',
      progress: 0,
      description: '专治任务拖延症！自动安排待办事项，强制分配责任人，让拖延症无处可逃',
      capabilities: ['任务管理', '时间规划', '责任分配', '日程协调', '提醒服务']
    },
    {
      id: 'editor-agent',
      name: '编辑',
      role: 'Editor Agent',
      status: 'idle', 
      progress: 0,
      description: '专治整理懒惰症！自动润色文字、统一格式，懒得修改也能输出完美文档',
      capabilities: ['文本优化', '格式标准化', '语言润色', '风格统一', '可读性提升']
    },
    {
      id: 'qa-agent',
      name: '质检',
      role: 'QA Agent',
      status: 'idle',
      progress: 0,
      description: '专治检查懒惰症！AI火眼金睛找错误，比强迫症还仔细，确保质量万无一失',
      capabilities: ['逻辑验证', '准确性检查', '质量控制', '错误检测', '一致性审核']
    }
  ];

  // 设置实时更新监听
  useAgentUpdates((agentData) => {
    setAgents(prev => prev.map(agent => 
      agent.id === agentData.agentId 
        ? { ...agent, status: agentData.status, progress: agentData.progress }
        : agent
    ));
  });

  useCollaborationUpdates((data) => {
    setCollaboration(prev => prev ? {
      ...prev,
      currentPhase: data.phase,
      progress: data.progress
    } : null);
  });

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
      const meetingData = {
        title: `SmartMeet 会议 - ${new Date().toLocaleString()}`,
        date: new Date().toISOString(),
        participants: ['用户'],
        status: 'live' as const
      };

      const mockMeeting: Meeting = {
        id: Date.now().toString(),
        ...meetingData
      };
      
      setCurrentMeeting(mockMeeting);

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

      // 加入WebSocket会话
      joinSession(mockCollaboration.sessionId);

      // 启动智能体进度模拟
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
      if (currentMeeting && collaboration) {
        setIsRecording(false);
        
        // 更新协作状态为完成
        const completedCollaboration = {
          ...collaboration,
          currentPhase: 'completed' as const,
          progress: 100,
          endTime: new Date().toISOString()
        };
        setCollaboration(completedCollaboration);
        
        setAgents(agents.map(agent => ({ 
          ...agent, 
          status: 'completed' as const,
          progress: 100 
        })));

        // 离开WebSocket会话
        leaveSession();

        // 显示会议纪要
        setTimeout(() => {
          setShowMinutes(true);
        }, 1000);
      }
    } catch (error) {
      console.error('停止会议失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 语音录制处理函数
  const handleRecordingStart = () => {
    console.log('开始语音录制');
  };

  const handleRecordingStop = () => {
    console.log('停止语音录制');
  };

  const handleTranscriptionUpdate = (text: string, speaker?: string) => {
    const newSegment: TranscriptionSegment = {
      id: Date.now().toString(),
      speaker: speaker || '未知发言人',
      text: text,
      timestamp: new Date(),
      confidence: Math.random() * 0.3 + 0.7, // 模拟70%-100%置信度
      duration: Math.floor(Math.random() * 10) + 3 // 模拟3-12秒发言时长
    };

    setTranscriptionSegments(prev => [...prev, newSegment]);
  };

  const simulateAgentProgress = () => {
    const interval = setInterval(() => {
      setAgents(prevAgents => {
        const updatedAgents = prevAgents.map(agent => {
          if (agent.status === 'working' && agent.progress < 100) {
            const increment = Math.random() * 8 + 2; // 2-10% 增长
            const newProgress = Math.min(agent.progress + increment, 100);
            return {
              ...agent,
              progress: Math.round(newProgress),
              status: newProgress === 100 ? 'completed' as const : 'working' as const
            };
          }
          return agent;
        });

        // 更新协作进度
        if (collaboration) {
          const overallProgress = updatedAgents.reduce((acc, agent) => acc + agent.progress, 0) / updatedAgents.length;
          setCollaboration(prev => prev ? { ...prev, progress: Math.round(overallProgress) } : null);
        }

        // 如果所有智能体都完成了，清除定时器
        if (updatedAgents.every(agent => agent.status === 'completed')) {
          clearInterval(interval);
        }

        return updatedAgents;
      });
    }, 1500);

    return () => clearInterval(interval);
  };

  const getPhaseText = (phase: string) => {
    switch (phase) {
      case 'preparation': return '准备阶段';
      case 'recording': return '录制中';
      case 'processing': return '处理中';
      case 'review': return '审核中';
      case 'completed': return '已完成';
      default: return '未知状态';
    }
  };

  const overallProgress = agents.length > 0 
    ? Math.round(agents.reduce((acc, agent) => acc + agent.progress, 0) / agents.length)
    : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Navigation />
      <div className="container mx-auto p-6 pt-20 space-y-6">
        {/* 头部区域 */}
        <div className="text-center space-y-4 animate-fade-in-up">
          <div className="flex items-center justify-center gap-3">
            <Brain className="w-12 h-12 text-purple-400 animate-pulse" />
            <h1 className="text-5xl font-bold text-white bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              SmartMeet AI
            </h1>
          </div>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            拖延症治疗中心 - AI帮你干活，你负责躺平
          </p>
        </div>

        {/* 多智能体协作工作流程展示 */}
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-white flex items-center justify-center gap-3 mb-2">
              <Users className="w-8 h-8 text-purple-400" />
              AI打工仔实时工作状态
            </h2>
            <p className="text-gray-300 text-lg">
              5个AI苦力24小时无休，比最勤快的员工还靠谱，专治各种职场拖延症
            </p>
          </div>

          {/* 智能体状态网格 */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {agents.map((agent, index) => (
              <AgentWorkflowCard key={agent.id} agent={agent} index={index} showWorkflow={true} />
            ))}
          </div>

          {/* 协作进度总览 */}
          {collaboration && (
            <Card className="glass-effect">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2 text-white">
                  <Activity className="w-6 h-6" />
                  <span>智能体协作进度总览</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center justify-center mb-2">
                      <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse mr-2"></div>
                      <span className="text-blue-400 font-semibold">正在干活的AI</span>
                    </div>
                    <p className="text-2xl font-bold text-white">
                      {agents.filter(a => a.status === 'working').length}
                    </p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center justify-center mb-2">
                      <div className="w-3 h-3 bg-green-400 rounded-full mr-2"></div>
                      <span className="text-green-400 font-semibold">完工的AI</span>
                    </div>
                    <p className="text-2xl font-bold text-white">
                      {agents.filter(a => a.status === 'completed').length}
                    </p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center justify-center mb-2">
                      <div className="w-3 h-3 bg-purple-400 rounded-full mr-2"></div>
                      <span className="text-purple-400 font-semibold">躺平阶段</span>
                    </div>
                    <p className="text-lg font-bold text-white">
                      {getPhaseText(collaboration.currentPhase)}
                    </p>
                  </div>
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center justify-center mb-2">
                      <div className="w-3 h-3 bg-yellow-400 rounded-full mr-2"></div>
                      <span className="text-yellow-400 font-semibold">偷懒进度</span>
                    </div>
                    <p className="text-2xl font-bold text-white">{overallProgress}%</p>
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm text-gray-300">
                    <span>AI苦力总进度 (你可以继续躺平)</span>
                    <span className="font-mono">{overallProgress}%</span>
                  </div>
                  <Progress value={overallProgress} className="h-3" />
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* 会议控制面板 */}
        <Card className="glass-effect animate-slide-in-right">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Activity className="w-6 h-6" />
              <span>懒人专用控制面板</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 会议状态 */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  {isRecording ? (
                    <Mic className="w-6 h-6 text-red-400 animate-pulse" />
                  ) : (
                    <MicOff className="w-6 h-6 text-gray-400" />
                  )}
                  <div>
                    <span className="text-white text-lg font-medium">
                      {isRecording ? 'AI正在拼命干活' : '准备让AI替你干活'}
                    </span>
                    {collaboration && (
                      <p className="text-sm text-gray-400">
                        {getPhaseText(collaboration.currentPhase)}
                      </p>
                    )}
                  </div>
                </div>
                
                {collaboration && (
                  <div className="flex items-center space-x-2">
                    <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/50">
                      {getPhaseText(collaboration.currentPhase)}
                    </Badge>
                    {collaboration.currentPhase === 'completed' && (
                      <Badge className="bg-green-500/20 text-green-300 border-green-500/50">
                        任务完成
                      </Badge>
                    )}
                  </div>
                )}
              </div>

              <div className="flex space-x-2">
                {!isRecording ? (
                  <Button 
                    onClick={startMeeting} 
                    disabled={loading}
                    className="bg-green-600 hover:bg-green-700 btn-press"
                    size="lg"
                  >
                    <Play className="w-5 h-5 mr-2" />
                    开始躺平模式
                  </Button>
                ) : (
                  <Button 
                    onClick={stopMeeting} 
                    disabled={loading}
                    variant="destructive"
                    className="btn-press"
                    size="lg"
                  >
                    <Square className="w-5 h-5 mr-2" />
                    停止AI苦力
                  </Button>
                )}
                
                {collaboration?.currentPhase === 'completed' && (
                  <Button
                    onClick={() => setShowMinutes(true)}
                    className="bg-purple-600 hover:bg-purple-700 btn-press"
                    size="lg"
                  >
                    <FileText className="w-5 h-5 mr-2" />
                    查看纪要
                  </Button>
                )}
              </div>
            </div>

            {/* 会议信息 */}
            {currentMeeting && (
              <div className="bg-white/5 rounded-lg p-4 space-y-2">
                <h3 className="text-white font-medium">{currentMeeting.title}</h3>
                <div className="flex items-center justify-between text-sm text-gray-300">
                  <span>开始时间: {new Date(currentMeeting.date).toLocaleString()}</span>
                  {collaboration && (
                    <span>预计完成: {new Date(collaboration.estimatedEndTime).toLocaleTimeString()}</span>
                  )}
                </div>
              </div>
            )}

            {/* 整体进度 */}
            {collaboration && (
              <div className="space-y-3">
                <div className="flex justify-between text-sm text-gray-300">
                  <span>整体协作进度</span>
                  <span className="font-mono">{overallProgress}%</span>
                </div>
                <Progress value={overallProgress} className="h-3" />
                <div className="grid grid-cols-3 gap-4 text-center text-sm">
                  <div>
                    <p className="text-gray-400">活跃智能体</p>
                    <p className="text-white font-semibold text-lg">
                      {agents.filter(a => a.status === 'working').length}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400">已完成</p>
                    <p className="text-white font-semibold text-lg">
                      {agents.filter(a => a.status === 'completed').length}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-400">总进度</p>
                    <p className="text-white font-semibold text-lg">{overallProgress}%</p>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* 语音录制和转录面板 */}
        {isRecording && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <VoiceRecorder
              isRecording={isRecording}
              onRecordingStart={handleRecordingStart}
              onRecordingStop={handleRecordingStop}
              onTranscriptionUpdate={handleTranscriptionUpdate}
            />
            <div className="h-96">
              <TranscriptionPanel
                segments={transcriptionSegments}
                isRecording={isRecording}
              />
            </div>
          </div>
        )}


        {/* 数据统计概览 */}
        <div className="space-y-4">
          <h3 className="text-2xl font-bold text-white flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-purple-400" />
            拖延症治疗成果
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="glass-effect text-center">
              <CardContent className="p-4">
                <Calendar className="w-8 h-8 mx-auto mb-2 text-blue-400" />
                <p className="text-2xl font-bold text-white">{stats.totalMeetings}</p>
                <p className="text-sm text-gray-400">总会议数</p>
              </CardContent>
            </Card>
            
            <Card className="glass-effect text-center">
              <CardContent className="p-4">
                <Clock className="w-8 h-8 mx-auto mb-2 text-green-400" />
                <p className="text-2xl font-bold text-white">{stats.totalMinutes}</p>
                <p className="text-sm text-gray-400">总时长(分钟)</p>
              </CardContent>
            </Card>
            
            <Card className="glass-effect text-center">
              <CardContent className="p-4">
                <BarChart3 className="w-8 h-8 mx-auto mb-2 text-purple-400" />
                <p className="text-2xl font-bold text-white">{stats.avgEfficiency}%</p>
                <p className="text-sm text-gray-400">平均效率</p>
              </CardContent>
            </Card>
            
            <Card className="glass-effect text-center">
              <CardContent className="p-4">
                <Zap className="w-8 h-8 mx-auto mb-2 text-yellow-400" />
                <p className="text-2xl font-bold text-white">{stats.savedTime}</p>
                <p className="text-sm text-gray-400">节省时间(分钟)</p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* 快速操作 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="glass-effect cursor-pointer card-hover">
            <CardContent className="flex items-center space-x-4 p-6">
              <Calendar className="w-10 h-10 text-blue-400" />
              <div>
                <h3 className="text-white font-semibold text-lg">历史会议</h3>
                <p className="text-gray-400 text-sm">查看过往记录和分析</p>
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-effect cursor-pointer card-hover">
            <CardContent className="flex items-center space-x-4 p-6">
              <BarChart3 className="w-10 h-10 text-green-400" />
              <div>
                <h3 className="text-white font-semibold text-lg">效率分析</h3>
                <p className="text-gray-400 text-sm">深度数据洞察报告</p>
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-effect cursor-pointer card-hover">
            <CardContent className="flex items-center space-x-4 p-6">
              <Settings className="w-10 h-10 text-purple-400" />
              <div>
                <h3 className="text-white font-semibold text-lg">智能体配置</h3>
                <p className="text-gray-400 text-sm">自定义协作策略</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* 实时通知 */}
      <RealTimeNotifications />

      {/* 会议纪要弹窗 */}
      <MeetingMinutesModal 
        open={showMinutes}
        onOpenChange={setShowMinutes}
        collaboration={collaboration}
      />
    </div>
  );
};

export default EnhancedDashboard;