import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import VoiceRecorder from '@/components/features/VoiceRecorder';
import { useWebSocket, useAgentUpdates, useCollaborationUpdates } from '@/hooks/useWebSocket';
import { Agent, Meeting, AgentCollaboration } from '@/types';
import { agentApi, meetingApi } from '@/services/api';
import { 
  Play, 
  Square, 
  Users, 
  Brain, 
  FileText, 
  Mic,
  MicOff,
  Clock,
  Activity,
  CheckCircle,
  ArrowLeft
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface TranscriptionSegment {
  id: string;
  speaker: string;
  text: string;
  timestamp: Date;
  confidence: number;
  duration: number;
}

const MinimalDashboard: React.FC = () => {
  const navigate = useNavigate();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [currentMeeting, setCurrentMeeting] = useState<Meeting | null>(null);
  const [collaboration, setCollaboration] = useState<AgentCollaboration | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [transcriptionSegments, setTranscriptionSegments] = useState<TranscriptionSegment[]>([]);
  const [meetingStats, setMeetingStats] = useState({
    duration: 0,
    segmentCount: 0,
    activeAgents: 0
  });

  // WebSocket连接
  const { 
    isConnected, 
    notifications, 
    clearNotifications 
  } = useWebSocket('ws://localhost:5001');

  // 智能体状态更新
  useAgentUpdates((updatedAgent) => {
    setAgents(prev => prev.map(agent => 
      agent.id === updatedAgent.id ? updatedAgent : agent
    ));
  });

  // 协作状态更新
  useCollaborationUpdates((updatedCollaboration) => {
    setCollaboration(updatedCollaboration);
  });

  // 初始化智能体
  useEffect(() => {
    const initializeAgents = async () => {
      try {
        const response = await agentApi.getAgents();
        if (response.success) {
          setAgents(response.data);
        }
      } catch (error) {
        console.error('获取智能体列表失败:', error);
      }
    };

    initializeAgents();
  }, []);

  // 开始会议
  const startMeeting = async () => {
    setLoading(true);
    try {
      const meetingData = {
        title: `智能会议 ${new Date().toLocaleString()}`,
        participants: ['用户'],
        date: new Date().toISOString()
      };

      const response = await meetingApi.startCollaboration(meetingData);
      if (response.success) {
        setCurrentMeeting(response.data);
        setIsRecording(true);
        clearNotifications();
      }
    } catch (error) {
      console.error('启动会议失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 结束会议
  const stopMeeting = async () => {
    setIsRecording(false);
    setCurrentMeeting(null);
    setCollaboration(null);
    setTranscriptionSegments([]);
  };

  // 处理转录数据
  const handleTranscriptionUpdate = (segment: TranscriptionSegment) => {
    setTranscriptionSegments(prev => [...prev, segment]);
    setMeetingStats(prev => ({
      ...prev,
      segmentCount: prev.segmentCount + 1
    }));
  };

  // 智能体统计
  const agentStats = agents.reduce((acc, agent) => {
    acc[agent.status] = (acc[agent.status] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const activeAgents = agents.filter(agent => agent.status === 'working' || agent.status === 'analyzing');

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b border-slate-100">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => navigate('/')}
                className="rounded-full"
              >
                <ArrowLeft className="w-4 h-4" />
              </Button>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-slate-900 rounded-lg flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <div>
                  <div className="font-medium text-slate-900">智能会议</div>
                  <div className="text-xs text-slate-500">AI 驱动的会议助手</div>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              <Badge 
                variant={isConnected ? "default" : "secondary"}
                className="rounded-full"
              >
                {isConnected ? '已连接' : '断开连接'}
              </Badge>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid lg:grid-cols-3 gap-8">
          
          {/* Main Control Panel */}
          <div className="lg:col-span-2 space-y-6">
            
            {/* Meeting Control */}
            <Card className="border-0 shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-medium text-slate-900">
                  会议控制
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                
                {/* Meeting Status */}
                <div className="text-center py-8">
                  {!isRecording ? (
                    <div className="space-y-4">
                      <div className="w-20 h-20 bg-slate-100 rounded-full flex items-center justify-center mx-auto">
                        <Play className="w-8 h-8 text-slate-600" />
                      </div>
                      <div>
                        <div className="text-lg font-medium text-slate-900 mb-2">
                          准备开始会议
                        </div>
                        <div className="text-sm text-slate-500">
                          点击开始按钮启动智能记录
                        </div>
                      </div>
                      <Button 
                        size="lg" 
                        className="rounded-full px-8 bg-slate-900 hover:bg-slate-800"
                        onClick={startMeeting}
                        disabled={loading}
                      >
                        {loading ? '启动中...' : '开始会议'}
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="w-20 h-20 bg-red-50 rounded-full flex items-center justify-center mx-auto">
                        <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse"></div>
                      </div>
                      <div>
                        <div className="text-lg font-medium text-slate-900 mb-2">
                          会议进行中
                        </div>
                        <div className="text-sm text-slate-500">
                          AI正在实时记录和分析
                        </div>
                      </div>
                      <Button 
                        variant="outline" 
                        size="lg" 
                        className="rounded-full px-8"
                        onClick={stopMeeting}
                      >
                        <Square className="w-4 h-4 mr-2" />
                        结束会议
                      </Button>
                    </div>
                  )}
                </div>

                {/* Voice Recorder */}
                {isRecording && currentMeeting && (
                  <div className="border-t border-slate-100 pt-6">
                    <VoiceRecorder
                      sessionId={currentMeeting.sessionId}
                      onTranscriptionUpdate={handleTranscriptionUpdate}
                      isRecording={isRecording}
                    />
                  </div>
                )}

              </CardContent>
            </Card>

            {/* Meeting Stats */}
            {isRecording && (
              <Card className="border-0 shadow-sm">
                <CardHeader className="pb-4">
                  <CardTitle className="text-lg font-medium text-slate-900">
                    会议统计
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-6">
                    <div className="text-center">
                      <div className="text-2xl font-light text-slate-900 mb-1">
                        {Math.floor(meetingStats.duration / 60)}:{(meetingStats.duration % 60).toString().padStart(2, '0')}
                      </div>
                      <div className="text-xs text-slate-500">会议时长</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-light text-slate-900 mb-1">
                        {transcriptionSegments.length}
                      </div>
                      <div className="text-xs text-slate-500">对话片段</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-light text-slate-900 mb-1">
                        {activeAgents.length}
                      </div>
                      <div className="text-xs text-slate-500">活跃助手</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Live Transcription */}
            {transcriptionSegments.length > 0 && (
              <Card className="border-0 shadow-sm">
                <CardHeader className="pb-4">
                  <CardTitle className="text-lg font-medium text-slate-900">
                    实时转录
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 max-h-60 overflow-y-auto">
                    {transcriptionSegments.slice(-5).map((segment) => (
                      <div key={segment.id} className="border-l-2 border-slate-200 pl-4">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-sm font-medium text-slate-900">
                            {segment.speaker}
                          </span>
                          <span className="text-xs text-slate-500">
                            {segment.timestamp.toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-sm text-slate-700 leading-relaxed">
                          {segment.text}
                        </p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            
            {/* AI Agents */}
            <Card className="border-0 shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-medium text-slate-900">
                  AI 助手
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {agents.map((agent) => (
                  <div key={agent.id} className="flex items-center gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                      agent.status === 'working' ? 'bg-green-100' :
                      agent.status === 'analyzing' ? 'bg-blue-100' :
                      'bg-slate-100'
                    }`}>
                      {agent.type === 'recorder' && <Mic className="w-4 h-4" />}
                      {agent.type === 'analyzer' && <Brain className="w-4 h-4" />}
                      {agent.type === 'secretary' && <FileText className="w-4 h-4" />}
                      {agent.type === 'editor' && <Users className="w-4 h-4" />}
                      {agent.type === 'qa' && <CheckCircle className="w-4 h-4" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-slate-900">
                        {agent.name}
                      </div>
                      <div className="text-xs text-slate-500 truncate">
                        {agent.description}
                      </div>
                    </div>
                    <Badge 
                      variant={agent.status === 'working' ? 'default' : 'secondary'}
                      className="text-xs rounded-full"
                    >
                      {agent.status === 'idle' && '待机'}
                      {agent.status === 'working' && '工作中'}
                      {agent.status === 'analyzing' && '分析中'}
                      {agent.status === 'completed' && '完成'}
                    </Badge>
                  </div>
                ))}
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card className="border-0 shadow-sm">
              <CardHeader className="pb-4">
                <CardTitle className="text-lg font-medium text-slate-900">
                  快速操作
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start rounded-lg"
                  disabled={!currentMeeting}
                >
                  <FileText className="w-4 h-4 mr-2" />
                  生成纪要
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start rounded-lg"
                  disabled={!currentMeeting}
                >
                  <Users className="w-4 h-4 mr-2" />
                  会议分析
                </Button>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full justify-start rounded-lg"
                >
                  <Clock className="w-4 h-4 mr-2" />
                  历史记录
                </Button>
              </CardContent>
            </Card>

          </div>
        </div>
      </div>
    </div>
  );
};

export default MinimalDashboard;