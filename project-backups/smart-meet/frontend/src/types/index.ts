// 智能体类型定义
export interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'idle' | 'working' | 'completed' | 'error';
  progress: number;
  description: string;
  capabilities: string[];
}

// 会议相关类型
export interface Meeting {
  id: string;
  title: string;
  date: string;
  duration: number;
  participants: string[];
  status: 'upcoming' | 'live' | 'completed';
  recording?: string;
}

// 会议纪要类型
export interface MeetingMinutes {
  id: string;
  meetingId: string;
  version: 'executive' | 'technical' | 'management' | 'client';
  content: string;
  keyPoints: string[];
  actionItems: ActionItem[];
  decisions: string[];
  createdAt: string;
  generatedBy: Agent[];
}

// 待办事项类型
export interface ActionItem {
  id: string;
  task: string;
  assignee: string;
  deadline: string;
  priority: 'high' | 'medium' | 'low';
  status: 'pending' | 'in_progress' | 'completed';
}

// 智能体协作状态
export interface AgentCollaboration {
  sessionId: string;
  agents: Agent[];
  currentPhase: 'preparation' | 'recording' | 'processing' | 'review' | 'completed';
  progress: number;
  startTime: string;
  estimatedEndTime: string;
}

// API响应类型
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message: string;
  timestamp: string;
}

// 语音转文字结果
export interface TranscriptionResult {
  text: string;
  speaker: string;
  timestamp: number;
  confidence: number;
}