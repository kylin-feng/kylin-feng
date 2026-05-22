import axios from 'axios';
import { ApiResponse, Agent, Meeting, MeetingMinutes, AgentCollaboration } from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证token
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error);
    return Promise.reject(error);
  }
);

// 智能体相关API
export const agentApi = {
  // 获取所有智能体状态
  getAgents: (): Promise<ApiResponse<Agent[]>> => 
    api.get('/agents').then(res => res.data),
  
  // 启动智能体协作
  startCollaboration: (meetingData: Partial<Meeting>): Promise<ApiResponse<AgentCollaboration>> =>
    api.post('/agents/collaborate', meetingData).then(res => res.data),
  
  // 获取协作状态
  getCollaborationStatus: (sessionId: string): Promise<ApiResponse<AgentCollaboration>> =>
    api.get(`/agents/collaborate/${sessionId}`).then(res => res.data),
};

// 会议相关API  
export const meetingApi = {
  // 获取会议列表
  getMeetings: (): Promise<ApiResponse<Meeting[]>> =>
    api.get('/meetings').then(res => res.data),
  
  // 创建新会议
  createMeeting: (meetingData: Partial<Meeting>): Promise<ApiResponse<Meeting>> =>
    api.post('/meetings', meetingData).then(res => res.data),
  
  // 开始会议录制
  startRecording: (meetingId: string): Promise<ApiResponse<{ success: boolean }>> =>
    api.post(`/meetings/${meetingId}/record`).then(res => res.data),
  
  // 结束会议录制
  stopRecording: (meetingId: string): Promise<ApiResponse<{ success: boolean }>> =>
    api.post(`/meetings/${meetingId}/stop-record`).then(res => res.data),
  
  // 获取会议纪要
  getMeetingMinutes: (meetingId: string): Promise<ApiResponse<MeetingMinutes[]>> =>
    api.get(`/meetings/${meetingId}/minutes`).then(res => res.data),
  
  // 生成会议纪要
  generateMinutes: (meetingId: string, version: string): Promise<ApiResponse<MeetingMinutes>> =>
    api.post(`/meetings/${meetingId}/generate-minutes`, { version }).then(res => res.data),
};

// 实时通信API
export const realtimeApi = {
  // 获取实时转录
  getTranscription: (meetingId: string): Promise<ApiResponse<any>> =>
    api.get(`/realtime/transcription/${meetingId}`).then(res => res.data),
  
  // 发送语音数据
  sendAudioData: (meetingId: string, audioData: Blob): Promise<ApiResponse<any>> => {
    const formData = new FormData();
    formData.append('audio', audioData);
    return api.post(`/realtime/audio/${meetingId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then(res => res.data);
  },
};

export default api;