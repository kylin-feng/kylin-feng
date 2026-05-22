import apiClient from './api';

export interface TranscriptionSegment {
  id: string;
  sessionId: string;
  text: string;
  speaker: string;
  startTime: number;
  endTime: number;
  confidence: number;
  language: string;
  words?: Array<{
    word: string;
    startTime: number;
    endTime: number;
    confidence: number;
  }>;
  metadata: {
    audioLength: number;
    sampleRate: number;
    channels: number;
  };
  timestamp: string;
}

export interface TranscriptionSession {
  sessionId: string;
  startTime: string;
  endTime?: string;
  duration?: number;
  status: 'active' | 'completed';
  totalSegments: number;
  speakers: string[];
  segments: TranscriptionSegment[];
  options: {
    language: string;
    enableSpeakerDiarization: boolean;
    enablePunctuation: boolean;
    enableWordTimestamps: boolean;
  };
}

export interface SpeakerStatistics {
  name: string;
  segmentCount: number;
  totalDuration: number;
  wordCount: number;
  averageConfidence: number;
}

export interface SearchResult {
  query: string;
  totalResults: number;
  segments: TranscriptionSegment[];
}

// 转录服务API
class TranscriptionApi {
  // 开始转录会话
  async startSession(sessionId: string, options = {}) {
    const response = await apiClient.post('/transcription/sessions', {
      sessionId,
      options
    });
    return response.data;
  }

  // 停止转录会话
  async stopSession(sessionId: string) {
    const response = await apiClient.delete(`/transcription/sessions/${sessionId}`);
    return response.data;
  }

  // 上传音频进行转录
  async transcribeAudio(sessionId: string, audioBlob: Blob, metadata: any = {}) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    
    // 添加元数据
    Object.keys(metadata).forEach(key => {
      formData.append(key, metadata[key]);
    });

    const response = await apiClient.post(
      `/transcription/sessions/${sessionId}/transcribe`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    return response.data;
  }

  // 获取转录结果
  async getSession(sessionId: string): Promise<{ data: TranscriptionSession }> {
    const response = await apiClient.get(`/transcription/sessions/${sessionId}`);
    return response.data;
  }

  // 导出转录结果
  async exportTranscription(sessionId: string, format: 'txt' | 'json' | 'srt' = 'txt') {
    const response = await apiClient.get(`/transcription/sessions/${sessionId}/export`, {
      params: { format },
      responseType: 'blob'
    });
    
    // 创建下载链接
    const blob = new Blob([response.data]);
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `transcription-${sessionId}.${format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    return response.data;
  }

  // 搜索转录内容
  async searchTranscription(
    sessionId: string, 
    query: string, 
    options: {
      caseSensitive?: boolean;
      wholeWord?: boolean;
      speaker?: string;
    } = {}
  ): Promise<{ data: SearchResult }> {
    const response = await apiClient.post(`/transcription/sessions/${sessionId}/search`, {
      query,
      options
    });
    return response.data;
  }

  // 获取发言人统计
  async getSpeakerStatistics(sessionId: string): Promise<{ data: SpeakerStatistics[] }> {
    const response = await apiClient.get(`/transcription/sessions/${sessionId}/speakers`);
    return response.data;
  }

  // 实时音频流转录 (WebRTC)
  async startRealtimeTranscription(
    sessionId: string,
    mediaStream: MediaStream,
    onTranscriptionUpdate: (segment: TranscriptionSegment) => void,
    options: {
      chunkDuration?: number; // 音频块持续时间（毫秒）
      sampleRate?: number;
      channels?: number;
    } = {}
  ) {
    const {
      chunkDuration = 3000, // 3秒
      sampleRate = 16000,
      channels = 1
    } = options;

    // 创建音频处理器
    const audioContext = new AudioContext({ sampleRate });
    const source = audioContext.createMediaStreamSource(mediaStream);
    
    // 使用ScriptProcessorNode进行音频处理
    const processor = audioContext.createScriptProcessor(4096, channels, channels);
    
    let audioChunks: Float32Array[] = [];
    let chunkStartTime = Date.now();
    
    processor.onaudioprocess = (event) => {
      const inputBuffer = event.inputBuffer;
      const channelData = inputBuffer.getChannelData(0);
      
      // 收集音频数据
      audioChunks.push(new Float32Array(channelData));
      
      // 当达到指定的块持续时间时，发送音频进行转录
      if (Date.now() - chunkStartTime >= chunkDuration) {
        this.processAudioChunk(sessionId, audioChunks, {
          sampleRate,
          channels,
          duration: chunkDuration
        }).then(segment => {
          if (segment) {
            onTranscriptionUpdate(segment);
          }
        }).catch(error => {
          console.error('实时转录失败:', error);
        });
        
        // 重置音频块
        audioChunks = [];
        chunkStartTime = Date.now();
      }
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
    
    return {
      stop: () => {
        processor.disconnect();
        source.disconnect();
        audioContext.close();
      }
    };
  }

  // 处理音频块
  private async processAudioChunk(
    sessionId: string,
    audioChunks: Float32Array[],
    metadata: any
  ): Promise<TranscriptionSegment | null> {
    if (audioChunks.length === 0) return null;

    try {
      // 合并音频块
      const totalLength = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
      const mergedAudio = new Float32Array(totalLength);
      let offset = 0;
      
      for (const chunk of audioChunks) {
        mergedAudio.set(chunk, offset);
        offset += chunk.length;
      }

      // 转换为WAV格式
      const wavBlob = this.float32ArrayToWav(mergedAudio, metadata.sampleRate, metadata.channels);
      
      // 发送到服务器进行转录
      const result = await this.transcribeAudio(sessionId, wavBlob, metadata);
      return result.data;
      
    } catch (error) {
      console.error('处理音频块失败:', error);
      return null;
    }
  }

  // 将Float32Array转换为WAV格式
  private float32ArrayToWav(
    buffer: Float32Array,
    sampleRate: number,
    channels: number
  ): Blob {
    const length = buffer.length;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV文件头
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * channels * 2, true);
    view.setUint16(32, channels * 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);
    
    // 音频数据
    let offset = 44;
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, buffer[i]));
      view.setInt16(offset, sample * 0x7FFF, true);
      offset += 2;
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
  }

  // 获取麦克风音频流
  async getMicrophoneStream(constraints: MediaStreamConstraints = {}): Promise<MediaStream> {
    const defaultConstraints = {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000,
        channelCount: 1
      }
    };
    
    const mergedConstraints = {
      ...defaultConstraints,
      ...constraints
    };
    
    try {
      return await navigator.mediaDevices.getUserMedia(mergedConstraints);
    } catch (error) {
      console.error('获取麦克风权限失败:', error);
      throw new Error('无法访问麦克风，请检查权限设置');
    }
  }
}

export const transcriptionApi = new TranscriptionApi();
export default transcriptionApi;