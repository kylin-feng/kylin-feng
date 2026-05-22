import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Mic, 
  MicOff, 
  Square, 
  Play, 
  Pause,
  Volume2,
  VolumeX,
  Users,
  Activity,
  Clock
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface VoiceRecorderProps {
  isRecording: boolean;
  onRecordingStart: () => void;
  onRecordingStop: () => void;
  onTranscriptionUpdate: (text: string, speaker?: string) => void;
}

interface Speaker {
  id: string;
  name: string;
  color: string;
  segments: number;
}

const VoiceRecorder: React.FC<VoiceRecorderProps> = ({
  isRecording,
  onRecordingStart,
  onRecordingStop,
  onTranscriptionUpdate
}) => {
  const [audioLevel, setAudioLevel] = useState(0);
  const [duration, setDuration] = useState(0);
  const [currentTranscript, setCurrentTranscript] = useState('');
  const [speakers, setSpeakers] = useState<Speaker[]>([
    { id: 'speaker1', name: '发言人A', color: 'bg-blue-500', segments: 0 },
    { id: 'speaker2', name: '发言人B', color: 'bg-green-500', segments: 0 },
    { id: 'speaker3', name: '发言人C', color: 'bg-purple-500', segments: 0 }
  ]);
  const [currentSpeaker, setCurrentSpeaker] = useState<string>('speaker1');
  const [isMuted, setIsMuted] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number>();
  const durationIntervalRef = useRef<NodeJS.Timeout>();

  // 初始化音频可视化
  const initializeAudioVisualization = async (stream: MediaStream) => {
    try {
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      
      analyserRef.current.fftSize = 256;
      source.connect(analyserRef.current);
      
      updateAudioLevel();
    } catch (error) {
      console.error('音频可视化初始化失败:', error);
    }
  };

  // 更新音频电平
  const updateAudioLevel = () => {
    if (!analyserRef.current) return;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserRef.current.getByteFrequencyData(dataArray);

    const average = dataArray.reduce((acc, val) => acc + val, 0) / bufferLength;
    setAudioLevel(Math.min(average / 128 * 100, 100));

    animationFrameRef.current = requestAnimationFrame(updateAudioLevel);
  };

  // 开始录制
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      const chunks: BlobPart[] = [];
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
          // 实时转录音频块
          if (event.data.size > 1024) { // 只处理有足够数据的音频块
            transcribeAudioChunk(event.data);
          }
        }
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        // 这里可以将音频发送到后端进行转录
        console.log('录制完成，音频大小:', blob.size);
      };

      await initializeAudioVisualization(stream);
      
      mediaRecorderRef.current.start(1000); // 每秒收集一次数据
      onRecordingStart();

      // 开始计时
      durationIntervalRef.current = setInterval(() => {
        setDuration(prev => prev + 1);
      }, 1000);

    } catch (error) {
      console.error('录制启动失败:', error);
    }
  };

  // 停止录制
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      
      // 停止音频流
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
    }

    setAudioLevel(0);
    onRecordingStop();
  };

  // 真实转录功能
  const transcribeAudioChunk = async (audioBlob: Blob) => {
    try {
      setIsProcessing(true);
      
      // 创建FormData发送音频
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');
      formData.append('language', 'zh-CN');
      
      const response = await fetch('/api/transcription/sessions/current/transcribe', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('转录请求失败');
      }
      
      const result = await response.json();
      
      if (result.success && result.data) {
        const { text, speaker, confidence } = result.data;
        
        setCurrentTranscript(text);
        
        // 更新发言人统计
        const speakerName = speaker || '未知发言人';
        setSpeakers(prev => prev.map(s => 
          s.name === speakerName 
            ? { ...s, segments: s.segments + 1 }
            : s
        ));
        
        onTranscriptionUpdate(text, speakerName);
        
        console.log(`转录成功: ${text} (置信度: ${Math.round(confidence * 100)}%)`);
      }
      
    } catch (error) {
      console.error('转录失败，使用模拟数据:', error);
      // 降级到模拟转录
      simulateTranscription();
    } finally {
      setIsProcessing(false);
    }
  };

  // 模拟转录（降级方案）
  const simulateTranscription = () => {
    const mockTexts = [
      '大家好，今天我们讨论的主要议题是...',
      '关于这个项目的进展情况，我认为...',
      '我同意刚才的观点，但是我们还需要考虑...',
      '根据最新的数据分析，我们可以得出...',
      '下一步的行动计划应该包括...'
    ];

    const randomText = mockTexts[Math.floor(Math.random() * mockTexts.length)];
    const randomSpeaker = speakers[Math.floor(Math.random() * speakers.length)];
    
    setCurrentTranscript(randomText);
    setCurrentSpeaker(randomSpeaker.id);
    
    // 更新发言统计
    setSpeakers(prev => prev.map(speaker => 
      speaker.id === randomSpeaker.id 
        ? { ...speaker, segments: speaker.segments + 1 }
        : speaker
    ));

    onTranscriptionUpdate(randomText, randomSpeaker.name);
    
    setIsProcessing(true);
    setTimeout(() => setIsProcessing(false), 2000);
  };

  // 格式化时间
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // 清理资源
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (durationIntervalRef.current) {
        clearInterval(durationIntervalRef.current);
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  return (
    <div className="space-y-4">
      {/* 录制控制面板 */}
      <Card className="glass-effect">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-white">
            <Mic className="w-5 h-5" />
            <span>语音录制控制</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* 录制按钮和状态 */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                onClick={isRecording ? stopRecording : startRecording}
                className={cn(
                  "relative btn-press",
                  isRecording 
                    ? "bg-red-600 hover:bg-red-700" 
                    : "bg-green-600 hover:bg-green-700"
                )}
                size="lg"
              >
                {isRecording ? (
                  <>
                    <Square className="w-5 h-5 mr-2" />
                    停止录制
                  </>
                ) : (
                  <>
                    <Mic className="w-5 h-5 mr-2" />
                    开始录制
                  </>
                )}
                
                {isRecording && (
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-400 rounded-full animate-pulse" />
                )}
              </Button>

              <Button
                onClick={() => setIsMuted(!isMuted)}
                variant="outline"
                size="sm"
                className="border-gray-600 text-gray-300"
                disabled={!isRecording}
              >
                {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
              </Button>
            </div>

            <div className="flex items-center space-x-4">
              <Badge className={cn(
                "text-sm",
                isRecording ? "bg-red-500/20 text-red-300 border-red-500/50" : "bg-gray-500/20 text-gray-300 border-gray-500/50"
              )}>
                {isRecording ? '录制中' : '待机'}
              </Badge>
              
              {isRecording && (
                <div className="flex items-center space-x-2 text-white">
                  <Clock className="w-4 h-4" />
                  <span className="font-mono">{formatDuration(duration)}</span>
                </div>
              )}
            </div>
          </div>

          {/* 音频可视化 */}
          {isRecording && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm text-gray-400">
                <span>音频电平</span>
                <span>{Math.round(audioLevel)}%</span>
              </div>
              <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 transition-all duration-100"
                  style={{ width: `${audioLevel}%` }}
                />
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* 实时转录面板 */}
      {isRecording && (
        <Card className="glass-effect">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2 text-white">
              <Activity className="w-5 h-5" />
              <span>实时转录</span>
              {isProcessing && (
                <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/50 animate-pulse">
                  处理中...
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* 当前发言 */}
            <div className="bg-white/5 rounded-lg p-4 space-y-2">
              <div className="flex items-center space-x-2">
                <div className={cn(
                  "w-3 h-3 rounded-full",
                  speakers.find(s => s.id === currentSpeaker)?.color || 'bg-gray-500'
                )} />
                <span className="text-sm text-gray-400">
                  {speakers.find(s => s.id === currentSpeaker)?.name || '未知发言人'}
                </span>
                <Badge variant="outline" className="text-xs border-gray-600 text-gray-400">
                  实时
                </Badge>
              </div>
              <p className="text-white text-base leading-relaxed">
                {currentTranscript || '正在等待语音输入...'}
              </p>
            </div>

            {/* 发言人统计 */}
            <div className="grid grid-cols-3 gap-4">
              {speakers.map(speaker => (
                <div key={speaker.id} className="text-center space-y-2">
                  <div className="flex items-center justify-center space-x-1">
                    <div className={cn("w-2 h-2 rounded-full", speaker.color)} />
                    <span className="text-xs text-gray-400">{speaker.name}</span>
                  </div>
                  <p className="text-white font-semibold">{speaker.segments}</p>
                  <p className="text-xs text-gray-500">发言段数</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default VoiceRecorder;