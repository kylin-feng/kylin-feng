import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  FileText, 
  Download, 
  Search, 
  Filter,
  Clock,
  User,
  MessageSquare,
  Copy,
  Check
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface TranscriptionSegment {
  id: string;
  speaker: string;
  text: string;
  timestamp: Date;
  confidence: number;
  duration: number;
}

interface TranscriptionPanelProps {
  segments: TranscriptionSegment[];
  isRecording: boolean;
}

const TranscriptionPanel: React.FC<TranscriptionPanelProps> = ({
  segments,
  isRecording
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSpeaker, setSelectedSpeaker] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // 获取所有发言人
  const speakers = Array.from(new Set(segments.map(s => s.speaker)));

  // 过滤转录片段
  const filteredSegments = segments.filter(segment => {
    const matchesSearch = searchQuery === '' || 
      segment.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      segment.speaker.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesSpeaker = !selectedSpeaker || segment.speaker === selectedSpeaker;
    
    return matchesSearch && matchesSpeaker;
  });

  // 自动滚动到最新消息
  useEffect(() => {
    if (isRecording && scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [segments, isRecording]);

  // 复制文本
  const copyText = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      console.error('复制失败:', error);
    }
  };

  // 导出转录文本
  const exportTranscription = () => {
    const content = filteredSegments
      .map(segment => `[${segment.timestamp.toLocaleTimeString()}] ${segment.speaker}: ${segment.text}`)
      .join('\n\n');
    
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `meeting-transcription-${new Date().toISOString().slice(0, 19)}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // 格式化时间
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('zh-CN', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  // 获取置信度颜色
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400';
    if (confidence >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  // 获取发言人颜色
  const getSpeakerColor = (speaker: string) => {
    const colors = [
      'bg-blue-500',
      'bg-green-500', 
      'bg-purple-500',
      'bg-yellow-500',
      'bg-pink-500',
      'bg-indigo-500'
    ];
    const hash = speaker.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return colors[hash % colors.length];
  };

  return (
    <Card className="glass-effect h-full flex flex-col">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2 text-white">
            <FileText className="w-5 h-5" />
            <span>实时转录记录</span>
            {isRecording && (
              <Badge className="bg-red-500/20 text-red-300 border-red-500/50 animate-pulse">
                录制中
              </Badge>
            )}
          </CardTitle>
          
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="border-gray-600 text-gray-300">
              {filteredSegments.length} 条记录
            </Badge>
            <Button
              onClick={exportTranscription}
              size="sm"
              variant="outline"
              className="border-gray-600 text-gray-300 hover:text-white"
              disabled={segments.length === 0}
            >
              <Download className="w-4 h-4 mr-1" />
              导出
            </Button>
          </div>
        </div>

        {/* 搜索和过滤 */}
        <div className="flex items-center space-x-2 mt-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="搜索转录内容..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-white/5 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
            />
          </div>
          
          <select
            value={selectedSpeaker || ''}
            onChange={(e) => setSelectedSpeaker(e.target.value || null)}
            className="px-3 py-2 bg-white/5 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-purple-500"
          >
            <option value="">所有发言人</option>
            {speakers.map(speaker => (
              <option key={speaker} value={speaker} className="bg-gray-800">
                {speaker}
              </option>
            ))}
          </select>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0">
        <ScrollArea className="h-full px-6 pb-6" ref={scrollAreaRef}>
          {filteredSegments.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 text-gray-400">
              <MessageSquare className="w-12 h-12 mb-4 opacity-50" />
              <p className="text-lg mb-2">暂无转录记录</p>
              <p className="text-sm">
                {isRecording ? '正在等待语音输入...' : '开始录制以查看实时转录'}
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {filteredSegments.map((segment, index) => (
                <div
                  key={segment.id}
                  className={cn(
                    "group relative bg-white/5 rounded-lg p-4 border transition-all duration-200",
                    "hover:bg-white/10 hover:border-gray-500",
                    index === filteredSegments.length - 1 && isRecording ? "border-blue-500/50" : "border-gray-700"
                  )}
                >
                  {/* 发言人和时间信息 */}
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center space-x-2">
                      <div className={cn(
                        "w-3 h-3 rounded-full",
                        getSpeakerColor(segment.speaker)
                      )} />
                      <span className="text-white font-medium">{segment.speaker}</span>
                      <Badge 
                        variant="outline" 
                        className="text-xs border-gray-600 text-gray-400"
                      >
                        <User className="w-3 h-3 mr-1" />
                        {segment.duration}s
                      </Badge>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className={cn(
                        "text-xs font-mono",
                        getConfidenceColor(segment.confidence)
                      )}>
                        {Math.round(segment.confidence * 100)}%
                      </span>
                      <div className="flex items-center space-x-1 text-gray-400 text-xs">
                        <Clock className="w-3 h-3" />
                        <span>{formatTime(segment.timestamp)}</span>
                      </div>
                    </div>
                  </div>

                  {/* 转录文本 */}
                  <p className="text-gray-100 leading-relaxed mb-3">
                    {segment.text}
                  </p>

                  {/* 操作按钮 */}
                  <div className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <Button
                      onClick={() => copyText(segment.text, segment.id)}
                      size="sm"
                      variant="ghost"
                      className="text-gray-400 hover:text-white p-1 h-auto"
                    >
                      {copiedId === segment.id ? (
                        <Check className="w-3 h-3" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </Button>
                  </div>

                  {/* 置信度指示器 */}
                  <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-700 rounded-b-lg overflow-hidden">
                    <div 
                      className={cn(
                        "h-full transition-all duration-300",
                        segment.confidence >= 0.8 ? "bg-green-500" :
                        segment.confidence >= 0.6 ? "bg-yellow-500" : "bg-red-500"
                      )}
                      style={{ width: `${segment.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export default TranscriptionPanel;