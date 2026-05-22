import 'dart:typed_data';
import '../../models/memo/voice_memo.dart';

abstract class VoiceService {
  Future<String> speechToText(Uint8List audioData);
  Future<void> textToSpeech(String text);
  Future<bool> startListening();
  Future<void> stopListening();
  Future<VoiceMemo> createMemoFromSpeech(String speechText);
  Future<MemoCategory> categorizeMemo(String content);
}

class SystemVoiceService implements VoiceService {
  bool _isListening = false;
  
  @override
  Future<String> speechToText(Uint8List audioData) async {
    try {
      // 调用系统语音识别或第三方ASR服务
      await Future.delayed(Duration(seconds: 1));
      
      // 模拟语音识别结果
      final mockResults = [
        '明天要买苹果和香蕉',
        '提醒我下午3点开会',
        '这个餐厅怎么样',
        '帮我查一下这个商品的价格',
        '最近的地铁站怎么走',
      ];
      
      return mockResults[DateTime.now().millisecond % mockResults.length];
    } catch (e) {
      throw VoiceServiceException('语音识别失败: $e');
    }
  }

  @override
  Future<void> textToSpeech(String text) async {
    try {
      // 调用系统TTS或第三方语音合成服务
      await Future.delayed(Duration(milliseconds: 500));
      print('🔊 语音播报: $text');
    } catch (e) {
      throw VoiceServiceException('语音合成失败: $e');
    }
  }

  @override
  Future<bool> startListening() async {
    try {
      if (_isListening) return false;
      
      // 启动语音监听
      await Future.delayed(Duration(milliseconds: 100));
      _isListening = true;
      print('🎤 开始监听语音...');
      return true;
    } catch (e) {
      throw VoiceServiceException('启动语音监听失败: $e');
    }
  }

  @override
  Future<void> stopListening() async {
    try {
      if (!_isListening) return;
      
      // 停止语音监听
      await Future.delayed(Duration(milliseconds: 100));
      _isListening = false;
      print('🎤 停止监听语音');
    } catch (e) {
      throw VoiceServiceException('停止语音监听失败: $e');
    }
  }

  @override
  Future<VoiceMemo> createMemoFromSpeech(String speechText) async {
    try {
      final category = await categorizeMemo(speechText);
      final tags = await _extractTags(speechText);
      final priority = _determinePriority(speechText);
      final reminderTime = _extractReminderTime(speechText);
      
      return VoiceMemo(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        content: speechText,
        category: category,
        createdAt: DateTime.now(),
        reminderTime: reminderTime,
        tags: tags,
        priority: priority,
      );
    } catch (e) {
      throw VoiceServiceException('创建语音备忘录失败: $e');
    }
  }

  @override
  Future<MemoCategory> categorizeMemo(String content) async {
    try {
      final lowerContent = content.toLowerCase();
      
      // 基于关键词进行简单分类
      if (_containsKeywords(lowerContent, ['买', '购买', '商品', '超市', '商场'])) {
        return MemoCategory.shopping;
      } else if (_containsKeywords(lowerContent, ['提醒', '记住', '别忘', '定时'])) {
        return MemoCategory.reminder;
      } else if (_containsKeywords(lowerContent, ['工作', '会议', '项目', '任务', '上班'])) {
        return MemoCategory.work;
      } else if (_containsKeywords(lowerContent, ['想法', '灵感', '创意', '主意'])) {
        return MemoCategory.idea;
      } else if (_containsKeywords(lowerContent, ['要做', '完成', '处理', '解决'])) {
        return MemoCategory.todo;
      } else {
        return MemoCategory.note;
      }
    } catch (e) {
      return MemoCategory.note; // 默认分类
    }
  }

  Future<List<String>> _extractTags(String content) async {
    final tags = <String>[];
    final words = content.split(' ');
    
    // 提取可能的标签词
    for (final word in words) {
      if (word.length > 1 && !_isCommonWord(word)) {
        tags.add(word);
      }
    }
    
    return tags.take(5).toList(); // 最多5个标签
  }

  Priority _determinePriority(String content) {
    final lowerContent = content.toLowerCase();
    
    if (_containsKeywords(lowerContent, ['紧急', '立即', '马上', '急'])) {
      return Priority.urgent;
    } else if (_containsKeywords(lowerContent, ['重要', '优先', '尽快'])) {
      return Priority.high;
    } else {
      return Priority.medium;
    }
  }

  DateTime? _extractReminderTime(String content) {
    final lowerContent = content.toLowerCase();
    final now = DateTime.now();
    
    if (lowerContent.contains('明天')) {
      return now.add(Duration(days: 1));
    } else if (lowerContent.contains('下午')) {
      if (lowerContent.contains('3点')) {
        return DateTime(now.year, now.month, now.day, 15, 0);
      } else if (lowerContent.contains('下午')) {
        return DateTime(now.year, now.month, now.day, 14, 0);
      }
    } else if (lowerContent.contains('晚上')) {
      return DateTime(now.year, now.month, now.day, 19, 0);
    }
    
    return null;
  }

  bool _containsKeywords(String content, List<String> keywords) {
    return keywords.any((keyword) => content.contains(keyword));
  }

  bool _isCommonWord(String word) {
    final commonWords = ['的', '了', '在', '是', '我', '你', '他', '她', '它', '这', '那', '和', '与'];
    return commonWords.contains(word);
  }
}

class VoiceServiceException implements Exception {
  final String message;
  VoiceServiceException(this.message);
  
  @override
  String toString() => 'VoiceServiceException: $message';
}