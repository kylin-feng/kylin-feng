import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'config.dart';

class AIService {

  /// 分析皮肤年龄
  static Future<SkinAnalysisResult> analyzeSkinAge(String imagePath) async {
    try {
      // 读取图片文件并转换为base64
      final File imageFile = File(imagePath);
      final List<int> imageBytes = await imageFile.readAsBytes();
      final String base64Image = base64Encode(imageBytes);
      
      // 构造请求数据
      final Map<String, dynamic> requestData = {
        'model': Config.multiModalModel,
        'messages': [
          {
            'role': 'user',
            'content': [
              {
                'type': 'text',
                'text': '''请分析这张面部照片的皮肤年龄。请根据以下特征进行评估：
1. 皮肤纹理和细纹
2. 皮肤弹性和紧致度
3. 色素沉着和斑点
4. 毛孔大小
5. 皮肤光泽度

请给出一个大致的皮肤年龄估算（范围），并简要说明判断依据。请用中文回答，格式如下：
皮肤年龄：XX-XX岁
主要特征：简要描述1-2个主要观察到的特征
建议：简单的护肤建议'''
              },
              {
                'type': 'image_url',
                'image_url': {
                  'url': 'data:image/jpeg;base64,$base64Image'
                }
              }
            ]
          }
        ],
        'max_tokens': 500,
        'temperature': 0.3
      };

      // 发送请求
      final response = await http.post(
        Uri.parse(Config.siliconFlowBaseUrl),
        headers: {
          'Authorization': 'Bearer ${Config.siliconFlowApiKey}',
          'Content-Type': 'application/json',
        },
        body: jsonEncode(requestData),
      ).timeout(Config.apiTimeout);

      if (response.statusCode == 200) {
        final Map<String, dynamic> responseData = jsonDecode(response.body);
        final String content = responseData['choices'][0]['message']['content'];
        
        return _parseAnalysisResult(content);
      } else {
        print('API调用失败: ${response.statusCode}');
        print('响应: ${response.body}');
        return SkinAnalysisResult.error('API调用失败，请稍后重试');
      }
    } catch (e) {
      print('分析过程中出错: $e');
      return SkinAnalysisResult.error('分析失败：$e');
    }
  }

  /// 解析AI返回的分析结果
  static SkinAnalysisResult _parseAnalysisResult(String content) {
    try {
      // 提取皮肤年龄
      RegExp ageRegex = RegExp(r'皮肤年龄[：:]\s*(\d+)[-~]?(\d+)?岁?');
      Match? ageMatch = ageRegex.firstMatch(content);
      
      int estimatedAge = 25; // 默认值
      if (ageMatch != null) {
        int minAge = int.parse(ageMatch.group(1)!);
        int? maxAge = ageMatch.group(2) != null ? int.parse(ageMatch.group(2)!) : null;
        estimatedAge = maxAge != null ? ((minAge + maxAge) / 2).round() : minAge;
      }

      // 提取主要特征
      RegExp featuresRegex = RegExp(r'主要特征[：:]\s*([^\n]+)');
      Match? featuresMatch = featuresRegex.firstMatch(content);
      String features = featuresMatch?.group(1) ?? '皮肤状态良好';

      // 提取建议
      RegExp adviceRegex = RegExp(r'建议[：:]\s*([^\n]+)');
      Match? adviceMatch = adviceRegex.firstMatch(content);
      String advice = adviceMatch?.group(1) ?? '保持良好的护肤习惯';

      return SkinAnalysisResult(
        estimatedAge: estimatedAge,
        features: features.trim(),
        advice: advice.trim(),
        fullAnalysis: content,
      );
    } catch (e) {
      print('解析结果时出错: $e');
      // 如果解析失败，至少尝试提取数字作为年龄
      RegExp numberRegex = RegExp(r'\d+');
      Iterable<Match> numbers = numberRegex.allMatches(content);
      int estimatedAge = 25;
      if (numbers.isNotEmpty) {
        List<int> ages = numbers.map((m) => int.parse(m.group(0)!)).where((age) => age >= 15 && age <= 80).toList();
        if (ages.isNotEmpty) {
          estimatedAge = ages.first;
        }
      }
      
      return SkinAnalysisResult(
        estimatedAge: estimatedAge,
        features: '基于AI分析的整体评估',
        advice: '建议保持良好的护肤习惯',
        fullAnalysis: content,
      );
    }
  }
}

/// 皮肤分析结果模型
class SkinAnalysisResult {
  final int estimatedAge;
  final String features;
  final String advice;
  final String fullAnalysis;
  final String? error;

  SkinAnalysisResult({
    required this.estimatedAge,
    required this.features,
    required this.advice,
    required this.fullAnalysis,
    this.error,
  });

  SkinAnalysisResult.error(String errorMessage)
      : estimatedAge = 0,
        features = '',
        advice = '',
        fullAnalysis = '',
        error = errorMessage;

  bool get hasError => error != null;
}