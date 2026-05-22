import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../../models/object/recognition_result.dart';
import '../../services/ai_vision/vision_service.dart';
import '../../services/ar_display/ar_display_service.dart';
import '../../services/voice/voice_service.dart';

class ObjectRecognitionController extends ChangeNotifier {
  final VisionService _visionService;
  final ARDisplayService _arDisplayService;
  final VoiceService _voiceService;

  ObjectRecognitionController({
    required VisionService visionService,
    required ARDisplayService arDisplayService,
    required VoiceService voiceService,
  })  : _visionService = visionService,
        _arDisplayService = arDisplayService,
        _voiceService = voiceService;

  RecognitionResult? _currentResult;
  ObjectInfo? _currentObjectInfo;
  bool _isRecognizing = false;
  String? _errorMessage;

  RecognitionResult? get currentResult => _currentResult;
  ObjectInfo? get currentObjectInfo => _currentObjectInfo;
  bool get isRecognizing => _isRecognizing;
  String? get errorMessage => _errorMessage;

  Future<void> recognizeObjectFromCamera() async {
    try {
      _setRecognizing(true);
      _clearError();

      // 这里应该从相机获取图像数据
      final imageData = await _getCameraImage();
      
      // 调用视觉识别服务
      final result = await _visionService.recognizeObject(imageData);
      _currentResult = result;

      // 获取详细的物体信息
      final objectInfo = await _visionService.getObjectInfo(result.objectName);
      _currentObjectInfo = objectInfo;

      // 在AR界面显示结果
      await _displayResult(result, objectInfo);

      // 语音播报结果
      await _speakResult(result, objectInfo);

      notifyListeners();
    } catch (e) {
      _setError('识别失败: $e');
    } finally {
      _setRecognizing(false);
    }
  }

  Future<void> handleVoiceCommand(String command) async {
    try {
      final lowerCommand = command.toLowerCase();
      
      if (lowerCommand.contains('这是什么') || lowerCommand.contains('给我介绍')) {
        await recognizeObjectFromCamera();
      } else if (lowerCommand.contains('详细信息') && _currentResult != null) {
        await _showDetailedInfo();
      } else if (lowerCommand.contains('重新识别')) {
        await recognizeObjectFromCamera();
      } else {
        await _voiceService.textToSpeech('请说"这是什么"来识别物体');
      }
    } catch (e) {
      _setError('处理语音命令失败: $e');
    }
  }

  Future<void> _displayResult(RecognitionResult result, ObjectInfo objectInfo) async {
    // 显示物体名称
    await _arDisplayService.showText(
      result.objectName,
      position: ARPosition.centerTop(),
      style: ARStyle.defaultText(),
    );

    // 显示简短描述
    await _arDisplayService.showText(
      objectInfo.description.length > 50 
          ? '${objectInfo.description.substring(0, 47)}...'
          : objectInfo.description,
      position: ARPosition.center(),
      style: ARStyle(fontSize: 14.0),
    );

    // 显示置信度
    await _arDisplayService.showText(
      '置信度: ${(result.confidence * 100).toInt()}%',
      position: ARPosition.centerBottom(),
      style: ARStyle(
        fontSize: 12.0,
        textColor: const Color(0xFF4CAF50),
      ),
    );

    // 如果有特殊属性，显示警告
    if (objectInfo.isPoisonous) {
      await _arDisplayService.showText(
        '⚠️ 有毒！请勿食用',
        position: ARPosition(x: 0.5, y: 0.3),
        style: ARStyle.warning(),
      );
    }
  }

  Future<void> _speakResult(RecognitionResult result, ObjectInfo objectInfo) async {
    String spokenText = '这是${result.objectName}';
    
    if (objectInfo.category.isNotEmpty) {
      spokenText += '，属于${objectInfo.category}';
    }
    
    if (objectInfo.isPoisonous) {
      spokenText += '。注意，这个物品有毒，请勿接触或食用';
    } else if (objectInfo.isEdible) {
      spokenText += '，可以食用';
    }
    
    if (objectInfo.features.isNotEmpty) {
      spokenText += '。主要特征包括：${objectInfo.features.take(3).join('、')}';
    }

    await _voiceService.textToSpeech(spokenText);
  }

  Future<void> _showDetailedInfo() async {
    if (_currentObjectInfo == null) return;

    final info = _currentObjectInfo!;
    
    // 隐藏当前显示的内容
    await _arDisplayService.hideAllOverlays();
    
    // 显示详细信息
    await _arDisplayService.showText(
      '${info.name} - 详细信息',
      position: ARPosition(x: 0.5, y: 0.15),
      style: ARStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
    );
    
    await _arDisplayService.showText(
      info.description,
      position: ARPosition(x: 0.5, y: 0.35),
      style: ARStyle(fontSize: 14.0),
    );
    
    if (info.features.isNotEmpty) {
      await _arDisplayService.showText(
        '特征: ${info.features.join(', ')}',
        position: ARPosition(x: 0.5, y: 0.55),
        style: ARStyle(fontSize: 12.0),
      );
    }
    
    // 语音播报详细信息
    await _voiceService.textToSpeech(
      '${info.name}的详细信息：${info.description}'
    );
  }

  Future<Uint8List> _getCameraImage() async {
    // 模拟从相机获取图像数据
    // 实际实现中应该调用相机API
    await Future.delayed(const Duration(milliseconds: 500));
    return Uint8List.fromList([1, 2, 3, 4, 5]); // 模拟图像数据
  }

  void _setRecognizing(bool recognizing) {
    _isRecognizing = recognizing;
    notifyListeners();
  }

  void _setError(String error) {
    _errorMessage = error;
    notifyListeners();
  }

  void _clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  @override
  void dispose() {
    super.dispose();
  }
}