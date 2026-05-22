import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'object_recognition_controller.dart';

class ObjectRecognitionPage extends StatefulWidget {
  const ObjectRecognitionPage({Key? key}) : super(key: key);

  @override
  State<ObjectRecognitionPage> createState() => _ObjectRecognitionPageState();
}

class _ObjectRecognitionPageState extends State<ObjectRecognitionPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Consumer<ObjectRecognitionController>(
        builder: (context, controller, child) {
          return Stack(
            children: [
              // 相机预览区域
              const CameraPreviewWidget(),
              
              // AR叠加信息
              if (controller.currentResult != null)
                AROverlayWidget(
                  result: controller.currentResult!,
                  objectInfo: controller.currentObjectInfo,
                ),
              
              // 加载指示器
              if (controller.isRecognizing)
                const Center(
                  child: CircularProgressIndicator(
                    color: Colors.white,
                  ),
                ),
              
              // 错误信息
              if (controller.errorMessage != null)
                Positioned(
                  bottom: 100,
                  left: 20,
                  right: 20,
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.red.withOpacity(0.8),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      controller.errorMessage!,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                      ),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
              
              // 控制按钮
              Positioned(
                bottom: 30,
                left: 0,
                right: 0,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    FloatingActionButton(
                      onPressed: controller.isRecognizing 
                          ? null 
                          : controller.recognizeObjectFromCamera,
                      backgroundColor: Colors.blue,
                      child: const Icon(Icons.camera_alt),
                    ),
                    if (controller.currentResult != null)
                      FloatingActionButton(
                        onPressed: () => _showDetailedInfo(controller),
                        backgroundColor: Colors.green,
                        child: const Icon(Icons.info),
                      ),
                  ],
                ),
              ),
              
              // 语音指示
              const Positioned(
                top: 50,
                left: 0,
                right: 0,
                child: VoiceIndicatorWidget(),
              ),
            ],
          );
        },
      ),
    );
  }

  void _showDetailedInfo(ObjectRecognitionController controller) {
    if (controller.currentObjectInfo == null) return;
    
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.black.withOpacity(0.9),
      builder: (context) => DetailedInfoBottomSheet(
        objectInfo: controller.currentObjectInfo!,
      ),
    );
  }
}

class CameraPreviewWidget extends StatelessWidget {
  const CameraPreviewWidget({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: double.infinity,
      color: Colors.grey[900],
      child: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.camera_alt,
              size: 100,
              color: Colors.grey,
            ),
            SizedBox(height: 20),
            Text(
              '相机预览',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 18,
              ),
            ),
            SizedBox(height: 10),
            Text(
              '说"这是什么"开始识别',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class AROverlayWidget extends StatelessWidget {
  final dynamic result;
  final dynamic objectInfo;
  
  const AROverlayWidget({
    Key? key,
    required this.result,
    this.objectInfo,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 120,
      left: 20,
      right: 20,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.7),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.blue, width: 2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // 物体名称
            Text(
              result.objectName,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            
            // 类别和置信度
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.blue,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    result.category,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 12,
                    ),
                  ),
                ),
                const Spacer(),
                Text(
                  '置信度: ${(result.confidence * 100).toInt()}%',
                  style: const TextStyle(
                    color: Colors.green,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            
            // 描述
            if (objectInfo?.description != null)
              Text(
                objectInfo.description.length > 100
                    ? '${objectInfo.description.substring(0, 97)}...'
                    : objectInfo.description,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 14,
                ),
              ),
            
            // 特殊警告
            if (objectInfo?.isPoisonous == true)
              Container(
                margin: const EdgeInsets.only(top: 8),
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.8),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.warning, color: Colors.white, size: 16),
                    SizedBox(width: 8),
                    Text(
                      '⚠️ 有毒！请勿食用',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            
            // 标签
            if (result.tags.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Wrap(
                  spacing: 4,
                  children: result.tags.take(4).map<Widget>((tag) {
                    return Chip(
                      label: Text(
                        tag,
                        style: const TextStyle(fontSize: 10),
                      ),
                      backgroundColor: Colors.grey[700],
                      labelStyle: const TextStyle(color: Colors.white),
                    );
                  }).toList(),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class VoiceIndicatorWidget extends StatefulWidget {
  const VoiceIndicatorWidget({Key? key}) : super(key: key);

  @override
  State<VoiceIndicatorWidget> createState() => _VoiceIndicatorWidgetState();
}

class _VoiceIndicatorWidgetState extends State<VoiceIndicatorWidget>
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
    _animationController.repeat(reverse: true);
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(_animation.value * 0.7),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.mic,
                      color: Colors.white,
                      size: 16,
                    ),
                    SizedBox(width: 8),
                    Text(
                      '说"这是什么"开始识别',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class DetailedInfoBottomSheet extends StatelessWidget {
  final dynamic objectInfo;
  
  const DetailedInfoBottomSheet({
    Key? key,
    required this.objectInfo,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  objectInfo.name,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.close, color: Colors.white),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          Text(
            '类别: ${objectInfo.category}',
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 12),
          
          Text(
            objectInfo.description,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              height: 1.5,
            ),
          ),
          
          if (objectInfo.features.isNotEmpty) ...[
            const SizedBox(height: 16),
            const Text(
              '主要特征:',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            ...objectInfo.features.map<Widget>((feature) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                  children: [
                    const Icon(Icons.check, color: Colors.green, size: 16),
                    const SizedBox(width: 8),
                    Text(
                      feature,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ],
          
          const SizedBox(height: 20),
        ],
      ),
    );
  }
}