import 'dart:ui';

abstract class ARDisplayService {
  Future<void> showText(String text, {ARPosition? position, ARStyle? style});
  Future<void> showImage(String imageUrl, {ARPosition? position, Size? size});
  Future<void> showDirection(String direction, double angle);
  Future<void> showOverlay(AROverlay overlay);
  Future<void> hideAllOverlays();
  Future<void> hideOverlay(String overlayId);
  Future<bool> isSupported();
}

class RokidARDisplayService implements ARDisplayService {
  final List<AROverlay> _activeOverlays = [];
  
  @override
  Future<void> showText(String text, {ARPosition? position, ARStyle? style}) async {
    try {
      final overlay = AROverlay(
        id: 'text_${DateTime.now().millisecondsSinceEpoch}',
        type: AROverlayType.text,
        content: text,
        position: position ?? ARPosition.center(),
        style: style ?? ARStyle.defaultText(),
        duration: const Duration(seconds: 5),
      );
      
      await showOverlay(overlay);
    } catch (e) {
      throw ARDisplayException('显示文字失败: $e');
    }
  }

  @override
  Future<void> showImage(String imageUrl, {ARPosition? position, Size? size}) async {
    try {
      final overlay = AROverlay(
        id: 'image_${DateTime.now().millisecondsSinceEpoch}',
        type: AROverlayType.image,
        content: imageUrl,
        position: position ?? ARPosition.center(),
        size: size ?? const Size(100, 100),
        duration: const Duration(seconds: 3),
      );
      
      await showOverlay(overlay);
    } catch (e) {
      throw ARDisplayException('显示图片失败: $e');
    }
  }

  @override
  Future<void> showDirection(String direction, double angle) async {
    try {
      final overlay = AROverlay(
        id: 'direction_${DateTime.now().millisecondsSinceEpoch}',
        type: AROverlayType.direction,
        content: direction,
        position: ARPosition.centerTop(),
        style: ARStyle.direction(),
        rotation: angle,
        duration: const Duration(seconds: 10),
      );
      
      await showOverlay(overlay);
    } catch (e) {
      throw ARDisplayException('显示方向指示失败: $e');
    }
  }

  @override
  Future<void> showOverlay(AROverlay overlay) async {
    try {
      // 调用Rokid AR SDK显示叠加层
      await Future.delayed(const Duration(milliseconds: 100));
      
      _activeOverlays.add(overlay);
      print('📱 AR显示: ${overlay.content} at ${overlay.position}');
      
      // 自动隐藏
      if (overlay.duration != null) {
        Future.delayed(overlay.duration!, () {
          hideOverlay(overlay.id);
        });
      }
    } catch (e) {
      throw ARDisplayException('显示AR叠加层失败: $e');
    }
  }

  @override
  Future<void> hideAllOverlays() async {
    try {
      await Future.delayed(const Duration(milliseconds: 50));
      _activeOverlays.clear();
      print('📱 隐藏所有AR叠加层');
    } catch (e) {
      throw ARDisplayException('隐藏AR叠加层失败: $e');
    }
  }

  @override
  Future<void> hideOverlay(String overlayId) async {
    try {
      await Future.delayed(const Duration(milliseconds: 50));
      _activeOverlays.removeWhere((overlay) => overlay.id == overlayId);
      print('📱 隐藏AR叠加层: $overlayId');
    } catch (e) {
      throw ARDisplayException('隐藏AR叠加层失败: $e');
    }
  }

  @override
  Future<bool> isSupported() async {
    try {
      // 检查设备是否支持AR显示
      await Future.delayed(const Duration(milliseconds: 100));
      return true; // 假设支持
    } catch (e) {
      return false;
    }
  }
}

class AROverlay {
  final String id;
  final AROverlayType type;
  final String content;
  final ARPosition position;
  final ARStyle? style;
  final Size? size;
  final double? rotation;
  final Duration? duration;

  AROverlay({
    required this.id,
    required this.type,
    required this.content,
    required this.position,
    this.style,
    this.size,
    this.rotation,
    this.duration,
  });
}

enum AROverlayType {
  text,
  image,
  direction,
  notification,
  menu,
}

class ARPosition {
  final double x; // 屏幕坐标 0-1
  final double y; // 屏幕坐标 0-1
  final double z; // 深度，可选

  ARPosition({required this.x, required this.y, this.z = 0});

  factory ARPosition.center() => ARPosition(x: 0.5, y: 0.5);
  factory ARPosition.centerTop() => ARPosition(x: 0.5, y: 0.2);
  factory ARPosition.centerBottom() => ARPosition(x: 0.5, y: 0.8);
  factory ARPosition.leftCenter() => ARPosition(x: 0.2, y: 0.5);
  factory ARPosition.rightCenter() => ARPosition(x: 0.8, y: 0.5);

  @override
  String toString() => 'ARPosition(x: $x, y: $y, z: $z)';
}

class ARStyle {
  final Color textColor;
  final double fontSize;
  final FontWeight fontWeight;
  final Color backgroundColor;
  final double opacity;
  final double borderRadius;

  ARStyle({
    this.textColor = const Color(0xFFFFFFFF),
    this.fontSize = 16.0,
    this.fontWeight = FontWeight.normal,
    this.backgroundColor = const Color(0x80000000),
    this.opacity = 1.0,
    this.borderRadius = 8.0,
  });

  factory ARStyle.defaultText() => ARStyle();
  
  factory ARStyle.direction() => ARStyle(
    textColor: const Color(0xFF00FF00),
    fontSize: 18.0,
    fontWeight: FontWeight.bold,
    backgroundColor: const Color(0x80000000),
  );
  
  factory ARStyle.notification() => ARStyle(
    textColor: const Color(0xFFFFFFFF),
    fontSize: 14.0,
    backgroundColor: const Color(0xCC2196F3),
  );
  
  factory ARStyle.warning() => ARStyle(
    textColor: const Color(0xFFFFFFFF),
    fontSize: 16.0,
    fontWeight: FontWeight.bold,
    backgroundColor: const Color(0xCCFF9800),
  );
}

class ARDisplayException implements Exception {
  final String message;
  ARDisplayException(this.message);
  
  @override
  String toString() => 'ARDisplayException: $message';
}