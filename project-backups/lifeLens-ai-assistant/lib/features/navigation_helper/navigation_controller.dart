import 'package:flutter/foundation.dart';
import '../../models/location/location_info.dart';
import '../../services/map/map_service.dart';
import '../../services/ar_display/ar_display_service.dart';
import '../../services/voice/voice_service.dart';

class NavigationController extends ChangeNotifier {
  final MapService _mapService;
  final ARDisplayService _arDisplayService;
  final VoiceService _voiceService;

  NavigationController({
    required MapService mapService,
    required ARDisplayService arDisplayService,
    required VoiceService voiceService,
  })  : _mapService = mapService,
        _arDisplayService = arDisplayService,
        _voiceService = voiceService;

  LocationInfo? _currentLocation;
  NavigationInfo? _currentRoute;
  List<POI> _nearbyPOIs = [];
  POI? _destination;
  bool _isNavigating = false;
  bool _isLoading = false;
  String? _errorMessage;
  int _currentStepIndex = 0;

  LocationInfo? get currentLocation => _currentLocation;
  NavigationInfo? get currentRoute => _currentRoute;
  List<POI> get nearbyPOIs => _nearbyPOIs;
  POI? get destination => _destination;
  bool get isNavigating => _isNavigating;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;
  int get currentStepIndex => _currentStepIndex;

  Future<void> getCurrentLocation() async {
    try {
      _setLoading(true);
      _clearError();

      _currentLocation = await _mapService.getCurrentLocation();
      
      // 搜索附近兴趣点
      await _searchNearbyPOIs();

      notifyListeners();
    } catch (e) {
      _setError('获取位置失败: $e');
    } finally {
      _setLoading(false);
    }
  }

  Future<void> searchNearbyPOI(String type) async {
    if (_currentLocation == null) {
      await getCurrentLocation();
    }

    try {
      _setLoading(true);
      _clearError();

      _nearbyPOIs = await _mapService.searchNearbyPOI(_currentLocation!, type);
      
      if (_nearbyPOIs.isNotEmpty) {
        await _displayNearbyPOIs(_nearbyPOIs, type);
        await _voiceService.textToSpeech(
          '找到${_nearbyPOIs.length}个附近的$type，为您显示最近的几个'
        );
      } else {
        await _voiceService.textToSpeech('附近没有找到$type');
      }

      notifyListeners();
    } catch (e) {
      _setError('搜索$type失败: $e');
    } finally {
      _setLoading(false);
    }
  }

  Future<void> navigateToDestination(POI destination) async {
    if (_currentLocation == null) {
      await getCurrentLocation();
    }

    try {
      _setLoading(true);
      _clearError();

      _destination = destination;
      
      // 获取路线
      _currentRoute = await _mapService.getRoute(
        _currentLocation!,
        destination.location,
        mode: 'walking',
      );

      if (_currentRoute != null) {
        _isNavigating = true;
        _currentStepIndex = 0;
        
        await _displayNavigationInfo(_currentRoute!);
        await _speakNavigationStart(_currentRoute!);
        
        // 开始导航
        await _startNavigation();
      }

      notifyListeners();
    } catch (e) {
      _setError('导航规划失败: $e');
    } finally {
      _setLoading(false);
    }
  }

  Future<void> handleVoiceCommand(String command) async {
    try {
      final lowerCommand = command.toLowerCase();
      
      if (lowerCommand.contains('我在哪里') || lowerCommand.contains('当前位置')) {
        await getCurrentLocation();
        if (_currentLocation != null) {
          await _voiceService.textToSpeech('您当前位置是${_currentLocation!.address}');
        }
      } else if (lowerCommand.contains('地铁站怎么走') || lowerCommand.contains('最近的地铁站')) {
        await searchNearbyPOI('地铁站');
      } else if (lowerCommand.contains('餐厅怎么走') || lowerCommand.contains('附近餐厅')) {
        await searchNearbyPOI('餐厅');
      } else if (lowerCommand.contains('厕所') || lowerCommand.contains('洗手间')) {
        await searchNearbyPOI('厕所');
      } else if (lowerCommand.contains('银行') || lowerCommand.contains('ATM')) {
        await searchNearbyPOI('银行');
      } else if (lowerCommand.contains('下一步') && _isNavigating) {
        await _nextNavigationStep();
      } else if (lowerCommand.contains('停止导航') && _isNavigating) {
        await _stopNavigation();
      } else {
        await _voiceService.textToSpeech('请说"最近的地铁站怎么走"或其他导航需求');
      }
    } catch (e) {
      _setError('处理语音命令失败: $e');
    }
  }

  Future<void> _searchNearbyPOIs() async {
    if (_currentLocation == null) return;

    try {
      // 搜索常用的POI类型
      final types = ['地铁站', '餐厅', '银行', '医院'];
      final allPOIs = <POI>[];

      for (final type in types) {
        final pois = await _mapService.searchNearbyPOI(
          _currentLocation!,
          type,
          radius: 500,
        );
        allPOIs.addAll(pois.take(2)); // 每种类型取前2个
      }

      _nearbyPOIs = allPOIs;
    } catch (e) {
      // 忽略错误，不影响主要功能
    }
  }

  Future<void> _displayNearbyPOIs(List<POI> pois, String type) async {
    // 显示搜索结果标题
    await _arDisplayService.showText(
      '附近的$type',
      position: ARPosition(x: 0.5, y: 0.15),
      style: ARStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
    );

    // 显示前3个POI
    for (int i = 0; i < pois.length && i < 3; i++) {
      final poi = pois[i];
      final yPosition = 0.3 + (i * 0.15);
      
      await _arDisplayService.showText(
        '${i + 1}. ${poi.name}',
        position: ARPosition(x: 0.5, y: yPosition),
        style: ARStyle(fontSize: 14.0),
      );
      
      await _arDisplayService.showText(
        '${poi.distance < 1000 ? '${poi.distance.toInt()}米' : '${(poi.distance / 1000).toStringAsFixed(1)}公里'}',
        position: ARPosition(x: 0.5, y: yPosition + 0.05),
        style: ARStyle(fontSize: 12.0, textColor: const Color(0xFF4CAF50)),
      );
      
      // 显示方向箭头
      await _arDisplayService.showDirection(
        _getDirectionText(poi),
        _calculateAngle(_currentLocation!, poi.location),
      );
    }
  }

  Future<void> _displayNavigationInfo(NavigationInfo route) async {
    // 显示目的地
    await _arDisplayService.showText(
      '导航到: ${_destination!.name}',
      position: ARPosition(x: 0.5, y: 0.15),
      style: ARStyle(fontSize: 16.0, fontWeight: FontWeight.bold),
    );

    // 显示总距离和时间
    await _arDisplayService.showText(
      '${route.distanceDisplay} • ${route.durationDisplay}',
      position: ARPosition(x: 0.5, y: 0.25),
      style: ARStyle(fontSize: 14.0),
    );

    // 显示当前步骤
    if (route.steps.isNotEmpty) {
      await _displayCurrentStep(route.steps[_currentStepIndex]);
    }
  }

  Future<void> _displayCurrentStep(RouteStep step) async {
    // 显示当前步骤指令
    await _arDisplayService.showText(
      step.instruction,
      position: ARPosition(x: 0.5, y: 0.4),
      style: ARStyle(fontSize: 14.0, backgroundColor: const Color(0xCC2196F3)),
    );

    // 显示距离
    await _arDisplayService.showText(
      '${step.distance < 1000 ? '${step.distance.toInt()}米' : '${(step.distance / 1000).toStringAsFixed(1)}公里'}',
      position: ARPosition(x: 0.5, y: 0.5),
      style: ARStyle(fontSize: 12.0),
    );

    // 显示方向指示
    await _arDisplayService.showDirection(
      step.direction,
      _calculateAngle(step.startLocation, step.endLocation),
    );
  }

  Future<void> _speakNavigationStart(NavigationInfo route) async {
    await _voiceService.textToSpeech(
      '开始导航到${_destination!.name}，总距离${route.distanceDisplay}，'
      '预计${route.durationDisplay}。${route.steps.first.instruction}'
    );
  }

  Future<void> _startNavigation() async {
    // 模拟导航过程
    // 实际实现中应该持续更新位置并检查是否需要下一步指令
  }

  Future<void> _nextNavigationStep() async {
    if (_currentRoute == null || !_isNavigating) return;

    if (_currentStepIndex < _currentRoute!.steps.length - 1) {
      _currentStepIndex++;
      final nextStep = _currentRoute!.steps[_currentStepIndex];
      
      await _displayCurrentStep(nextStep);
      await _voiceService.textToSpeech(nextStep.instruction);
      
      notifyListeners();
    } else {
      // 到达目的地
      await _voiceService.textToSpeech('已到达目的地${_destination!.name}');
      await _stopNavigation();
    }
  }

  Future<void> _stopNavigation() async {
    _isNavigating = false;
    _currentRoute = null;
    _destination = null;
    _currentStepIndex = 0;
    
    await _arDisplayService.hideAllOverlays();
    await _voiceService.textToSpeech('导航已停止');
    
    notifyListeners();
  }

  String _getDirectionText(POI poi) {
    // 简化的方向计算
    if (poi.distance < 100) return '就在附近';
    if (poi.distance < 500) return '很近';
    return '较远';
  }

  double _calculateAngle(LocationInfo from, LocationInfo to) {
    // 简化的角度计算
    final deltaLat = to.latitude - from.latitude;
    final deltaLng = to.longitude - from.longitude;
    return (deltaLng > 0 ? 90 : 270) + (deltaLat > 0 ? 0 : 180);
  }

  void _setLoading(bool loading) {
    _isLoading = loading;
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