import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../../models/restaurant/restaurant_info.dart';
import '../../models/location/location_info.dart';
import '../../services/restaurant/restaurant_service.dart';
import '../../services/ai_vision/vision_service.dart';
import '../../services/ar_display/ar_display_service.dart';
import '../../services/voice/voice_service.dart';
import '../../services/map/map_service.dart';

class RestaurantController extends ChangeNotifier {
  final RestaurantService _restaurantService;
  final VisionService _visionService;
  final ARDisplayService _arDisplayService;
  final VoiceService _voiceService;
  final MapService _mapService;

  RestaurantController({
    required RestaurantService restaurantService,
    required VisionService visionService,
    required ARDisplayService arDisplayService,
    required VoiceService voiceService,
    required MapService mapService,
  })  : _restaurantService = restaurantService,
        _visionService = visionService,
        _arDisplayService = arDisplayService,
        _voiceService = voiceService,
        _mapService = mapService;

  RestaurantInfo? _currentRestaurant;
  List<RestaurantInfo> _nearbyRestaurants = [];
  LocationInfo? _currentLocation;
  bool _isLoading = false;
  String? _errorMessage;

  RestaurantInfo? get currentRestaurant => _currentRestaurant;
  List<RestaurantInfo> get nearbyRestaurants => _nearbyRestaurants;
  LocationInfo? get currentLocation => _currentLocation;
  bool get isLoading => _isLoading;
  String? get errorMessage => _errorMessage;

  Future<void> recognizeRestaurantFromSignage() async {
    try {
      _setLoading(true);
      _clearError();

      // 获取相机图像识别餐厅招牌
      final imageData = await _getCameraImage();
      
      // 使用OCR识别餐厅名称
      final restaurantName = await _visionService.extractText(imageData);
      
      if (restaurantName.isNotEmpty) {
        // 根据名称查询餐厅信息
        final restaurant = await _restaurantService.getRestaurantByName(restaurantName);
        
        if (restaurant != null) {
          _currentRestaurant = restaurant;
          await _displayRestaurantInfo(restaurant);
          await _speakRestaurantInfo(restaurant);
        } else {
          await _voiceService.textToSpeech('未找到该餐厅的信息，请尝试其他方式搜索');
        }
      }

      notifyListeners();
    } catch (e) {
      _setError('识别餐厅失败: $e');
    } finally {
      _setLoading(false);
    }
  }

  Future<void> searchNearbyRestaurants() async {
    try {
      _setLoading(true);
      _clearError();

      // 获取当前位置
      _currentLocation = await _mapService.getCurrentLocation();
      
      // 搜索附近餐厅
      _nearbyRestaurants = await _restaurantService.searchNearbyRestaurants(
        _currentLocation!,
        radius: 1000,
      );

      if (_nearbyRestaurants.isNotEmpty) {
        await _displayNearbyRestaurants(_nearbyRestaurants);
        await _voiceService.textToSpeech(
          '找到${_nearbyRestaurants.length}家附近的餐厅，为您推荐评分最高的几家'
        );
      } else {
        await _voiceService.textToSpeech('附近没有找到餐厅信息');
      }

      notifyListeners();
    } catch (e) {
      _setError('搜索附近餐厅失败: $e');
    } finally {
      _setLoading(false);
    }
  }

  Future<void> handleVoiceCommand(String command) async {
    try {
      final lowerCommand = command.toLowerCase();
      
      if (lowerCommand.contains('这家店怎么样') || lowerCommand.contains('餐厅评价')) {
        await recognizeRestaurantFromSignage();
      } else if (lowerCommand.contains('附近餐厅') || lowerCommand.contains('推荐餐厅')) {
        await searchNearbyRestaurants();
      } else if (lowerCommand.contains('详细信息') && _currentRestaurant != null) {
        await _showDetailedInfo(_currentRestaurant!);
      } else if (lowerCommand.contains('评价') && _currentRestaurant != null) {
        await _showReviews(_currentRestaurant!);
      } else {
        await _voiceService.textToSpeech('请说"这家店怎么样"或"附近餐厅推荐"');
      }
    } catch (e) {
      _setError('处理语音命令失败: $e');
    }
  }

  Future<void> _displayRestaurantInfo(RestaurantInfo restaurant) async {
    // 显示餐厅名称和评分
    await _arDisplayService.showText(
      '${restaurant.name} ⭐${restaurant.ratingDisplay}',
      position: ARPosition.centerTop(),
      style: ARStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
    );

    // 显示人均价格和距离
    await _arDisplayService.showText(
      '${restaurant.priceDisplay} • ${restaurant.distanceDisplay}',
      position: ARPosition(x: 0.5, y: 0.25),
      style: ARStyle(fontSize: 14.0),
    );

    // 显示招牌菜
    if (restaurant.specialDishes.isNotEmpty) {
      await _arDisplayService.showText(
        '招牌菜: ${restaurant.specialDishes.take(2).join('、')}',
        position: ARPosition(x: 0.5, y: 0.35),
        style: ARStyle(fontSize: 12.0, textColor: const Color(0xFFFFEB3B)),
      );
    }

    // 显示评价数量
    await _arDisplayService.showText(
      '${restaurant.reviewCount}条评价',
      position: ARPosition(x: 0.5, y: 0.45),
      style: ARStyle(fontSize: 10.0, textColor: const Color(0xFFBDBDBD)),
    );
  }

  Future<void> _displayNearbyRestaurants(List<RestaurantInfo> restaurants) async {
    // 显示前3家推荐餐厅
    final topRestaurants = restaurants.take(3).toList();
    
    for (int i = 0; i < topRestaurants.length; i++) {
      final restaurant = topRestaurants[i];
      final yPosition = 0.2 + (i * 0.15);
      
      await _arDisplayService.showText(
        '${i + 1}. ${restaurant.name} ⭐${restaurant.ratingDisplay}',
        position: ARPosition(x: 0.5, y: yPosition),
        style: ARStyle(fontSize: 14.0),
      );
      
      await _arDisplayService.showText(
        '${restaurant.priceDisplay} • ${restaurant.distanceDisplay}',
        position: ARPosition(x: 0.5, y: yPosition + 0.05),
        style: ARStyle(fontSize: 10.0, textColor: const Color(0xFFBDBDBD)),
      );
    }
  }

  Future<void> _speakRestaurantInfo(RestaurantInfo restaurant) async {
    String spokenText = '${restaurant.name}，评分${restaurant.ratingDisplay}分';
    
    if (restaurant.reviewCount > 0) {
      spokenText += '，有${restaurant.reviewCount}条评价';
    }
    
    spokenText += '，人均消费${restaurant.averagePrice.toInt()}元';
    
    if (restaurant.distance < 1000) {
      spokenText += '，距离您${restaurant.distance.toInt()}米';
    } else {
      spokenText += '，距离您${(restaurant.distance / 1000).toStringAsFixed(1)}公里';
    }
    
    if (restaurant.specialDishes.isNotEmpty) {
      spokenText += '。招牌菜有${restaurant.specialDishes.take(2).join('和')}';
    }
    
    if (restaurant.rating >= 4.5) {
      spokenText += '。这家餐厅评价很高，推荐您尝试';
    } else if (restaurant.rating >= 4.0) {
      spokenText += '。这家餐厅评价不错';
    }

    await _voiceService.textToSpeech(spokenText);
  }

  Future<void> _showDetailedInfo(RestaurantInfo restaurant) async {
    await _arDisplayService.hideAllOverlays();
    
    // 显示详细信息
    await _arDisplayService.showText(
      '${restaurant.name} - 详细信息',
      position: ARPosition(x: 0.5, y: 0.15),
      style: ARStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
    );
    
    await _arDisplayService.showText(
      '地址: ${restaurant.address}',
      position: ARPosition(x: 0.5, y: 0.25),
      style: ARStyle(fontSize: 12.0),
    );
    
    await _arDisplayService.showText(
      '电话: ${restaurant.phone}',
      position: ARPosition(x: 0.5, y: 0.35),
      style: ARStyle(fontSize: 12.0),
    );
    
    await _arDisplayService.showText(
      '营业时间: ${restaurant.openingHours}',
      position: ARPosition(x: 0.5, y: 0.45),
      style: ARStyle(fontSize: 12.0),
    );
    
    await _arDisplayService.showText(
      '菜系: ${restaurant.cuisine}',
      position: ARPosition(x: 0.5, y: 0.55),
      style: ARStyle(fontSize: 12.0),
    );

    // 语音播报详细信息
    await _voiceService.textToSpeech(
      '${restaurant.name}位于${restaurant.address}，营业时间${restaurant.openingHours}，主营${restaurant.cuisine}'
    );
  }

  Future<void> _showReviews(RestaurantInfo restaurant) async {
    try {
      final reviews = await _restaurantService.getRestaurantReviews(restaurant.id);
      
      if (reviews.isNotEmpty) {
        await _arDisplayService.hideAllOverlays();
        
        // 显示评价标题
        await _arDisplayService.showText(
          '${restaurant.name} - 用户评价',
          position: ARPosition(x: 0.5, y: 0.15),
          style: ARStyle(fontSize: 16.0, fontWeight: FontWeight.bold),
        );
        
        // 显示前两条评价
        for (int i = 0; i < reviews.length && i < 2; i++) {
          final review = reviews[i];
          final yPosition = 0.3 + (i * 0.2);
          
          await _arDisplayService.showText(
            '${review.userName}: ⭐${review.rating}',
            position: ARPosition(x: 0.5, y: yPosition),
            style: ARStyle(fontSize: 12.0, fontWeight: FontWeight.bold),
          );
          
          await _arDisplayService.showText(
            review.comment.length > 50 
                ? '${review.comment.substring(0, 47)}...'
                : review.comment,
            position: ARPosition(x: 0.5, y: yPosition + 0.08),
            style: ARStyle(fontSize: 10.0),
          );
        }
        
        // 语音播报评价摘要
        final avgRating = reviews.map((r) => r.rating).reduce((a, b) => a + b) / reviews.length;
        await _voiceService.textToSpeech(
          '用户评价平均${avgRating.toStringAsFixed(1)}分，${reviews.first.comment}'
        );
      }
    } catch (e) {
      await _voiceService.textToSpeech('暂时无法获取评价信息');
    }
  }

  Future<Uint8List> _getCameraImage() async {
    // 模拟从相机获取图像数据
    await Future.delayed(const Duration(milliseconds: 500));
    return Uint8List.fromList([1, 2, 3, 4, 5]);
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