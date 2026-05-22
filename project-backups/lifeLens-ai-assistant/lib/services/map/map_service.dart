import '../../models/location/location_info.dart';

abstract class MapService {
  Future<LocationInfo> getCurrentLocation();
  Future<NavigationInfo> getRoute(LocationInfo origin, LocationInfo destination, {String mode = 'walking'});
  Future<List<POI>> searchNearbyPOI(LocationInfo location, String type, {double radius = 1000});
  Future<List<POI>> searchPOI(String query, LocationInfo location);
  Future<String> reverseGeocode(double latitude, double longitude);
  Future<LocationInfo?> geocode(String address);
}

class GaodeMapService implements MapService {
  final String apiKey;
  
  GaodeMapService({required this.apiKey});

  @override
  Future<LocationInfo> getCurrentLocation() async {
    try {
      // 调用高德地图API或设备GPS获取当前位置
      await Future.delayed(Duration(seconds: 1));
      
      return LocationInfo(
        latitude: 39.904200,
        longitude: 116.407396,
        address: '北京市东城区天安门广场',
        district: '东城区',
        city: '北京市',
        province: '北京市',
        country: '中国',
        timestamp: DateTime.now(),
      );
    } catch (e) {
      throw MapServiceException('获取当前位置失败: $e');
    }
  }

  @override
  Future<NavigationInfo> getRoute(
    LocationInfo origin,
    LocationInfo destination, {
    String mode = 'walking',
  }) async {
    try {
      // 调用高德地图路径规划API
      await Future.delayed(Duration(seconds: 2));
      
      final steps = [
        RouteStep(
          instruction: '从起点出发，向北步行',
          distance: 200.0,
          duration: 3,
          direction: '北',
          startLocation: origin,
          endLocation: LocationInfo(
            latitude: origin.latitude + 0.001,
            longitude: origin.longitude,
            address: '中间点1',
            timestamp: DateTime.now(),
          ),
        ),
        RouteStep(
          instruction: '右转，继续向东步行',
          distance: 300.0,
          duration: 4,
          direction: '东',
          startLocation: LocationInfo(
            latitude: origin.latitude + 0.001,
            longitude: origin.longitude,
            address: '中间点1',
            timestamp: DateTime.now(),
          ),
          endLocation: destination,
        ),
      ];
      
      return NavigationInfo(
        origin: origin,
        destination: destination,
        distance: 500.0,
        duration: 7,
        mode: mode,
        steps: steps,
        overview: '步行约7分钟，总距离500米',
      );
    } catch (e) {
      throw MapServiceException('获取路线失败: $e');
    }
  }

  @override
  Future<List<POI>> searchNearbyPOI(
    LocationInfo location,
    String type, {
    double radius = 1000,
  }) async {
    try {
      // 搜索附近兴趣点
      await Future.delayed(Duration(seconds: 1));
      
      switch (type) {
        case '地铁站':
          return [
            POI(
              id: 'subway_1',
              name: '天安门东地铁站',
              type: '地铁站',
              location: LocationInfo(
                latitude: 39.903738,
                longitude: 116.410638,
                address: '1号线天安门东站',
                timestamp: DateTime.now(),
              ),
              distance: 300.0,
              isOpen: true,
              openingHours: '05:00-23:30',
            ),
          ];
        case '餐厅':
          return [
            POI(
              id: 'restaurant_1',
              name: '全聚德烤鸭店',
              type: '餐厅',
              location: LocationInfo(
                latitude: 39.905000,
                longitude: 116.408000,
                address: '前门大街30号',
                timestamp: DateTime.now(),
              ),
              distance: 500.0,
              phone: '010-67023062',
              rating: 4.2,
              isOpen: true,
              openingHours: '11:00-21:00',
            ),
          ];
        default:
          return [];
      }
    } catch (e) {
      throw MapServiceException('搜索附近兴趣点失败: $e');
    }
  }

  @override
  Future<List<POI>> searchPOI(String query, LocationInfo location) async {
    try {
      // 根据关键词搜索兴趣点
      await Future.delayed(Duration(seconds: 1));
      
      return [
        POI(
          id: 'poi_1',
          name: query,
          type: '通用',
          location: LocationInfo(
            latitude: location.latitude + 0.001,
            longitude: location.longitude + 0.001,
            address: '搜索结果地址',
            timestamp: DateTime.now(),
          ),
          distance: 200.0,
        ),
      ];
    } catch (e) {
      throw MapServiceException('搜索兴趣点失败: $e');
    }
  }

  @override
  Future<String> reverseGeocode(double latitude, double longitude) async {
    try {
      // 逆地理编码
      await Future.delayed(Duration(milliseconds: 500));
      return '北京市东城区天安门广场';
    } catch (e) {
      throw MapServiceException('逆地理编码失败: $e');
    }
  }

  @override
  Future<LocationInfo?> geocode(String address) async {
    try {
      // 地理编码
      await Future.delayed(Duration(milliseconds: 500));
      
      return LocationInfo(
        latitude: 39.904200,
        longitude: 116.407396,
        address: address,
        timestamp: DateTime.now(),
      );
    } catch (e) {
      throw MapServiceException('地理编码失败: $e');
    }
  }
}

class MapServiceException implements Exception {
  final String message;
  MapServiceException(this.message);
  
  @override
  String toString() => 'MapServiceException: $message';
}