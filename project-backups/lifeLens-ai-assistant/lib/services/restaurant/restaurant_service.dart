import '../../models/restaurant/restaurant_info.dart';
import '../../models/location/location_info.dart';

abstract class RestaurantService {
  Future<RestaurantInfo?> getRestaurantByName(String name);
  Future<List<RestaurantInfo>> searchNearbyRestaurants(LocationInfo location, {double radius = 1000});
  Future<List<RestaurantInfo>> searchRestaurants(String query, LocationInfo location);
  Future<List<Review>> getRestaurantReviews(String restaurantId);
}

class DianpingRestaurantService implements RestaurantService {
  final String apiKey;
  final String secretKey;
  
  DianpingRestaurantService({
    required this.apiKey,
    required this.secretKey,
  });

  @override
  Future<RestaurantInfo?> getRestaurantByName(String name) async {
    try {
      // 调用大众点评API根据名称搜索餐厅
      await Future.delayed(Duration(seconds: 1));
      
      return RestaurantInfo(
        id: '12345',
        name: name,
        rating: 4.5,
        reviewCount: 1288,
        address: '北京市朝阳区三里屯路19号',
        phone: '010-12345678',
        averagePrice: 120.0,
        cuisine: '川菜',
        specialDishes: ['麻婆豆腐', '宫保鸡丁', '水煮鱼'],
        openingHours: '11:00-22:00',
        distance: 500.0,
        tags: ['川菜', '正宗', '环境好', '服务佳'],
        reviews: [],
      );
    } catch (e) {
      throw RestaurantServiceException('获取餐厅信息失败: $e');
    }
  }

  @override
  Future<List<RestaurantInfo>> searchNearbyRestaurants(
    LocationInfo location, {
    double radius = 1000,
  }) async {
    try {
      // 搜索附近餐厅
      await Future.delayed(Duration(seconds: 2));
      
      return [
        RestaurantInfo(
          id: '1',
          name: '蜀大侠火锅',
          rating: 4.6,
          reviewCount: 2566,
          address: '朝阳区三里屯路12号',
          phone: '010-87654321',
          averagePrice: 89.0,
          cuisine: '火锅',
          specialDishes: ['毛肚', '鸭血', '土豆片'],
          openingHours: '17:00-02:00',
          distance: 200.0,
          tags: ['火锅', '川味', '夜宵'],
          reviews: [],
        ),
        RestaurantInfo(
          id: '2',
          name: '胜博殿',
          rating: 4.3,
          reviewCount: 1899,
          address: '朝阳区工体北路8号',
          phone: '010-65432109',
          averagePrice: 68.0,
          cuisine: '日料',
          specialDishes: ['炸猪排', '照烧鸡腿', '天妇罗'],
          openingHours: '11:30-21:30',
          distance: 350.0,
          tags: ['日料', '炸物', '快餐'],
          reviews: [],
        ),
      ];
    } catch (e) {
      throw RestaurantServiceException('搜索附近餐厅失败: $e');
    }
  }

  @override
  Future<List<RestaurantInfo>> searchRestaurants(
    String query,
    LocationInfo location,
  ) async {
    try {
      // 根据关键词搜索餐厅
      await Future.delayed(Duration(seconds: 1));
      return [];
    } catch (e) {
      throw RestaurantServiceException('搜索餐厅失败: $e');
    }
  }

  @override
  Future<List<Review>> getRestaurantReviews(String restaurantId) async {
    try {
      // 获取餐厅评价
      await Future.delayed(Duration(seconds: 1));
      
      return [
        Review(
          userId: 'user1',
          userName: '美食家小王',
          rating: 4.5,
          comment: '味道很棒，环境也不错，服务态度很好！',
          date: DateTime.now().subtract(Duration(days: 3)),
        ),
        Review(
          userId: 'user2',
          userName: '吃货小李',
          rating: 4.0,
          comment: '菜品新鲜，分量足，性价比高。',
          date: DateTime.now().subtract(Duration(days: 7)),
        ),
      ];
    } catch (e) {
      throw RestaurantServiceException('获取餐厅评价失败: $e');
    }
  }
}

class RestaurantServiceException implements Exception {
  final String message;
  RestaurantServiceException(this.message);
  
  @override
  String toString() => 'RestaurantServiceException: $message';
}