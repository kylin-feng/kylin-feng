import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'restaurant_controller.dart';
import '../../models/restaurant/restaurant_info.dart';

class RestaurantPage extends StatefulWidget {
  const RestaurantPage({Key? key}) : super(key: key);

  @override
  State<RestaurantPage> createState() => _RestaurantPageState();
}

class _RestaurantPageState extends State<RestaurantPage> {
  late RestaurantController _controller;

  @override
  void initState() {
    super.initState();
    // 初始化餐厅控制器
    _controller = RestaurantController(
      restaurantService: context.read(),
      visionService: context.read(),
      arDisplayService: context.read(),
      voiceService: context.read(),
      mapService: context.read(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider.value(
      value: _controller,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Consumer<RestaurantController>(
          builder: (context, controller, child) {
            return Stack(
              children: [
                // 相机预览区域
                const CameraPreviewWidget(),
                
                // AR叠加信息
                if (controller.currentRestaurant != null)
                  RestaurantAROverlayWidget(
                    restaurant: controller.currentRestaurant!,
                  ),
                
                // 附近餐厅列表
                if (controller.nearbyRestaurants.isNotEmpty)
                  NearbyRestaurantsWidget(
                    restaurants: controller.nearbyRestaurants,
                    onRestaurantTap: (restaurant) {
                      _showRestaurantDetails(restaurant);
                    },
                  ),
                
                // 加载指示器
                if (controller.isLoading)
                  const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(color: Colors.orange),
                        SizedBox(height: 16),
                        Text(
                          '正在搜索餐厅...',
                          style: TextStyle(color: Colors.white),
                        ),
                      ],
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
                        style: const TextStyle(color: Colors.white),
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
                        onPressed: controller.isLoading 
                            ? null 
                            : controller.recognizeRestaurantFromSignage,
                        backgroundColor: Colors.orange,
                        child: const Icon(Icons.restaurant_menu),
                      ),
                      FloatingActionButton(
                        onPressed: controller.isLoading 
                            ? null 
                            : controller.searchNearbyRestaurants,
                        backgroundColor: Colors.deepOrange,
                        child: const Icon(Icons.location_on),
                      ),
                    ],
                  ),
                ),
                
                // 语音指示
                const Positioned(
                  top: 50,
                  left: 0,
                  right: 0,
                  child: RestaurantVoiceIndicator(),
                ),
              ],
            );
          },
        ),
      ),
    );
  }

  void _showRestaurantDetails(RestaurantInfo restaurant) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.black.withOpacity(0.9),
      isScrollControlled: true,
      builder: (context) => RestaurantDetailSheet(restaurant: restaurant),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
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
              Icons.restaurant,
              size: 100,
              color: Colors.grey,
            ),
            SizedBox(height: 20),
            Text(
              '餐厅识别预览',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 18,
              ),
            ),
            SizedBox(height: 10),
            Text(
              '对准餐厅招牌或说"这家店怎么样"',
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

class RestaurantAROverlayWidget extends StatelessWidget {
  final RestaurantInfo restaurant;
  
  const RestaurantAROverlayWidget({
    Key? key,
    required this.restaurant,
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
          color: Colors.black.withOpacity(0.8),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.orange, width: 2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // 餐厅名称和评分
            Row(
              children: [
                Expanded(
                  child: Text(
                    restaurant.name,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: _getRatingColor(restaurant.rating),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.star,
                        color: Colors.white,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        restaurant.ratingDisplay,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            
            // 基本信息
            Row(
              children: [
                Icon(
                  Icons.restaurant_menu,
                  color: Colors.orange[300],
                  size: 16,
                ),
                const SizedBox(width: 4),
                Text(
                  restaurant.cuisine,
                  style: TextStyle(
                    color: Colors.orange[300],
                    fontSize: 14,
                  ),
                ),
                const SizedBox(width: 16),
                Icon(
                  Icons.attach_money,
                  color: Colors.green[300],
                  size: 16,
                ),
                Text(
                  restaurant.priceDisplay,
                  style: TextStyle(
                    color: Colors.green[300],
                    fontSize: 14,
                  ),
                ),
                const Spacer(),
                Icon(
                  Icons.location_on,
                  color: Colors.blue[300],
                  size: 16,
                ),
                Text(
                  restaurant.distanceDisplay,
                  style: TextStyle(
                    color: Colors.blue[300],
                    fontSize: 14,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            
            // 招牌菜
            if (restaurant.specialDishes.isNotEmpty) ...[
              const Text(
                '招牌菜:',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 4),
              Wrap(
                spacing: 8,
                children: restaurant.specialDishes.take(3).map((dish) {
                  return Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 4,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.orange.withOpacity(0.3),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      dish,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                      ),
                    ),
                  );
                }).toList(),
              ),
              const SizedBox(height: 8),
            ],
            
            // 评价数量
            Text(
              '${restaurant.reviewCount}条评价',
              style: const TextStyle(
                color: Colors.grey,
                fontSize: 12,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getRatingColor(double rating) {
    if (rating >= 4.5) return Colors.green;
    if (rating >= 4.0) return Colors.orange;
    if (rating >= 3.5) return Colors.amber;
    return Colors.red;
  }
}

class NearbyRestaurantsWidget extends StatelessWidget {
  final List<RestaurantInfo> restaurants;
  final Function(RestaurantInfo) onRestaurantTap;
  
  const NearbyRestaurantsWidget({
    Key? key,
    required this.restaurants,
    required this.onRestaurantTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 120,
      left: 0,
      right: 0,
      child: Container(
        height: 200,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 20),
              child: Text(
                '附近推荐餐厅',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(height: 8),
            Expanded(
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 20),
                itemCount: restaurants.length,
                itemBuilder: (context, index) {
                  final restaurant = restaurants[index];
                  return GestureDetector(
                    onTap: () => onRestaurantTap(restaurant),
                    child: Container(
                      width: 160,
                      margin: const EdgeInsets.only(right: 12),
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.8),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.grey[600]!),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            restaurant.name,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              fontWeight: FontWeight.bold,
                            ),
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                          const SizedBox(height: 4),
                          Row(
                            children: [
                              const Icon(
                                Icons.star,
                                color: Colors.amber,
                                size: 14,
                              ),
                              const SizedBox(width: 2),
                              Text(
                                restaurant.ratingDisplay,
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                ),
                              ),
                              const Spacer(),
                              Text(
                                restaurant.priceDisplay,
                                style: const TextStyle(
                                  color: Colors.green,
                                  fontSize: 12,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text(
                            restaurant.distanceDisplay,
                            style: const TextStyle(
                              color: Colors.blue,
                              fontSize: 11,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            restaurant.cuisine,
                            style: const TextStyle(
                              color: Colors.grey,
                              fontSize: 11,
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class RestaurantVoiceIndicator extends StatefulWidget {
  const RestaurantVoiceIndicator({Key? key}) : super(key: key);

  @override
  State<RestaurantVoiceIndicator> createState() => _RestaurantVoiceIndicatorState();
}

class _RestaurantVoiceIndicatorState extends State<RestaurantVoiceIndicator>
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
                  color: Colors.orange.withOpacity(_animation.value * 0.7),
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
                      '说"这家店怎么样"或"附近餐厅"',
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

class RestaurantDetailSheet extends StatelessWidget {
  final RestaurantInfo restaurant;
  
  const RestaurantDetailSheet({
    Key? key,
    required this.restaurant,
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
                  restaurant.name,
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
          
          _buildDetailRow(Icons.star, '评分', '${restaurant.ratingDisplay} (${restaurant.reviewCount}条评价)'),
          _buildDetailRow(Icons.attach_money, '人均消费', restaurant.priceDisplay),
          _buildDetailRow(Icons.location_on, '距离', restaurant.distanceDisplay),
          _buildDetailRow(Icons.restaurant_menu, '菜系', restaurant.cuisine),
          _buildDetailRow(Icons.access_time, '营业时间', restaurant.openingHours),
          _buildDetailRow(Icons.phone, '电话', restaurant.phone),
          _buildDetailRow(Icons.place, '地址', restaurant.address),
          
          if (restaurant.specialDishes.isNotEmpty) ...[
            const SizedBox(height: 16),
            const Text(
              '招牌菜:',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: restaurant.specialDishes.map((dish) {
                return Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.orange.withOpacity(0.3),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Text(
                    dish,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                    ),
                  ),
                );
              }).toList(),
            ),
          ],
          
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget _buildDetailRow(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: Colors.orange, size: 20),
          const SizedBox(width: 12),
          Text(
            '$label: ',
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 14,
              fontWeight: FontWeight.bold,
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
              ),
            ),
          ),
        ],
      ),
    );
  }
}