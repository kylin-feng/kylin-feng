import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'core/api_config.dart';
import 'services/ai_vision/vision_service.dart';
import 'services/ar_display/ar_display_service.dart';
import 'services/voice/voice_service.dart';
import 'services/restaurant/restaurant_service.dart';
import 'services/commerce/commerce_service.dart';
import 'services/map/map_service.dart';
import 'features/object_recognition/object_recognition_controller.dart';
import 'features/object_recognition/object_recognition_page.dart';
import 'features/restaurant_recommendation/restaurant_page.dart';
import 'features/shopping_assistant/shopping_page.dart';

void main() {
  runApp(const LifeLensApp());
}

class LifeLensApp extends StatelessWidget {
  const LifeLensApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        // 服务提供者
        Provider<VisionService>(
          create: (_) => BaiduVisionService(
            apiKey: ApiConfig.baiduAiApiKey,
            secretKey: ApiConfig.baiduAiSecretKey,
          ),
        ),
        Provider<ARDisplayService>(
          create: (_) => RokidARDisplayService(),
        ),
        Provider<VoiceService>(
          create: (_) => SystemVoiceService(),
        ),
        Provider<RestaurantService>(
          create: (_) => DianpingRestaurantService(
            apiKey: ApiConfig.dianpingApiKey,
            secretKey: ApiConfig.dianpingSecretKey,
          ),
        ),
        Provider<CommerceService>(
          create: (_) => MultiPlatformCommerceService(
            platforms: [
              JDPlatformService(
                apiKey: ApiConfig.jdApiKey,
                secretKey: ApiConfig.jdSecretKey,
              ),
              TaobaoPlatformService(
                apiKey: ApiConfig.taobaoApiKey,
                secretKey: ApiConfig.taobaoSecretKey,
              ),
            ],
          ),
        ),
        Provider<MapService>(
          create: (_) => GaodeMapService(
            apiKey: ApiConfig.gaodeMapApiKey,
          ),
        ),
        
        // 控制器提供者
        ChangeNotifierProvider<ObjectRecognitionController>(
          create: (context) => ObjectRecognitionController(
            visionService: context.read<VisionService>(),
            arDisplayService: context.read<ARDisplayService>(),
            voiceService: context.read<VoiceService>(),
          ),
        ),
      ],
      child: MaterialApp(
        title: 'LifeLens AI Assistant',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          fontFamily: 'PingFang',
          visualDensity: VisualDensity.adaptivePlatformDensity,
        ),
        home: const MainPage(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}

class MainPage extends StatefulWidget {
  const MainPage({Key? key}) : super(key: key);

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  int _currentIndex = 0;
  
  final List<Widget> _pages = [
    const ObjectRecognitionPage(),
    const RestaurantPage(),
    const ShoppingPage(),
    const NavigationPage(),
    const VoiceMemoPage(),
  ];

  final List<String> _titles = [
    '智能识物',
    '餐饮推荐',
    '购物助手',
    '导航助手',
    '语音备忘',
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _pages,
      ),
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        backgroundColor: Colors.black,
        selectedItemColor: Colors.blue,
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.camera_alt),
            label: '识物',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.restaurant),
            label: '餐厅',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.shopping_cart),
            label: '购物',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.navigation),
            label: '导航',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.mic),
            label: '备忘',
          ),
        ],
      ),
    );
  }
}

// RestaurantPage 现在在单独的文件中实现

// ShoppingPage 现在在单独的文件中实现

class NavigationPage extends StatelessWidget {
  const NavigationPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.navigation,
              size: 100,
              color: Colors.grey,
            ),
            SizedBox(height: 20),
            Text(
              '导航问路助手',
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
              ),
            ),
            SizedBox(height: 10),
            Text(
              '即将推出...',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 16,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class VoiceMemoPage extends StatelessWidget {
  const VoiceMemoPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      backgroundColor: Colors.black,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.mic,
              size: 100,
              color: Colors.grey,
            ),
            SizedBox(height: 20),
            Text(
              '语音备忘录',
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
              ),
            ),
            SizedBox(height: 10),
            Text(
              '即将推出...',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 16,
              ),
            ),
          ],
        ),
      ),
    );
  }
}