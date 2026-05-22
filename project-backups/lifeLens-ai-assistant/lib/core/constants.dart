class AppConstants {
  static const String appName = 'LifeLens';
  static const String appVersion = '1.0.0';
  
  // API 配置
  static const String baiduAiApiUrl = 'https://aip.baidubce.com';
  static const String gaodeMapApiUrl = 'https://restapi.amap.com';
  static const String dianpingApiUrl = 'https://api.dianping.com';
  
  // 灵珠平台配置
  static const String lingzhuPlatformUrl = 'https://lingzhu-platform.com';
  
  // 超时配置
  static const int connectionTimeout = 30000;
  static const int receiveTimeout = 30000;
  
  // 缓存配置
  static const String cacheDir = 'lifeLens_cache';
  static const int maxCacheSize = 100 * 1024 * 1024; // 100MB
  
  // AR显示配置
  static const double arTextSize = 16.0;
  static const int arDisplayDuration = 5000; // 5秒
  
  // 语音配置
  static const String voiceLanguage = 'zh-CN';
  static const double speechRate = 1.0;
  
  // 识别置信度阈值
  static const double recognitionThreshold = 0.7;
  
  // 功能标识
  static const String objectRecognition = 'object_recognition';
  static const String restaurantRecommendation = 'restaurant_recommendation';
  static const String shoppingAssistant = 'shopping_assistant';
  static const String navigationHelper = 'navigation_helper';
  static const String voiceMemo = 'voice_memo';
}