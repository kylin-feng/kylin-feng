class ApiConfig {
  // 百度AI API
  static const String baiduAiApiKey = 'YOUR_BAIDU_AI_API_KEY';
  static const String baiduAiSecretKey = 'YOUR_BAIDU_AI_SECRET_KEY';
  
  // 高德地图API
  static const String gaodeMapApiKey = 'YOUR_GAODE_MAP_API_KEY';
  
  // 大众点评API
  static const String dianpingApiKey = 'YOUR_DIANPING_API_KEY';
  static const String dianpingSecretKey = 'YOUR_DIANPING_SECRET_KEY';
  
  // 京东API
  static const String jdApiKey = 'YOUR_JD_API_KEY';
  static const String jdSecretKey = 'YOUR_JD_SECRET_KEY';
  
  // 淘宝API
  static const String taobaoApiKey = 'YOUR_TAOBAO_API_KEY';
  static const String taobaoSecretKey = 'YOUR_TAOBAO_SECRET_KEY';
  
  // 灵珠平台配置
  static const String lingzhuApiKey = 'YOUR_LINGZHU_API_KEY';
  static const String lingzhuWorkflowId = 'YOUR_WORKFLOW_ID';
  
  // 获取完整的API密钥配置
  static Map<String, String> getApiKeys() {
    return {
      'baidu_ai_key': baiduAiApiKey,
      'baidu_ai_secret': baiduAiSecretKey,
      'gaode_map_key': gaodeMapApiKey,
      'dianping_key': dianpingApiKey,
      'dianping_secret': dianpingSecretKey,
      'jd_key': jdApiKey,
      'jd_secret': jdSecretKey,
      'taobao_key': taobaoApiKey,
      'taobao_secret': taobaoSecretKey,
      'lingzhu_key': lingzhuApiKey,
      'lingzhu_workflow': lingzhuWorkflowId,
    };
  }
}