class Config {
  // 在生产环境中，建议使用以下方式之一来管理API密钥：
  // 1. 环境变量
  // 2. 安全的配置文件（不提交到版本控制）
  // 3. 密钥管理服务
  
  static const String siliconFlowApiKey = 'your-siliconflow-api-key';
  static const String siliconFlowBaseUrl = 'https://api.siliconflow.cn/v1/chat/completions';
  static const String multiModalModel = 'Qwen/Qwen3-VL-235B-A22B-Instruct';
  
  // API调用超时设置
  static const Duration apiTimeout = Duration(seconds: 60);
  
  // 图片处理设置
  static const int maxImageSize = 5 * 1024 * 1024; // 5MB
  static const List<String> supportedImageFormats = ['jpg', 'jpeg', 'png'];
}