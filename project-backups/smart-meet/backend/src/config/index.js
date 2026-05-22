import dotenv from 'dotenv';

// 加载环境变量
dotenv.config();

export const config = {
  port: process.env.PORT || 5000,
  nodeEnv: process.env.NODE_ENV || 'development',
  
  // API配置
  apiBaseUrl: process.env.API_BASE_URL || 'http://localhost:5000/api',
  
  // 大模型配置
  qianwen: {
    apiKey: process.env.QIANWEN_API_KEY,
    apiUrl: process.env.QIANWEN_API_URL || 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
  },
  
  deepseek: {
    apiKey: process.env.DEEPSEEK_API_KEY,
    apiUrl: process.env.DEEPSEEK_API_URL || 'https://api.deepseek.com/v1/chat/completions'
  },
  
  // OpenAI配置
  openai: {
    apiKey: process.env.OPENAI_API_KEY,
    apiUrl: process.env.OPENAI_API_URL || 'https://api.openai.com/v1'
  },
  
  // Azure语音服务配置
  azure: {
    speechKey: process.env.AZURE_SPEECH_KEY,
    speechRegion: process.env.AZURE_SPEECH_REGION || 'eastus'
  },
  
  // 阿里云语音服务配置
  alibaba: {
    speechAppKey: process.env.ALIBABA_SPEECH_APP_KEY
  },
  
  // 语音转录服务配置
  speech: {
    provider: process.env.SPEECH_PROVIDER || 'alibaba', // openai, azure, alibaba
    apiKey: process.env.SPEECH_API_KEY,
    apiUrl: process.env.SPEECH_API_URL || 'https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/asr'
  },
  
  // CORS配置
  cors: {
    origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
    credentials: true
  },
  
  // 文件上传配置
  upload: {
    maxFileSize: process.env.MAX_FILE_SIZE || '50MB',
    uploadDir: process.env.UPLOAD_DIR || './uploads'
  }
};

export default config;