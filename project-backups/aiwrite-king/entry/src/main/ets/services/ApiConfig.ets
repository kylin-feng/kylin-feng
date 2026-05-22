import http from '@ohos.net.http';

/**
 * API角色枚举
 */
enum APIRole {
  SYSTEM = 'system',
  USER = 'user',
  ASSISTANT = 'assistant'
}

/**
 * 写作风格枚举
 */
enum WritingStyle {
  FORMAL = 'formal',
  CASUAL = 'casual',
  ACADEMIC = 'academic',
  CREATIVE = 'creative'
}

/**
 * API消息接口
 */
interface ApiMessage {
  role: APIRole;
  content: string;
}

/**
 * API请求数据接口
 */
interface ApiRequest {
  model: string;
  messages: ApiMessage[];
  temperature: number;
  max_tokens: number;
}

/**
 * API响应选择接口
 */
interface ApiChoice {
  message: ApiMessage;
}

/**
 * API响应接口
 */
interface ApiResponse {
  choices: ApiChoice[];
}

/**
 * API配置服务
 * 管理大模型API的配置和调用
 */
export class ApiConfig {
  private static instance: ApiConfig;
  private readonly API_BASE_URL: string = 'https://api.siliconflow.cn/v1/chat/completions';
  private readonly API_KEY: string = 'your-siliconflow-api-key';
  private readonly MODEL: string = 'Qwen/QwQ-32B';
  
  private constructor() {}
  
  static getInstance(): ApiConfig {
    if (!ApiConfig.instance) {
      ApiConfig.instance = new ApiConfig();
    }
    return ApiConfig.instance;
  }

  /**
   * 调用大模型API
   * @param prompt 用户输入的提示
   * @param template 写作模板类型
   * @returns Promise<string> AI生成的内容
   */
  async callAI(prompt: string, template: string = ''): Promise<string> {
    try {
      const systemPrompt = this.getSystemPrompt(template);
      
      const requestData: ApiRequest = {
        model: this.MODEL,
        messages: [
          {
            role: APIRole.SYSTEM,
            content: systemPrompt
          },
          {
            role: APIRole.USER,
            content: prompt
          }
        ],
        temperature: 0.7,
        max_tokens: 1000
      };

      const httpRequest = http.createHttp();
      
      try {
        const response = await httpRequest.request(this.API_BASE_URL, {
          method: http.RequestMethod.POST,
          header: {
            'Authorization': `Bearer ${this.API_KEY}`,
            'Content-Type': 'application/json'
          },
          extraData: JSON.stringify(requestData)
        }).catch((error: Error) => {
          console.error('HTTP请求失败:', error);
          throw new Error('网络请求失败，请检查网络连接');
        });

        if (response.responseCode !== 200) {
          throw new Error(`API调用失败: ${response.responseCode}`);
        }

        const result: ApiResponse = JSON.parse(response.result as string) as ApiResponse;
        
        if (result.choices && result.choices.length > 0) {
          return result.choices[0].message.content;
        } else {
          throw new Error('API返回格式错误');
        }
      } finally {
        httpRequest.destroy();
      }
    } catch (error) {
      console.error('API调用失败:', error);
      if (error instanceof Error) {
        throw new Error(error.message);
      } else {
        throw new Error('API调用失败');
      }
    }
  }

  /**
   * 获取系统提示词
   * @param template 模板类型
   * @returns string 系统提示词
   */
  private getSystemPrompt(template: string): string {
    const basePrompt: string = '你是一个专业的AI写作助手，擅长各种类型的写作。请根据用户的需求，提供高质量的写作建议和内容。';
    
    const templatePrompts: Record<string, string> = {
      '创意写作': '请以创意和想象力为主，提供富有创意的写作内容。',
      '商务邮件': '请以专业、正式的商务沟通风格，提供商务邮件写作建议。',
      '学术论文': '请以严谨、学术的风格，提供学术论文写作建议。',
      '故事创作': '请以引人入胜的故事叙述风格，提供故事创作建议。',
      '诗歌创作': '请以优美的诗歌语言，提供诗歌创作建议。',
      '技术文档': '请以清晰、准确的技术说明风格，提供技术文档写作建议。'
    };

    const templatePrompt: string = templatePrompts[template] || '';
    return `${basePrompt} ${templatePrompt}`;
  }

  /**
   * 生成写作建议
   * @param prompt 用户输入
   * @param template 模板类型
   * @returns Promise<string> 写作建议
   */
  async generateWritingSuggestion(prompt: string, template: string = ''): Promise<string> {
    const enhancedPrompt = `请基于以下内容提供写作建议：\n\n${prompt}\n\n请提供具体的写作建议，包括结构建议、内容要点和具体的写作内容。`;
    return await this.callAI(enhancedPrompt, template);
  }

  /**
   * 优化文本
   * @param text 原文本
   * @returns Promise<string> 优化后的文本
   */
  async optimizeText(text: string): Promise<string> {
    const prompt = `请优化以下文本，使其更加流畅、准确和富有表现力：\n\n${text}\n\n请提供优化后的版本，并说明优化的要点。`;
    return await this.callAI(prompt, '');
  }

  /**
   * 续写文本
   * @param text 现有文本
   * @returns Promise<string> 续写内容
   */
  async continueWriting(text: string): Promise<string> {
    const prompt = `请续写以下内容，保持逻辑连贯和风格一致：\n\n${text}\n\n请提供续写内容。`;
    return await this.callAI(prompt, '');
  }

  /**
   * 改写文本
   * @param text 原文本
   * @param style 改写风格
   * @returns Promise<string> 改写后的文本
   */
  async rewriteText(text: string, style: WritingStyle = WritingStyle.FORMAL): Promise<string> {
    const styleMap: Record<WritingStyle, string> = {
      [WritingStyle.FORMAL]: '正式',
      [WritingStyle.CASUAL]: '随意',
      [WritingStyle.ACADEMIC]: '学术',
      [WritingStyle.CREATIVE]: '创意'
    };
    
    const prompt = `请将以下文本改写为${styleMap[style] || '正式'}风格：\n\n${text}\n\n请提供改写后的版本。`;
    return await this.callAI(prompt, '');
  }
}
