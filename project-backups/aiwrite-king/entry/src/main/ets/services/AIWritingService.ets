import { ApiConfig } from './ApiConfig';
import { RateLimitService } from './RateLimitService';
import { Context } from '@kit.AbilityKit';

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
 * 限流状态接口
 */
interface RateLimitStatus {
  canCall: boolean;
  remaining: number;
}

/**
 * 写作模板接口
 */
interface WritingTemplate {
  name: string;
  description: string;
  prompt: string;
}

/**
 * 写作模板枚举
 */
enum TemplateType {
  CREATIVE = '创意写作',
  BUSINESS = '商务邮件',
  ACADEMIC = '学术论文',
  STORY = '故事创作',
  POETRY = '诗歌创作',
  TECHNICAL = '技术文档'
}

/**
 * AI写作服务类
 * 提供AI写作相关的功能
 */
export class AIWritingService {
  private static instance: AIWritingService;
  private apiConfig: ApiConfig;
  private rateLimitService: RateLimitService | null = null;
  
  private constructor() {
    this.apiConfig = ApiConfig.getInstance();
  }
  
  static getInstance(): AIWritingService {
    if (!AIWritingService.instance) {
      AIWritingService.instance = new AIWritingService();
    }
    return AIWritingService.instance;
  }

  /**
   * 初始化限流服务
   * @param context 应用上下文
   */
  initRateLimit(context: Context): void {
    this.rateLimitService = RateLimitService.getInstance(context);
  }

  /**
   * 检查限流状态
   * @returns Promise<RateLimitStatus> 限流状态
   */
  async checkRateLimit(): Promise<RateLimitStatus> {
    if (!this.rateLimitService) {
      return { canCall: false, remaining: 0 };
    }
    
    const canCall = await this.rateLimitService.canCallAPI();
    const remaining = await this.rateLimitService.getRemainingCalls();
    
    return { canCall, remaining };
  }

  /**
   * 生成AI写作建议
   * @param prompt 用户输入的提示文本
   * @param template 写作模板类型
   * @returns Promise<string> AI生成的建议
   */
  async generateWritingSuggestion(prompt: string, template: string = ''): Promise<string> {
    // 检查限流
    const rateLimit = await this.checkRateLimit();
    if (!rateLimit.canCall) {
      throw new Error(`今日AI调用次数已达上限，剩余次数：${rateLimit.remaining}`);
    }

    try {
      // 调用真实API
      const result = await this.apiConfig.generateWritingSuggestion(prompt, template);
      
      // 记录API调用
      if (this.rateLimitService) {
        await this.rateLimitService.recordAPICall();
      }
      
      return result;
    } catch (error) {
      console.error('AI写作建议生成失败:', error);
      throw new Error('AI写作建议生成失败');
    }
  }

  /**
   * 优化文本
   * @param text 需要优化的文本
   * @returns Promise<string> 优化后的文本
   */
  async optimizeText(text: string): Promise<string> {
    // 检查限流
    const rateLimit = await this.checkRateLimit();
    if (!rateLimit.canCall) {
      throw new Error(`今日AI调用次数已达上限，剩余次数：${rateLimit.remaining}`);
    }

    try {
      // 调用真实API
      const result = await this.apiConfig.optimizeText(text);
      
      // 记录API调用
      if (this.rateLimitService) {
        await this.rateLimitService.recordAPICall();
      }
      
      return result;
    } catch (error) {
      console.error('文本优化失败:', error);
      throw new Error('文本优化失败');
    }
  }

  /**
   * 续写文本
   * @param text 现有文本
   * @returns Promise<string> 续写内容
   */
  async continueWriting(text: string): Promise<string> {
    // 检查限流
    const rateLimit = await this.checkRateLimit();
    if (!rateLimit.canCall) {
      throw new Error(`今日AI调用次数已达上限，剩余次数：${rateLimit.remaining}`);
    }

    try {
      // 调用真实API
      const result = await this.apiConfig.continueWriting(text);
      
      // 记录API调用
      if (this.rateLimitService) {
        await this.rateLimitService.recordAPICall();
      }
      
      return result;
    } catch (error) {
      console.error('文本续写失败:', error);
      throw new Error('文本续写失败');
    }
  }

  /**
   * 改写文本
   * @param text 原文本
   * @param style 改写风格
   * @returns Promise<string> 改写后的文本
   */
  async rewriteText(text: string, style: WritingStyle = WritingStyle.FORMAL): Promise<string> {
    // 检查限流
    const rateLimit = await this.checkRateLimit();
    if (!rateLimit.canCall) {
      throw new Error(`今日AI调用次数已达上限，剩余次数：${rateLimit.remaining}`);
    }

    try {
      // 调用真实API
      const result = await this.apiConfig.rewriteText(text, style);
      
      // 记录API调用
      if (this.rateLimitService) {
        await this.rateLimitService.recordAPICall();
      }
      
      return result;
    } catch (error) {
      console.error('文本改写失败:', error);
      throw new Error('文本改写失败');
    }
  }

  /**
   * 获取写作模板
   */
  getWritingTemplates(): WritingTemplate[] {
    const templates: WritingTemplate[] = [
      {
        name: TemplateType.CREATIVE,
        description: '自由创作，发挥想象力',
        prompt: '请基于以下内容进行创意写作：'
      },
      {
        name: TemplateType.BUSINESS,
        description: '专业的商务沟通',
        prompt: '请帮我写一封商务邮件：'
      },
      {
        name: TemplateType.ACADEMIC,
        description: '严谨的学术写作',
        prompt: '请帮我写一篇学术论文：'
      },
      {
        name: TemplateType.STORY,
        description: '引人入胜的故事',
        prompt: '请帮我创作一个故事：'
      },
      {
        name: TemplateType.POETRY,
        description: '优美的诗歌作品',
        prompt: '请帮我创作一首诗歌：'
      },
      {
        name: TemplateType.TECHNICAL,
        description: '清晰的技术说明',
        prompt: '请帮我写一份技术文档：'
      }
    ];
    return templates;
  }

}
