import { preferences } from '@kit.ArkData';
import { Context } from '@kit.AbilityKit';
import deviceInfo from '@ohos.deviceInfo';

/**
 * 限流状态枚举
 */
enum RateLimitStatus {
  ALLOWED = 'allowed',
  LIMITED = 'limited',
  EXCEEDED = 'exceeded'
}

/**
 * 时间单位枚举
 */
enum TimeUnit {
  DAY = 'day',
  HOUR = 'hour',
  MINUTE = 'minute'
}

/**
 * 限流服务
 * 管理用户API调用次数限制
 */
export class RateLimitService {
  private static instance: RateLimitService;
  private context: Context;
  private readonly DAILY_LIMIT: number = 10; // 每天最多10次
  private readonly LIMIT_TYPE: TimeUnit = TimeUnit.DAY;
  private deviceId: string = '';
  
  private constructor(context: Context) {
    this.context = context;
    this.initDeviceId();
    // 启动时清除可能的旧数据
    this.clearAllLimitData();
  }
  
  static getInstance(context: Context): RateLimitService {
    if (!RateLimitService.instance) {
      RateLimitService.instance = new RateLimitService(context);
    }
    return RateLimitService.instance;
  }

  /**
   * 初始化设备ID
   */
  private async initDeviceId(): Promise<void> {
    try {
      // 尝试获取UDID
      this.deviceId = deviceInfo.udid;
      console.log('设备UDID:', this.deviceId);
    } catch (error) {
      console.error('获取设备UDID失败:', error);
    }
    
    // 如果UDID获取失败或为空，使用更稳定的备用方案
    if (!this.deviceId || this.deviceId === '') {
      // 使用应用包名 + 时间戳作为设备ID
      this.deviceId = 'app_device_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      console.log('使用备用设备ID:', this.deviceId);
    }
    
    console.log('最终设备ID:', this.deviceId);
  }

  /**
   * 获取设备相关的存储键
   */
  private getDeviceKey(key: string): string {
    return `${this.deviceId}_${key}`;
  }

  /**
   * 检查是否可以调用API
   * @returns Promise<boolean> 是否可以调用
   */
  async canCallAPI(): Promise<boolean> {
    try {
      // 确保设备ID已初始化
      if (!this.deviceId) {
        await this.initDeviceId();
      }
      
      const dataStore = await preferences.getPreferences(this.context, 'rate_limit');
      const today = this.getTodayString();
      const deviceKey = this.getDeviceKey(`usage_${today}`);
      const todayUsage = Number(await dataStore.get(deviceKey, 0));
      
      console.log(`设备 ${this.deviceId} 今日已使用: ${todayUsage}/${this.DAILY_LIMIT}`);
      return todayUsage < this.DAILY_LIMIT;
    } catch (error) {
      console.error('检查限流失败:', error);
      return false;
    }
  }

  /**
   * 获取限流状态
   * @returns Promise<RateLimitStatus> 限流状态
   */
  async getRateLimitStatus(): Promise<RateLimitStatus> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'rate_limit');
      const today = this.getTodayString();
      const deviceKey = this.getDeviceKey(`usage_${today}`);
      const todayUsage = Number(await dataStore.get(deviceKey, 0));
      
      if (todayUsage >= this.DAILY_LIMIT) {
        return RateLimitStatus.EXCEEDED;
      } else if (todayUsage >= this.DAILY_LIMIT * 0.8) {
        return RateLimitStatus.LIMITED;
      } else {
        return RateLimitStatus.ALLOWED;
      }
    } catch (error) {
      console.error('获取限流状态失败:', error);
      return RateLimitStatus.EXCEEDED;
    }
  }

  /**
   * 记录API调用
   * @returns Promise<boolean> 记录是否成功
   */
  async recordAPICall(): Promise<boolean> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'rate_limit');
      const today = this.getTodayString();
      const deviceKey = this.getDeviceKey(`usage_${today}`);
      const todayUsage = Number(await dataStore.get(deviceKey, 0));
      
      await dataStore.put(deviceKey, todayUsage + 1);
      await dataStore.flush();
      
      console.log(`设备 ${this.deviceId} 记录API调用，当前使用: ${todayUsage + 1}/${this.DAILY_LIMIT}`);
      return true;
    } catch (error) {
      console.error('记录API调用失败:', error);
      return false;
    }
  }

  /**
   * 获取今日剩余次数
   * @returns Promise<number> 剩余次数
   */
  async getRemainingCalls(): Promise<number> {
    try {
      // 确保设备ID已初始化
      if (!this.deviceId) {
        await this.initDeviceId();
      }
      
      const dataStore = await preferences.getPreferences(this.context, 'rate_limit');
      const today = this.getTodayString();
      const deviceKey = this.getDeviceKey(`usage_${today}`);
      const todayUsage = Number(await dataStore.get(deviceKey, 0));
      
      const remaining = Math.max(0, this.DAILY_LIMIT - todayUsage);
      console.log(`设备ID: ${this.deviceId}`);
      console.log(`存储键: ${deviceKey}`);
      console.log(`今日使用: ${todayUsage}`);
      console.log(`每日限制: ${this.DAILY_LIMIT}`);
      console.log(`剩余次数: ${remaining}`);
      return remaining;
    } catch (error) {
      console.error('获取剩余次数失败:', error);
      return 0;
    }
  }

  /**
   * 获取今日使用次数
   * @returns Promise<number> 使用次数
   */
  async getTodayUsage(): Promise<number> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'rate_limit');
      const today = this.getTodayString();
      const deviceKey = this.getDeviceKey(`usage_${today}`);
      return Number(await dataStore.get(deviceKey, 0));
    } catch (error) {
      console.error('获取今日使用次数失败:', error);
      return 0;
    }
  }

  /**
   * 重置限流（用于测试或特殊情况）
   * @returns Promise<boolean> 重置是否成功
   */
  async resetLimit(): Promise<boolean> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'rate_limit');
      const today = this.getTodayString();
      const deviceKey = this.getDeviceKey(`usage_${today}`);
      await dataStore.put(deviceKey, Number(0));
      await dataStore.flush();
      console.log(`设备 ${this.deviceId} 重置限流成功`);
      return true;
    } catch (error) {
      console.error('重置限流失败:', error);
      return false;
    }
  }

  /**
   * 清除所有限流数据（用于修复数据问题）
   * @returns Promise<boolean> 清除是否成功
   */
  async clearAllLimitData(): Promise<boolean> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'rate_limit');
      const keys: Record<string, Object> = await dataStore.getAll() as Record<string, Object>;
      
      for (const key of Object.keys(keys)) {
        if (key.startsWith('usage_') || key.startsWith('device_') || key.startsWith('app_device_')) {
          await dataStore.delete(key);
          console.log('删除旧数据键:', key);
        }
      }
      
      await dataStore.flush();
      console.log('清除所有限流数据成功');
      return true;
    } catch (error) {
      console.error('清除限流数据失败:', error);
      return false;
    }
  }

  /**
   * 获取今日日期字符串
   * @returns string 日期字符串 (YYYY-MM-DD)
   */
  private getTodayString(): string {
    const today = new Date();
    const year = Number(today.getFullYear());
    const month = String(Number(today.getMonth()) + 1).padStart(2, '0');
    const day = String(Number(today.getDate())).padStart(2, '0');
    return `${String(year)}-${month}-${day}`;
  }

  /**
   * 清理过期的使用记录（可选，用于节省存储空间）
   * @returns Promise<boolean> 清理是否成功
   */
  async cleanupExpiredRecords(): Promise<boolean> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'rate_limit');
      const today = this.getTodayString();
      const keys: Record<string, Object> = await dataStore.getAll() as Record<string, Object>;
      
      for (const key of Object.keys(keys)) {
        if (key.startsWith(`${this.deviceId}_usage_`) && key !== this.getDeviceKey(`usage_${today}`)) {
          await dataStore.delete(key);
        }
      }
      
      await dataStore.flush();
      console.log(`设备 ${this.deviceId} 清理过期记录成功`);
      return true;
    } catch (error) {
      console.error('清理过期记录失败:', error);
      return false;
    }
  }
}
