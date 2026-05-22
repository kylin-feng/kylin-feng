/// <reference types="@tarojs/taro" />

declare module "*.png";
declare module "*.gif";
declare module "*.jpg";
declare module "*.jpeg";
declare module "*.svg";
declare module "*.css";
declare module "*.less";
declare module "*.scss";
declare module "*.sass";
declare module "*.styl";

// 优学教程平台相关类型定义
declare namespace Course {
  // 用户信息接口
  interface UserInfo {
    id: string;
    nickname: string;
    avatar: string;
    phone?: string;
    email?: string;
    vipLevel: number;
    balance: number;
    totalPurchased: number;
    createTime: string;
    lastLoginTime: string;
  }

  // 教程/课程接口
  interface CourseInfo {
    id: string;
    title: string;
    description: string;
    cover: string;
    price: number;
    originalPrice?: number;
    categoryId: string;
    categoryName: string;
    teacherInfo: TeacherInfo;
    difficulty: 'beginner' | 'intermediate' | 'advanced';
    duration: number; // 总时长(分钟)
    studentCount: number;
    rating: number;
    tags: string[];
    chapters: ChapterInfo[];
    createTime: string;
    updateTime: string;
    status: 'published' | 'draft' | 'offline';
    isPurchased?: boolean;
    progress?: number; // 学习进度百分比
    trialVideoUrl?: string; // 试看视频URL
    highlights: string[]; // 课程亮点
  }

  // 章节信息接口
  interface ChapterInfo {
    id: string;
    courseId: string;
    title: string;
    description?: string;
    order: number;
    duration: number; // 时长(分钟)
    videoUrl: string;
    isLocked: boolean;
    isTrial: boolean; // 是否为试看章节
    isCompleted?: boolean;
  }

  // 讲师信息接口
  interface TeacherInfo {
    id: string;
    name: string;
    avatar: string;
    title: string; // 职称
    introduction: string;
    experience: number; // 教学经验(年)
    studentCount: number;
    courseCount: number;
  }

  // 课程分类接口
  interface CategoryInfo {
    id: string;
    name: string;
    icon: string;
    description?: string;
    courseCount: number;
    parentId?: string;
    children?: CategoryInfo[];
  }

  // 购买订单接口
  interface OrderInfo {
    id: string;
    userId: string;
    courseId: string;
    courseTitle: string;
    courseCover: string;
    originalPrice: number;
    actualPrice: number;
    discountAmount: number;
    paymentMethod: 'wechat' | 'alipay';
    status: 'pending' | 'paid' | 'cancelled' | 'refunded';
    createTime: string;
    payTime?: string;
    expireTime: string;
  }

  // 学习记录接口
  interface StudyRecord {
    id: string;
    userId: string;
    courseId: string;
    chapterId: string;
    progress: number; // 章节观看进度百分比
    duration: number; // 已观看时长(秒)
    isCompleted: boolean;
    lastWatchTime: string;
  }

  // 视频播放配置接口
  interface VideoConfig {
    url: string;
    quality: 'sd' | 'hd' | 'uhd';
    hasWatermark: boolean;
    watermarkText?: string;
    allowDownload: boolean;
    allowSeek: boolean;
    trialDuration?: number; // 试看时长(秒)
    speedOptions: number[]; // 播放速度选项
  }

  // 支付配置接口
  interface PaymentConfig {
    courseId: string;
    courseTitle: string;
    originalPrice: number;
    currentPrice: number;
    discountInfo?: {
      type: 'percent' | 'amount';
      value: number;
      description: string;
    };
    coupons?: CouponInfo[];
  }

  // 优惠券接口
  interface CouponInfo {
    id: string;
    name: string;
    type: 'percent' | 'amount';
    value: number;
    minAmount: number;
    description: string;
    startTime: string;
    endTime: string;
    isUsed: boolean;
  }

  // API响应基础接口
  interface BaseResponse<T = any> {
    code: number;
    message: string;
    data: T;
    timestamp: number;
  }

  // 分页数据接口
  interface PageData<T> {
    list: T[];
    total: number;
    page: number;
    pageSize: number;
    hasMore: boolean;
  }

  // 登录响应接口
  interface LoginResponse {
    token: string;
    userInfo: UserInfo;
    expiresIn: number;
  }

  // 系统配置接口
  interface SystemConfig {
    appName: string;
    appVersion: string;
    apiBaseUrl: string;
    uploadUrl: string;
    videoBaseUrl: string;
    maxTrialDuration: number;
    watermarkConfig: {
      enabled: boolean;
      text: string;
      position: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' | 'center';
      opacity: number;
    };
  }

  // 搜索条件接口
  interface SearchParams {
    keyword?: string;
    categoryId?: string;
    difficulty?: string;
    priceRange?: {
      min: number;
      max: number;
    };
    sortBy: 'latest' | 'popular' | 'price-asc' | 'price-desc' | 'rating';
    page: number;
    pageSize: number;
  }
}

// 微信小程序相关类型扩展
declare namespace WechatMiniprogram {
  interface Wx {
    env: {
      USER_DATA_PATH: string;
    };
  }
}

// 全局环境变量
declare namespace NodeJS {
  interface ProcessEnv {
    NODE_ENV: 'development' | 'production' | 'test';
    TARO_ENV: 'weapp' | 'alipay' | 'swan' | 'tt' | 'h5' | 'rn' | 'qq' | 'quickapp';
  }
}


