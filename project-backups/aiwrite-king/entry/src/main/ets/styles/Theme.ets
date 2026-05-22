/**
 * 全局主题样式常量
 * 提供统一的设计系统和样式规范
 */

/**
 * 颜色主题
 */
export class ThemeColors {
  // 主色调
  static readonly PRIMARY = '#007AFF';
  static readonly PRIMARY_LIGHT = '#4DA6FF';
  static readonly PRIMARY_DARK = '#0056CC';
  
  // 功能色
  static readonly SUCCESS = '#34C759';
  static readonly WARNING = '#FF9500';
  static readonly ERROR = '#FF3B30';
  static readonly INFO = '#5AC8FA';
  
  // 中性色
  static readonly WHITE = '#FFFFFF';
  static readonly BLACK = '#000000';
  static readonly GRAY_50 = '#F8F9FA';
  static readonly GRAY_100 = '#F2F2F7';
  static readonly GRAY_200 = '#E5E5E5';
  static readonly GRAY_300 = '#D1D1D6';
  static readonly GRAY_400 = '#C7C7CC';
  static readonly GRAY_500 = '#8E8E93';
  static readonly GRAY_600 = '#666666';
  static readonly GRAY_700 = '#48484A';
  static readonly GRAY_800 = '#3A3A3C';
  static readonly GRAY_900 = '#1A1A1A';
  
  // 文本色
  static readonly TEXT_PRIMARY = '#1A1A1A';
  static readonly TEXT_SECONDARY = '#666666';
  static readonly TEXT_TERTIARY = '#999999';
  static readonly TEXT_DISABLED = '#C7C7CC';
  
  // 背景色
  static readonly BACKGROUND_PRIMARY = '#FFFFFF';
  static readonly BACKGROUND_SECONDARY = '#F8F9FA';
  static readonly BACKGROUND_TERTIARY = '#F2F2F7';
  
  // 边框色
  static readonly BORDER_LIGHT = '#E5E5E5';
  static readonly BORDER_MEDIUM = '#D1D1D6';
  static readonly BORDER_DARK = '#8E8E93';
}

/**
 * 字体样式
 */
export class ThemeTypography {
  // 字体大小
  static readonly FONT_SIZE_XS = 12;
  static readonly FONT_SIZE_SM = 14;
  static readonly FONT_SIZE_MD = 16;
  static readonly FONT_SIZE_LG = 18;
  static readonly FONT_SIZE_XL = 20;
  static readonly FONT_SIZE_XXL = 22;
  static readonly FONT_SIZE_XXXL = 24;
  
  // 行高
  static readonly LINE_HEIGHT_TIGHT = 1.2;
  static readonly LINE_HEIGHT_NORMAL = 1.4;
  static readonly LINE_HEIGHT_RELAXED = 1.6;
  
  // 字重
  static readonly FONT_WEIGHT_LIGHT = FontWeight.Light;
  static readonly FONT_WEIGHT_NORMAL = FontWeight.Normal;
  static readonly FONT_WEIGHT_MEDIUM = FontWeight.Medium;
  static readonly FONT_WEIGHT_BOLD = FontWeight.Bold;
}

/**
 * 间距系统
 */
export class ThemeSpacing {
  static readonly SPACE_XS = 4;
  static readonly SPACE_SM = 8;
  static readonly SPACE_MD = 12;
  static readonly SPACE_LG = 16;
  static readonly SPACE_XL = 20;
  static readonly SPACE_XXL = 24;
  static readonly SPACE_XXXL = 32;
  static readonly SPACE_XXXXL = 40;
}

/**
 * 圆角系统
 */
export class ThemeRadius {
  static readonly RADIUS_SM = 4;
  static readonly RADIUS_MD = 8;
  static readonly RADIUS_LG = 12;
  static readonly RADIUS_XL = 16;
  static readonly RADIUS_XXL = 20;
  static readonly RADIUS_XXXL = 24;
  static readonly RADIUS_FULL = 999;
}

/**
 * 阴影系统
 */
export class ThemeShadows {
  static readonly SHADOW_SM = {
    radius: 4,
    color: '#00000005',
    offsetX: 0,
    offsetY: 2
  };
  
  static readonly SHADOW_MD = {
    radius: 8,
    color: '#00000010',
    offsetX: 0,
    offsetY: 4
  };
  
  static readonly SHADOW_LG = {
    radius: 12,
    color: '#00000015',
    offsetX: 0,
    offsetY: 8
  };
  
  static readonly SHADOW_XL = {
    radius: 16,
    color: '#00000020',
    offsetX: 0,
    offsetY: 12
  };
}

/**
 * 动画系统
 */
export class ThemeAnimation {
  static readonly DURATION_FAST = 150;
  static readonly DURATION_NORMAL = 300;
  static readonly DURATION_SLOW = 500;
  
  static readonly EASING_EASE_IN = 'ease-in';
  static readonly EASING_EASE_OUT = 'ease-out';
  static readonly EASING_EASE_IN_OUT = 'ease-in-out';
}

/**
 * 组件样式预设
 */
export class ThemeComponents {
  // 按钮样式
  static readonly BUTTON_PRIMARY = {
    backgroundColor: ThemeColors.PRIMARY,
    fontColor: ThemeColors.WHITE,
    fontSize: ThemeTypography.FONT_SIZE_SM,
    fontWeight: ThemeTypography.FONT_WEIGHT_MEDIUM,
    borderRadius: ThemeRadius.RADIUS_XXL,
    padding: { left: 20, right: 20, top: 12, bottom: 12 }
  };
  
  static readonly BUTTON_SECONDARY = {
    backgroundColor: ThemeColors.GRAY_100,
    fontColor: ThemeColors.PRIMARY,
    fontSize: ThemeTypography.FONT_SIZE_SM,
    fontWeight: ThemeTypography.FONT_WEIGHT_MEDIUM,
    borderRadius: ThemeRadius.RADIUS_XXL,
    padding: { left: 20, right: 20, top: 12, bottom: 12 }
  };
  
  // 卡片样式
  static readonly CARD = {
    backgroundColor: ThemeColors.WHITE,
    borderRadius: ThemeRadius.RADIUS_XL,
    padding: ThemeSpacing.SPACE_XL,
    shadow: ThemeShadows.SHADOW_MD,
    border: {
      width: 1,
      color: ThemeColors.BORDER_LIGHT
    }
  };
  
  // 输入框样式
  static readonly INPUT = {
    backgroundColor: ThemeColors.GRAY_50,
    borderRadius: ThemeRadius.RADIUS_LG,
    padding: { left: 16, right: 16, top: 12, bottom: 12 },
    border: {
      width: 1,
      color: ThemeColors.BORDER_LIGHT
    },
    fontSize: ThemeTypography.FONT_SIZE_MD
  };
}

/**
 * 主题工具类
 */
export class ThemeUtils {
  /**
   * 获取状态颜色
   */
  static getStatusColor(status: 'success' | 'warning' | 'error' | 'info'): string {
    switch (status) {
      case 'success': return ThemeColors.SUCCESS;
      case 'warning': return ThemeColors.WARNING;
      case 'error': return ThemeColors.ERROR;
      case 'info': return ThemeColors.INFO;
      default: return ThemeColors.GRAY_500;
    }
  }
  
  /**
   * 获取状态背景色
   */
  static getStatusBackgroundColor(status: 'success' | 'warning' | 'error' | 'info'): string {
    switch (status) {
      case 'success': return '#E8F5E8';
      case 'warning': return '#FFF4E6';
      case 'error': return '#FFE5E5';
      case 'info': return '#E6F7FF';
      default: return ThemeColors.GRAY_100;
    }
  }
  
  /**
   * 创建渐变背景
   */
  static createGradientBackground(startColor: string, endColor: string): any {
    return {
      linearGradient: {
        direction: GradientDirection.Bottom,
        colors: [[startColor, 0.0], [endColor, 1.0]]
      }
    };
  }
  
  /**
   * 创建阴影样式
   */
  static createShadow(elevation: number): any {
    const shadowMap = {
      1: ThemeShadows.SHADOW_SM,
      2: ThemeShadows.SHADOW_MD,
      3: ThemeShadows.SHADOW_LG,
      4: ThemeShadows.SHADOW_XL
    };
    return shadowMap[elevation] || ThemeShadows.SHADOW_MD;
  }
}
