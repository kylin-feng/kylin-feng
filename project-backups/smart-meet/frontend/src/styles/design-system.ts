// Design System Configuration
// 基于 shadcn/ui 极简设计理念

export const designTokens = {
  // 颜色系统 - 极简单色调色板
  colors: {
    // 主要颜色
    primary: {
      50: '#f8fafc',   // 极浅灰
      100: '#f1f5f9',  // 浅灰
      200: '#e2e8f0',  // 中灰
      300: '#cbd5e1',  // 
      400: '#94a3b8',  // 
      500: '#64748b',  // 中等灰
      600: '#475569',  // 
      700: '#334155',  // 深灰
      800: '#1e293b',  // 更深灰
      900: '#0f172a',  // 最深灰/黑
    },
    
    // 功能性颜色
    semantic: {
      success: '#10b981',   // 成功 - 绿色
      warning: '#f59e0b',   // 警告 - 黄色
      error: '#ef4444',     // 错误 - 红色
      info: '#3b82f6',      // 信息 - 蓝色
    },
    
    // 状态颜色
    status: {
      idle: '#cbd5e1',      // 待机
      working: '#10b981',   // 工作中
      analyzing: '#3b82f6', // 分析中
      completed: '#6b7280', // 完成
    }
  },

  // 间距系统 - 8px 基础网格
  spacing: {
    xs: '0.25rem',    // 4px
    sm: '0.5rem',     // 8px
    md: '1rem',       // 16px
    lg: '1.5rem',     // 24px
    xl: '2rem',       // 32px
    '2xl': '3rem',    // 48px
    '3xl': '4rem',    // 64px
    '4xl': '6rem',    // 96px
  },

  // 圆角系统
  borderRadius: {
    none: '0',
    sm: '0.25rem',    // 4px
    md: '0.375rem',   // 6px
    lg: '0.5rem',     // 8px
    xl: '0.75rem',    // 12px
    '2xl': '1rem',    // 16px
    full: '9999px',   // 完全圆形
  },

  // 阴影系统 - 极简阴影
  shadows: {
    none: 'none',
    sm: '0 1px 2px 0 rgb(0 0 0 / 0.05)',
    md: '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
    lg: '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
    xl: '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
  },

  // 字体系统
  typography: {
    fontFamily: {
      sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'monospace'],
    },
    fontSize: {
      xs: ['0.75rem', { lineHeight: '1rem' }],      // 12px
      sm: ['0.875rem', { lineHeight: '1.25rem' }],   // 14px
      base: ['1rem', { lineHeight: '1.5rem' }],      // 16px
      lg: ['1.125rem', { lineHeight: '1.75rem' }],   // 18px
      xl: ['1.25rem', { lineHeight: '1.75rem' }],    // 20px
      '2xl': ['1.5rem', { lineHeight: '2rem' }],     // 24px
      '3xl': ['1.875rem', { lineHeight: '2.25rem' }], // 30px
      '4xl': ['2.25rem', { lineHeight: '2.5rem' }],  // 36px
      '5xl': ['3rem', { lineHeight: '1' }],          // 48px
      '6xl': ['3.75rem', { lineHeight: '1' }],       // 60px
    },
    fontWeight: {
      light: '300',
      normal: '400',
      medium: '500',
      semibold: '600',
      bold: '700',
    }
  },

  // 动画系统 - 极简过渡
  animations: {
    duration: {
      fast: '150ms',
      normal: '250ms',
      slow: '350ms',
    },
    easing: {
      default: 'cubic-bezier(0.4, 0, 0.2, 1)',
      in: 'cubic-bezier(0.4, 0, 1, 1)',
      out: 'cubic-bezier(0, 0, 0.2, 1)',
      inOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    }
  },

  // 组件配置
  components: {
    button: {
      sizes: {
        sm: { padding: '0.5rem 0.75rem', fontSize: '0.875rem' },
        md: { padding: '0.625rem 1rem', fontSize: '1rem' },
        lg: { padding: '0.75rem 1.5rem', fontSize: '1.125rem' },
      },
      variants: {
        primary: {
          background: 'slate.900',
          color: 'white',
          hover: 'slate.800',
        },
        secondary: {
          background: 'slate.100',
          color: 'slate.900',
          hover: 'slate.200',
        },
        outline: {
          background: 'transparent',
          color: 'slate.700',
          border: 'slate.200',
          hover: 'slate.50',
        }
      }
    },
    
    card: {
      background: 'white',
      border: 'slate.200',
      borderRadius: 'lg',
      shadow: 'sm',
      padding: '1.5rem',
    },

    input: {
      background: 'white',
      border: 'slate.200',
      borderRadius: 'md',
      padding: '0.625rem 0.75rem',
      fontSize: '1rem',
      focus: {
        border: 'slate.400',
        outline: 'none',
      }
    }
  }
};

// 实用工具函数
export const getStatusColor = (status: string) => {
  return designTokens.colors.status[status as keyof typeof designTokens.colors.status] || designTokens.colors.status.idle;
};

export const getSemanticColor = (type: 'success' | 'warning' | 'error' | 'info') => {
  return designTokens.colors.semantic[type];
};

// CSS 类名生成器
export const cn = (...classes: (string | undefined | null | false)[]) => {
  return classes.filter(Boolean).join(' ');
};

export default designTokens;