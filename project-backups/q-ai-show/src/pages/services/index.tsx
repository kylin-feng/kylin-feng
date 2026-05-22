import { View, Text, ScrollView } from "@tarojs/components";
import { useLoad } from "@tarojs/taro";
import "./index.scss";

interface Service {
  id: string;
  title: string;
  description: string;
  icon: string;
  features: string[];
  price: string;
  duration: string;
}

export default function Services() {
  useLoad(() => {
    console.log("服务介绍页面加载");
  });

  const services: Service[] = [
    {
      id: "1",
      title: "基础版小程序",
      description: "适合小型企业的基础功能小程序，包含基本展示和简单交互功能",
      icon: "🚀",
      features: [
        "页面展示功能",
        "基础交互设计", 
        "响应式布局",
        "基础数据管理",
        "微信授权登录"
      ],
      price: "¥3,000 - ¥8,000",
      duration: "1-2周"
    },
    {
      id: "2", 
      title: "标准版小程序",
      description: "功能完善的标准小程序，适合中型企业，包含完整的业务逻辑和用户管理",
      icon: "💼",
      features: [
        "完整业务逻辑",
        "用户管理系统",
        "支付功能集成",
        "数据统计分析",
        "客服系统",
        "营销活动功能"
      ],
      price: "¥8,000 - ¥20,000",
      duration: "2-4周"
    },
    {
      id: "3",
      title: "企业版小程序",
      description: "功能强大的企业级小程序，适合大型企业，包含复杂业务逻辑和高级功能",
      icon: "🏢",
      features: [
        "复杂业务逻辑",
        "多角色权限管理",
        "高级数据分析",
        "第三方系统集成",
        "自定义功能开发",
        "性能优化",
        "安全加固"
      ],
      price: "¥20,000 - ¥50,000",
      duration: "1-3个月"
    },
    {
      id: "4",
      title: "定制开发",
      description: "完全定制化的小程序开发服务，根据具体需求进行个性化开发",
      icon: "🎯",
      features: [
        "需求分析",
        "UI/UX设计",
        "功能定制开发",
        "性能优化",
        "测试部署",
        "后期维护",
        "技术培训"
      ],
      price: "面议",
      duration: "根据需求"
    }
  ];

  const advantages = [
    {
      title: "专业团队",
      description: "5年以上开发经验的专业团队",
      icon: "👥"
    },
    {
      title: "快速交付",
      description: "高效开发流程，快速交付项目",
      icon: "⚡"
    },
    {
      title: "质量保证",
      description: "严格测试，确保产品质量",
      icon: "✅"
    },
    {
      title: "后期维护",
      description: "提供完善的后期维护服务",
      icon: "🔧"
    }
  ];

  return (
    <View className="services-page">
      <View className="header-section">
        <Text className="page-title">服务介绍</Text>
        <Text className="page-subtitle">专业的小程序开发服务，助力企业数字化转型</Text>
      </View>

      <ScrollView className="services-container" scrollY>
        <View className="services-list">
          {services.map(service => (
            <View key={service.id} className="service-card">
              <View className="service-header">
                <Text className="service-icon">{service.icon}</Text>
                <View className="service-info">
                  <Text className="service-title">{service.title}</Text>
                  <Text className="service-price">{service.price}</Text>
                </View>
              </View>

              <Text className="service-description">{service.description}</Text>

              <View className="service-features">
                <Text className="features-title">包含功能</Text>
                <View className="features-list">
                  {service.features.map((feature, index) => (
                    <Text key={index} className="feature-item">• {feature}</Text>
                  ))}
                </View>
              </View>

              <View className="service-meta">
                <Text className="meta-item">开发周期: {service.duration}</Text>
              </View>
            </View>
          ))}
        </View>

        <View className="advantages-section">
          <Text className="section-title">我们的优势</Text>
          <View className="advantages-grid">
            {advantages.map((advantage, index) => (
              <View key={index} className="advantage-item">
                <Text className="advantage-icon">{advantage.icon}</Text>
                <Text className="advantage-title">{advantage.title}</Text>
                <Text className="advantage-desc">{advantage.description}</Text>
              </View>
            ))}
          </View>
        </View>

        <View className="process-section">
          <Text className="section-title">开发流程</Text>
          <View className="process-steps">
            <View className="process-step">
              <View className="step-number">1</View>
              <View className="step-content">
                <Text className="step-title">需求分析</Text>
                <Text className="step-desc">深入了解您的业务需求</Text>
              </View>
            </View>
            <View className="process-step">
              <View className="step-number">2</View>
              <View className="step-content">
                <Text className="step-title">设计开发</Text>
                <Text className="step-desc">UI设计和功能开发</Text>
              </View>
            </View>
            <View className="process-step">
              <View className="step-number">3</View>
              <View className="step-content">
                <Text className="step-title">测试部署</Text>
                <Text className="step-desc">全面测试并上线部署</Text>
              </View>
            </View>
            <View className="process-step">
              <View className="step-number">4</View>
              <View className="step-content">
                <Text className="step-title">后期维护</Text>
                <Text className="step-desc">持续优化和维护服务</Text>
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}
