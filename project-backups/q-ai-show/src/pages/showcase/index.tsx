import { View, Text, Image, ScrollView } from "@tarojs/components";
import { useLoad } from "@tarojs/taro";
import Taro from "@tarojs/taro";
import "./index.scss";

interface Project {
  id: string;
  title: string;
  description: string;
  category: string;
  image: string;
  features: string[];
  techStack: string[];
  duration: string;
  status: string;
}

export default function Showcase() {
  useLoad(() => {
    console.log("项目展示页面加载");
  });

  const projects: Project[] = [
    {
      id: "1",
      title: "智慧超市管理系统",
      description: "为连锁超市打造的智能管理小程序，包含商品管理、库存监控、会员系统、数据分析等功能",
      category: "零售行业",
      image: require("../../images/robot.png"),
      features: ["商品管理", "库存监控", "会员系统", "数据分析", "订单管理"],
      techStack: ["Taro", "React", "TypeScript", "Sass"],
      duration: "2个月",
      status: "已完成"
    },
    {
      id: "2", 
      title: "餐厅点餐小程序",
      description: "现代化餐厅点餐系统，支持在线点餐、支付、排队叫号、会员积分等功能",
      category: "餐饮行业",
      image: require("../../images/send.png"),
      features: ["在线点餐", "支付系统", "排队叫号", "会员积分", "菜品推荐"],
      techStack: ["Taro", "React", "TypeScript", "Sass"],
      duration: "1.5个月",
      status: "已完成"
    },
    {
      id: "3",
      title: "健身房会员管理",
      description: "专业的健身房会员管理系统，包含课程预约、教练管理、会员卡管理、健身数据统计等功能",
      category: "健身行业",
      image: require("../../images/love.png"),
      features: ["课程预约", "教练管理", "会员卡管理", "健身数据", "社区互动"],
      techStack: ["Taro", "React", "TypeScript", "Sass"],
      duration: "2.5个月",
      status: "开发中"
    },
    {
      id: "4",
      title: "美容院预约系统",
      description: "专业美容院预约管理小程序，支持服务预约、技师管理、客户档案、营销活动等功能",
      category: "美容行业",
      image: require("../../images/robot.png"),
      features: ["服务预约", "技师管理", "客户档案", "营销活动", "评价系统"],
      techStack: ["Taro", "React", "TypeScript", "Sass"],
      duration: "1个月",
      status: "已完成"
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "已完成": return "#52c41a";
      case "开发中": return "#1890ff";
      case "规划中": return "#faad14";
      default: return "#666";
    }
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      "零售行业": "#667eea",
      "餐饮行业": "#f093fb", 
      "健身行业": "#4facfe",
      "美容行业": "#43e97b"
    };
    return colors[category] || "#667eea";
  };

  return (
    <View className="showcase-page">
      <View className="header-section">
        <Text className="page-title">项目展示</Text>
        <Text className="page-subtitle">专业的小程序开发服务，助力企业数字化转型</Text>
      </View>

      <ScrollView className="projects-container" scrollY>
        {projects.map(project => (
          <View key={project.id} className="project-card">
            <View className="project-header">
              <Image 
                className="project-image" 
                src={project.image} 
                mode="aspectFill"
              />
              <View className="project-info">
                <Text className="project-title">{project.title}</Text>
                <View 
                  className="project-category"
                  style={{ backgroundColor: getCategoryColor(project.category) }}
                >
                  <Text className="category-text">{project.category}</Text>
                </View>
              </View>
            </View>

            <Text className="project-description">{project.description}</Text>

            <View className="project-features">
              <Text className="features-title">核心功能</Text>
              <View className="features-list">
                {project.features.map((feature, index) => (
                  <Text key={index} className="feature-tag">{feature}</Text>
                ))}
              </View>
            </View>

            <View className="project-tech">
              <Text className="tech-title">技术栈</Text>
              <View className="tech-list">
                {project.techStack.map((tech, index) => (
                  <Text key={index} className="tech-tag">{tech}</Text>
                ))}
              </View>
            </View>

            <View className="project-footer">
              <View className="project-meta">
                <Text className="meta-item">开发周期: {project.duration}</Text>
                <Text 
                  className="meta-item status"
                  style={{ color: getStatusColor(project.status) }}
                >
                  状态: {project.status}
                </Text>
              </View>
            </View>
          </View>
        ))}
      </ScrollView>

      <View className="cta-section">
        <Text className="cta-title">需要定制开发？</Text>
        <Text className="cta-desc">我们提供专业的小程序开发服务</Text>
        <View 
          className="cta-button"
          onClick={() => Taro.switchTab({ url: "/pages/contact/index" })}
        >
          <Text className="cta-button-text">立即咨询</Text>
        </View>
      </View>
    </View>
  );
}
