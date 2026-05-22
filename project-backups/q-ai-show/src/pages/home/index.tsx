import { View, Text, Image } from "@tarojs/components";
import { useLoad } from "@tarojs/taro";
import Taro from "@tarojs/taro";
import "./index.scss";

export default function Index() {
  useLoad(() => {
    console.log("首页加载");
  });

  const skills = [
    { name: "Taro", level: 95 },
    { name: "React", level: 90 },
    { name: "TypeScript", level: 85 },
    { name: "微信小程序", level: 95 },
    { name: "Node.js", level: 80 },
    { name: "UI/UX设计", level: 75 }
  ];

  const experiences = [
    {
      title: "5年+ 开发经验",
      desc: "专注小程序开发，服务过50+企业客户"
    },
    {
      title: "全栈开发能力", 
      desc: "前端、后端、数据库全栈技术栈"
    },
    {
      title: "项目交付能力",
      desc: "平均项目周期2-4周，按时交付率100%"
    }
  ];

  const goToShowcase = () => {
    Taro.switchTab({ url: "/pages/showcase/index" });
  };

  const goToContact = () => {
    Taro.switchTab({ url: "/pages/contact/index" });
  };

  return (
    <View className="home-page">
      <View className="hero-section">
        <View className="avatar-container">
          <Image 
            className="avatar" 
            src={require("../../images/robot.png")} 
            mode="aspectFill"
          />
        </View>
        <Text className="name">小程序开发专家</Text>
        <Text className="title">专业 · 高效 · 可靠</Text>
        <Text className="intro">
          5年+开发经验，专注微信小程序开发，为企业提供专业的技术解决方案
        </Text>
      </View>

      <View className="skills-section">
        <Text className="section-title">技术专长</Text>
        <View className="skills-list">
          {skills.map((skill, index) => (
            <View key={index} className="skill-item">
              <Text className="skill-name">{skill.name}</Text>
              <View className="skill-bar">
                <View 
                  className="skill-progress" 
                  style={{ width: `${skill.level}%` }}
                />
              </View>
              <Text className="skill-level">{skill.level}%</Text>
            </View>
          ))}
        </View>
      </View>

      <View className="experience-section">
        <Text className="section-title">核心优势</Text>
        <View className="experience-list">
          {experiences.map((exp, index) => (
            <View key={index} className="experience-item">
              <Text className="exp-title">{exp.title}</Text>
              <Text className="exp-desc">{exp.desc}</Text>
            </View>
          ))}
        </View>
      </View>

      <View className="cta-section">
        <View className="cta-buttons">
          <View className="cta-button primary" onClick={goToShowcase}>
            <Text className="cta-text">查看项目</Text>
          </View>
          <View className="cta-button secondary" onClick={goToContact}>
            <Text className="cta-text">联系我</Text>
          </View>
        </View>
      </View>
    </View>
  );
}
