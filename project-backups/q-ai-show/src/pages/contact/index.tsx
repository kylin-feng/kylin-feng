import { View, Text, Input, Button } from "@tarojs/components";
import { useLoad } from "@tarojs/taro";
import Taro from "@tarojs/taro";
import { useState } from "react";
import "./index.scss";

export default function Contact() {
  const [formData, setFormData] = useState({
    name: "",
    phone: "",
    company: "",
    message: ""
  });

  useLoad(() => {
    console.log("联系页面加载");
  });

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = () => {
    if (!formData.name || !formData.phone) {
      Taro.showToast({
        title: "请填写姓名和电话",
        icon: "error"
      });
      return;
    }

    Taro.showToast({
      title: "提交成功，我们会尽快联系您！",
      icon: "success"
    });

    // 重置表单
    setFormData({
      name: "",
      phone: "",
      company: "",
      message: ""
    });
  };

  const contactMethods = [
    {
      title: "微信咨询",
      value: "wx_developer_2024",
      icon: "💬",
      action: () => {
        Taro.setClipboardData({
          data: "wx_developer_2024",
          success: () => {
            Taro.showToast({
              title: "微信号已复制",
              icon: "success"
            });
          }
        });
      }
    },
    {
      title: "电话咨询",
      value: "138-0000-0000",
      icon: "📞",
      action: () => {
        Taro.makePhoneCall({
          phoneNumber: "13800000000"
        });
      }
    },
    {
      title: "邮箱联系",
      value: "developer@example.com",
      icon: "📧",
      action: () => {
        Taro.setClipboardData({
          data: "developer@example.com",
          success: () => {
            Taro.showToast({
              title: "邮箱已复制",
              icon: "success"
            });
          }
        });
      }
    }
  ];

  return (
    <View className="contact-page">
      <View className="header-section">
        <Text className="page-title">联系我们</Text>
        <Text className="page-subtitle">专业的小程序开发服务，期待与您合作</Text>
      </View>

      <View className="contact-methods">
        <Text className="section-title">联系方式</Text>
        <View className="methods-list">
          {contactMethods.map((method, index) => (
            <View key={index} className="method-item" onClick={method.action}>
              <Text className="method-icon">{method.icon}</Text>
              <View className="method-info">
                <Text className="method-title">{method.title}</Text>
                <Text className="method-value">{method.value}</Text>
              </View>
              <Text className="method-arrow">></Text>
            </View>
          ))}
        </View>
      </View>

      <View className="form-section">
        <Text className="section-title">在线咨询</Text>
        <View className="form-container">
          <View className="form-group">
            <Text className="form-label">姓名 *</Text>
            <Input
              className="form-input"
              placeholder="请输入您的姓名"
              value={formData.name}
              onInput={(e) => handleInputChange("name", e.detail.value)}
            />
          </View>

          <View className="form-group">
            <Text className="form-label">联系电话 *</Text>
            <Input
              className="form-input"
              placeholder="请输入您的联系电话"
              value={formData.phone}
              onInput={(e) => handleInputChange("phone", e.detail.value)}
            />
          </View>

          <View className="form-group">
            <Text className="form-label">公司名称</Text>
            <Input
              className="form-input"
              placeholder="请输入您的公司名称"
              value={formData.company}
              onInput={(e) => handleInputChange("company", e.detail.value)}
            />
          </View>

          <View className="form-group">
            <Text className="form-label">项目需求</Text>
            <Input
              className="form-input textarea"
              placeholder="请描述您的项目需求"
              value={formData.message}
              onInput={(e) => handleInputChange("message", e.detail.value)}
            />
          </View>

          <Button 
            className="submit-button"
            type="primary"
            onClick={handleSubmit}
          >
            提交咨询
          </Button>
        </View>
      </View>

      <View className="info-section">
        <Text className="section-title">服务承诺</Text>
        <View className="promises-list">
          <View className="promise-item">
            <Text className="promise-icon">⚡</Text>
            <Text className="promise-text">24小时内响应</Text>
          </View>
          <View className="promise-item">
            <Text className="promise-icon">🔒</Text>
            <Text className="promise-text">严格保密协议</Text>
          </View>
          <View className="promise-item">
            <Text className="promise-icon">💰</Text>
            <Text className="promise-text">合理报价</Text>
          </View>
          <View className="promise-item">
            <Text className="promise-icon">🛡️</Text>
            <Text className="promise-text">质量保证</Text>
          </View>
        </View>
      </View>
    </View>
  );
}
