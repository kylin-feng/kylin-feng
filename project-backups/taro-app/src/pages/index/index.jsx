import React, { useState, useEffect } from 'react'
import { View, Text, ScrollView } from '@tarojs/components'
import Taro from '@tarojs/taro'
import './index.scss'

const Index = () => {
  const [animationClass, setAnimationClass] = useState('')

  useEffect(() => {
    // 页面加载动画
    setTimeout(() => {
      setAnimationClass('animate-fade-in')
    }, 100)
  }, [])

  const handleNavigate = (url) => {
    Taro.navigateTo({ url })
  }

  const features = [
    {
      id: 1,
      title: 'AI实验室',
      subtitle: 'Q.AI Lab',
      description: '前沿AI技术研究',
      icon: '🔬',
      bgColor: 'bg-primary',
      route: '/pages/research/index'
    },
    {
      id: 2,
      title: 'AI工具链',
      subtitle: 'AI Tools',
      description: '智能生产力工具',
      icon: '🛠',
      bgColor: 'bg-secondary',
      route: '/pages/tools/index'
    },
    {
      id: 3,
      title: 'Q AI圈',
      subtitle: 'Education',
      description: 'AI教育与社区',
      icon: '🎓',
      bgColor: 'bg-accent',
      route: '/pages/education/index'
    },
    {
      id: 4,
      title: 'API服务',
      subtitle: 'API Service',
      description: 'AI能力开放平台',
      icon: '⚡',
      bgColor: 'bg-muted',
      route: '/pages/api/index'
    }
  ]

  return (
    <ScrollView className='index-container' scrollY>
      {/* Hero Section */}
      <View className={`hero-section ${animationClass}`}>
        <View className='hero-content'>
          {/* Logo */}
          <View className='logo-container'>
            <View className='logo-icon shape-circle bg-primary'>
              <Text className='logo-text'>Q</Text>
            </View>
            <Text className='logo-title font-light'>Q.AI</Text>
          </View>

          {/* Slogan */}
          <View className='slogan-container mt-xl'>
            <Text className='slogan-main font-light'>让天下没有</Text>
            <Text className='slogan-highlight font-medium text-primary'>难用的AI</Text>
          </View>

          <Text className='slogan-subtitle text-muted mt-md'>
            构建简单易用的AI生态系统
          </Text>
        </View>
      </View>

      {/* Features Grid */}
      <View className='features-section container'>
        <View className='section-header mb-lg'>
          <Text className='section-title font-medium'>产品矩阵</Text>
          <Text className='section-subtitle text-muted'>
            全方位AI应用场景
          </Text>
        </View>

        <View className='features-grid'>
          {features.map((feature, index) => (
            <View 
              key={feature.id}
              className={`feature-card card hover-lift animate-scale-in`}
              style={{
                animationDelay: `${index * 0.1}s`
              }}
              onClick={() => handleNavigate(feature.route)}
            >
              {/* Icon */}
              <View className='feature-icon-container'>
                <View className={`feature-icon shape-rounded-lg ${feature.bgColor}`}>
                  <Text className='feature-emoji'>{feature.icon}</Text>
                </View>
              </View>

              {/* Content */}
              <View className='feature-content'>
                <Text className='feature-title font-medium'>{feature.title}</Text>
                <Text className='feature-subtitle text-muted'>{feature.subtitle}</Text>
                <Text className='feature-description text-muted mt-xs'>
                  {feature.description}
                </Text>
              </View>

              {/* Arrow Indicator */}
              <View className='feature-arrow'>
                <View className='arrow-icon shape-circle bg-muted'>
                  <Text className='arrow-text'>→</Text>
                </View>
              </View>
            </View>
          ))}
        </View>
      </View>

      {/* Stats Section */}
      <View className='stats-section container mt-2xl'>
        <View className='stats-grid'>
          <View className='stat-item animate-slide-up'>
            <Text className='stat-number font-light'>1M+</Text>
            <Text className='stat-label text-muted'>用户</Text>
          </View>
          <View className='stat-item animate-slide-up'>
            <Text className='stat-number font-light'>50+</Text>
            <Text className='stat-label text-muted'>AI模型</Text>
          </View>
          <View className='stat-item animate-slide-up'>
            <Text className='stat-number font-light'>99.9%</Text>
            <Text className='stat-label text-muted'>可用性</Text>
          </View>
        </View>
      </View>

      {/* CTA Section */}
      <View className='cta-section container mt-2xl mb-2xl'>
        <View className='cta-card card p-xl text-center'>
          <Text className='cta-title font-medium mb-md'>开始体验</Text>
          <Text className='cta-description text-muted mb-lg'>
            加入Q.AI生态，让AI为您的工作增效
          </Text>
          <View className='cta-buttons flex justify-center gap-md'>
            <View className='btn btn-primary btn-md'>
              <Text>立即开始</Text>
            </View>
            <View className='btn btn-secondary btn-md'>
              <Text>了解更多</Text>
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  )
}

export default Index