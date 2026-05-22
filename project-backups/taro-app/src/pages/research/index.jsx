import React, { useState, useEffect } from 'react'
import { View, Text, ScrollView } from '@tarojs/components'
import Taro from '@tarojs/taro'
import './index.scss'

const Research = () => {
  const [activeTab, setActiveTab] = useState(0)
  const [animationClass, setAnimationClass] = useState('')

  useEffect(() => {
    setTimeout(() => {
      setAnimationClass('animate-fade-in')
    }, 100)
  }, [])

  const researchAreas = [
    {
      id: 'multi-agent',
      title: 'Multi-Agent',
      subtitle: '多智能体系统',
      description: '协作式智能系统研究',
      icon: '🤝',
      bgColor: '#222',
      progress: 85,
      status: 'active',
      papers: 12,
      projects: 4
    },
    {
      id: 'context',
      title: 'Context Management', 
      subtitle: '上下文管理',
      description: '长文本处理与检索',
      icon: '📚',
      bgColor: '#3b82f6',
      progress: 78,
      status: 'research',
      papers: 8,
      projects: 3
    },
    {
      id: 'workflow',
      title: 'Workflow',
      subtitle: '工作流程',
      description: '自动化任务编排',
      icon: '⚡',
      bgColor: '#10b981',
      progress: 92,
      status: 'active',
      papers: 15,
      projects: 6
    },
    {
      id: 'memory',
      title: 'Memory Control',
      subtitle: '记忆控制',
      description: '智能知识压缩',
      icon: '🧠',
      bgColor: '#8b5cf6',
      progress: 65,
      status: 'planning',
      papers: 6,
      projects: 2
    }
  ]

  const handleTabChange = (index) => {
    setActiveTab(index)
  }

  const getStatusText = (status) => {
    const statusMap = {
      active: '进行中',
      research: '研究中', 
      planning: '规划中'
    }
    return statusMap[status] || status
  }

  const getStatusColor = (status) => {
    const colorMap = {
      active: '#22c55e',
      research: '#f59e0b',
      planning: '#94a3b8'
    }
    return colorMap[status] || '#94a3b8'
  }

  return (
    <ScrollView className='research-container' scrollY>
      {/* Header */}
      <View className={`research-header ${animationClass}`}>
        <View className='container'>
          <View className='header-content'>
            <Text className='header-title font-light'>Q.AI Lab</Text>
            <Text className='header-subtitle text-muted'>
              前沿AI技术研究实验室
            </Text>
          </View>
          
          {/* Status Indicators */}
          <View className='status-overview'>
            <View className='status-item'>
              <View className='status-dot' style={{ backgroundColor: '#22c55e' }}></View>
              <Text className='status-text'>4个项目进行中</Text>
            </View>
            <View className='status-item'>
              <View className='status-dot' style={{ backgroundColor: '#f59e0b' }}></View>
              <Text className='status-text'>41篇论文发表</Text>
            </View>
          </View>
        </View>
      </View>

      {/* Research Areas Grid */}
      <View className='research-content container'>
        <View className='section-header mb-lg'>
          <Text className='section-title font-medium'>研究方向</Text>
          <Text className='section-subtitle text-muted'>
            核心技术领域探索
          </Text>
        </View>

        <View className='research-grid'>
          {researchAreas.map((area, index) => (
            <View 
              key={area.id}
              className={`research-card card hover-lift animate-scale-in`}
              style={{ animationDelay: `${index * 0.15}s` }}
              onClick={() => handleTabChange(index)}
            >
              {/* Card Header */}
              <View className='card-header'>
                <View className='icon-container'>
                  <View 
                    className='research-icon shape-rounded-lg'
                    style={{ backgroundColor: area.bgColor }}
                  >
                    <Text className='icon-emoji'>{area.icon}</Text>
                  </View>
                </View>
                
                <View className='status-badge'>
                  <View 
                    className='status-indicator shape-circle'
                    style={{ backgroundColor: getStatusColor(area.status) }}
                  ></View>
                  <Text className='status-label'>{getStatusText(area.status)}</Text>
                </View>
              </View>

              {/* Card Content */}
              <View className='card-content'>
                <Text className='area-title font-medium'>{area.title}</Text>
                <Text className='area-subtitle text-muted'>{area.subtitle}</Text>
                <Text className='area-description text-muted mt-xs'>
                  {area.description}
                </Text>
              </View>

              {/* Progress Bar */}
              <View className='progress-section mt-md'>
                <View className='progress-header'>
                  <Text className='progress-label text-muted'>研究进度</Text>
                  <Text className='progress-value font-medium'>{area.progress}%</Text>
                </View>
                <View className='progress-bar bg-muted'>
                  <View 
                    className='progress-fill'
                    style={{ 
                      width: `${area.progress}%`,
                      backgroundColor: area.bgColor 
                    }}
                  ></View>
                </View>
              </View>

              {/* Stats */}
              <View className='card-stats flex justify-between mt-md'>
                <View className='stat-item'>
                  <Text className='stat-number font-medium'>{area.papers}</Text>
                  <Text className='stat-label text-muted'>论文</Text>
                </View>
                <View className='stat-item'>
                  <Text className='stat-number font-medium'>{area.projects}</Text>
                  <Text className='stat-label text-muted'>项目</Text>
                </View>
              </View>
            </View>
          ))}
        </View>
      </View>

      {/* Research Detail */}
      <View className='research-detail container mt-2xl'>
        <View className='detail-card card p-xl'>
          <View className='detail-header mb-lg'>
            <View className='detail-icon-container'>
              <View 
                className='detail-icon shape-rounded-lg'
                style={{ backgroundColor: researchAreas[activeTab]?.bgColor }}
              >
                <Text className='detail-emoji'>{researchAreas[activeTab]?.icon}</Text>
              </View>
            </View>
            <View className='detail-text'>
              <Text className='detail-title font-medium'>
                {researchAreas[activeTab]?.title}
              </Text>
              <Text className='detail-subtitle text-muted'>
                {researchAreas[activeTab]?.subtitle}
              </Text>
            </View>
          </View>

          <View className='detail-content'>
            <Text className='content-section-title font-medium mb-md'>核心研究内容</Text>
            <View className='content-items'>
              <View className='content-item'>
                <View className='item-dot bg-primary shape-circle'></View>
                <Text className='item-text text-muted'>算法优化与性能提升</Text>
              </View>
              <View className='content-item'>
                <View className='item-dot bg-primary shape-circle'></View>
                <Text className='item-text text-muted'>实际应用场景验证</Text>
              </View>
              <View className='content-item'>
                <View className='item-dot bg-primary shape-circle'></View>
                <Text className='item-text text-muted'>开源工具与框架开发</Text>
              </View>
            </div>
          </View>

          <View className='detail-actions mt-lg'>
            <View className='btn btn-primary btn-md'>
              <Text>查看详情</Text>
            </View>
            <View className='btn btn-secondary btn-md'>
              <Text>相关论文</Text>
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  )
}

export default Research