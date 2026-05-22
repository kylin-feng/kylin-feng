import React, { useState, useEffect } from 'react'
import { View, Text, ScrollView, Image } from '@tarojs/components'
import Taro from '@tarojs/taro'
import './index.scss'

const Profile = () => {
  const [animationClass, setAnimationClass] = useState('')
  const [userStats, setUserStats] = useState({
    toolsUsed: 12,
    projectsCreated: 8,
    timesSaved: 24,
    points: 1250
  })

  useEffect(() => {
    setTimeout(() => {
      setAnimationClass('animate-fade-in')
    }, 100)
  }, [])

  const menuItems = [
    {
      id: 1,
      title: '我的工具',
      subtitle: '常用工具收藏',
      icon: '🛠',
      color: '#3b82f6',
      count: userStats.toolsUsed,
      route: '/pages/my-tools/index'
    },
    {
      id: 2,
      title: '使用记录',
      subtitle: '工具使用历史',
      icon: '📊',
      color: '#10b981',
      count: userStats.projectsCreated,
      route: '/pages/usage-history/index'
    },
    {
      id: 3,
      title: '积分商城',
      subtitle: '兑换精彩奖励',
      icon: '🎁',
      color: '#f59e0b',
      count: userStats.points,
      route: '/pages/points-mall/index'
    },
    {
      id: 4,
      title: '帮助中心',
      subtitle: '常见问题解答',
      icon: '❓',
      color: '#8b5cf6',
      count: null,
      route: '/pages/help/index'
    }
  ]

  const quickActions = [
    { id: 1, name: '反馈建议', icon: '💬', color: '#06b6d4' },
    { id: 2, name: '邀请好友', icon: '👥', color: '#ec4899' },
    { id: 3, name: '设置', icon: '⚙️', color: '#64748b' }
  ]

  const handleMenuClick = (item) => {
    if (item.route) {
      Taro.navigateTo({ url: item.route })
    }
  }

  const handleActionClick = (action) => {
    Taro.showToast({
      title: action.name,
      icon: 'none',
      duration: 1500
    })
  }

  return (
    <ScrollView className='profile-container' scrollY>
      {/* User Header */}
      <View className={`profile-header ${animationClass}`}>
        <View className='container'>
          {/* Avatar Section */}
          <View className='avatar-section'>
            <View className='avatar-container'>
              <View className='avatar-ring bg-primary'>
                <View className='avatar-image shape-circle bg-muted'>
                  <Text className='avatar-text font-medium'>Q</Text>
                </View>
              </View>
              <View className='online-indicator bg-primary shape-circle'></View>
            </View>
            
            <View className='user-info'>
              <Text className='user-name font-medium'>Q.AI 用户</Text>
              <Text className='user-level text-muted'>高级会员</Text>
              <View className='user-tags flex gap-xs mt-xs'>
                <View className='tag bg-primary'>
                  <Text className='tag-text'>VIP</Text>
                </View>
                <View className='tag bg-secondary'>
                  <Text className='tag-text'>早期用户</Text>
                </View>
              </View>
            </View>
          </View>

          {/* Stats Overview */}
          <View className='stats-overview mt-xl'>
            <View className='stats-grid'>
              <View className='stat-item animate-scale-in'>
                <Text className='stat-number font-light'>{userStats.toolsUsed}</Text>
                <Text className='stat-label text-muted'>使用工具</Text>
              </View>
              <View className='stat-item animate-scale-in'>
                <Text className='stat-number font-light'>{userStats.projectsCreated}</Text>
                <Text className='stat-label text-muted'>创建项目</Text>
              </View>
              <View className='stat-item animate-scale-in'>
                <Text className='stat-number font-light'>{userStats.timesSaved}h</Text>
                <Text className='stat-label text-muted'>节省时间</Text>
              </View>
            </View>
          </View>
        </View>
      </View>

      {/* Menu Section */}
      <View className='menu-section container'>
        <View className='section-header mb-lg'>
          <Text className='section-title font-medium'>功能中心</Text>
        </View>

        <View className='menu-grid'>
          {menuItems.map((item, index) => (
            <View
              key={item.id}
              className={`menu-card card hover-lift animate-scale-in`}
              style={{ animationDelay: `${index * 0.1}s` }}
              onClick={() => handleMenuClick(item)}
            >
              {/* Card Header */}
              <View className='menu-header'>
                <View 
                  className='menu-icon shape-rounded-lg'
                  style={{ backgroundColor: item.color }}
                >
                  <Text className='icon-emoji'>{item.icon}</Text>
                </View>
                
                {item.count !== null && (
                  <View className='count-badge bg-muted'>
                    <Text className='count-text font-medium'>{item.count}</Text>
                  </View>
                )}
              </View>

              {/* Card Content */}
              <View className='menu-content'>
                <Text className='menu-title font-medium'>{item.title}</Text>
                <Text className='menu-subtitle text-muted'>{item.subtitle}</Text>
              </View>

              {/* Arrow */}
              <View className='menu-arrow'>
                <View className='arrow-icon shape-circle bg-muted'>
                  <Text className='arrow-text'>→</Text>
                </View>
              </View>

              {/* Hover Glow */}
              <View className='card-glow' style={{ backgroundColor: item.color }}></View>
            </View>
          ))}
        </View>
      </View>

      {/* Quick Actions */}
      <View className='actions-section container mt-xl'>
        <View className='section-header mb-lg'>
          <Text className='section-title font-medium'>快捷操作</Text>
        </View>

        <View className='actions-list'>
          {quickActions.map((action, index) => (
            <View
              key={action.id}
              className={`action-item card hover-scale animate-slide-up`}
              style={{ animationDelay: `${index * 0.1}s` }}
              onClick={() => handleActionClick(action)}
            >
              <View className='action-content'>
                <View 
                  className='action-icon shape-circle'
                  style={{ backgroundColor: action.color }}
                >
                  <Text className='action-emoji'>{action.icon}</Text>
                </View>
                <Text className='action-name font-medium'>{action.name}</Text>
              </View>
              
              <View className='action-arrow'>
                <Text className='arrow-text text-muted'>→</Text>
              </View>
            </View>
          ))}
        </View>
      </View>

      {/* Achievement Card */}
      <View className='achievement-section container mt-xl mb-2xl'>
        <View className='achievement-card card p-xl'>
          <View className='achievement-header mb-lg'>
            <View className='achievement-icon bg-primary shape-circle'>
              <Text className='achievement-emoji'>🏆</Text>
            </View>
            <View className='achievement-content'>
              <Text className='achievement-title font-medium'>本周成就</Text>
              <Text className='achievement-subtitle text-muted'>
                恭喜您获得"效率达人"称号
              </Text>
            </View>
          </View>

          <View className='progress-section'>
            <View className='progress-header'>
              <Text className='progress-label text-muted'>下一等级进度</Text>
              <Text className='progress-value font-medium'>75%</Text>
            </View>
            <View className='progress-bar bg-muted'>
              <View className='progress-fill bg-primary' style={{ width: '75%' }}></View>
            </View>
          </View>

          <View className='achievement-rewards mt-lg'>
            <Text className='rewards-title font-medium mb-md'>本周奖励</Text>
            <View className='rewards-list flex gap-md'>
              <View className='reward-item'>
                <View className='reward-icon bg-secondary shape-circle'>
                  <Text className='reward-emoji'>⭐</Text>
                </View>
                <Text className='reward-text text-muted'>+50积分</Text>
              </View>
              <View className='reward-item'>
                <View className='reward-icon bg-accent shape-circle'>
                  <Text className='reward-emoji'>🎖</Text>
                </View>
                <Text className='reward-text text-muted'>新徽章</Text>
              </View>
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  )
}

export default Profile