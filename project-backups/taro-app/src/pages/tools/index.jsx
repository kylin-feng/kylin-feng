import React, { useState, useEffect, useRef } from 'react'
import { View, Text, ScrollView, Input } from '@tarojs/components'
import Taro from '@tarojs/taro'
import './index.scss'

const Tools = () => {
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [animationClass, setAnimationClass] = useState('')
  const [isSearchFocused, setIsSearchFocused] = useState(false)

  useEffect(() => {
    setTimeout(() => {
      setAnimationClass('animate-fade-in')
    }, 100)
  }, [])

  const categories = [
    { id: 'all', name: '全部', count: 12 },
    { id: 'coding', name: '编程', count: 4 },
    { id: 'writing', name: '写作', count: 3 },
    { id: 'design', name: '设计', count: 2 },
    { id: 'analysis', name: '分析', count: 3 }
  ]

  const tools = [
    {
      id: 1,
      name: 'AI代码助手',
      description: '智能代码生成与优化',
      category: 'coding',
      icon: '💻',
      color: '#3b82f6',
      usage: '1.2k',
      rating: 4.9,
      tags: ['Python', 'JavaScript', 'React'],
      isHot: true
    },
    {
      id: 2,
      name: '智能写作',
      description: 'AI辅助内容创作',
      category: 'writing',
      icon: '✍️',
      color: '#10b981',
      usage: '856',
      rating: 4.7,
      tags: ['文案', '博客', 'SEO'],
      isNew: true
    },
    {
      id: 3,
      name: '设计灵感',
      description: '创意设计生成器',
      category: 'design',
      icon: '🎨',
      color: '#8b5cf6',
      usage: '632',
      rating: 4.8,
      tags: ['UI', 'Logo', '配色'],
      isHot: false
    },
    {
      id: 4,
      name: '数据分析',
      description: '智能数据洞察',
      category: 'analysis',
      icon: '📊',
      color: '#f59e0b',
      usage: '945',
      rating: 4.6,
      tags: ['报表', '可视化', '预测'],
      isNew: false
    },
    {
      id: 5,
      name: 'API调试',
      description: '智能接口测试',
      category: 'coding',
      icon: '🔧',
      color: '#ef4444',
      usage: '723',
      rating: 4.5,
      tags: ['REST', 'GraphQL', '测试'],
      isHot: false
    },
    {
      id: 6,
      name: '翻译助手',
      description: '多语言智能翻译',
      category: 'writing',
      icon: '🌐',
      color: '#06b6d4',
      usage: '1.5k',
      rating: 4.8,
      tags: ['英文', '技术', '学术'],
      isNew: false
    }
  ]

  const filteredTools = tools.filter(tool => {
    const matchesCategory = selectedCategory === 'all' || tool.category === selectedCategory
    const matchesSearch = tool.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         tool.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         tool.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
    return matchesCategory && matchesSearch
  })

  const handleToolClick = (tool) => {
    Taro.showToast({
      title: `启动 ${tool.name}`,
      icon: 'success',
      duration: 1500
    })
  }

  const handleSearchFocus = () => {
    setIsSearchFocused(true)
  }

  const handleSearchBlur = () => {
    setIsSearchFocused(false)
  }

  return (
    <ScrollView className='tools-container' scrollY>
      {/* Header */}
      <View className={`tools-header ${animationClass}`}>
        <View className='container'>
          <View className='header-content'>
            <Text className='header-title font-light'>AI工具链</Text>
            <Text className='header-subtitle text-muted'>
              提升工作效率的智能工具集
            </Text>
          </View>

          {/* Search */}
          <View className={`search-container ${isSearchFocused ? 'focused' : ''}`}>
            <View className='search-input-wrapper'>
              <View className='search-icon'>🔍</View>
              <Input
                className='search-input'
                placeholder='搜索工具...'
                value={searchQuery}
                onInput={(e) => setSearchQuery(e.detail.value)}
                onFocus={handleSearchFocus}
                onBlur={handleSearchBlur}
              />
              {searchQuery && (
                <View 
                  className='search-clear'
                  onClick={() => setSearchQuery('')}
                >
                  ✕
                </View>
              )}
            </View>
          </View>
        </View>
      </View>

      {/* Categories */}
      <View className='categories-section container'>
        <ScrollView className='categories-scroll' scrollX showScrollbar={false}>
          <View className='categories-list'>
            {categories.map((category, index) => (
              <View
                key={category.id}
                className={`category-item ${selectedCategory === category.id ? 'active' : ''} hover-scale`}
                style={{ animationDelay: `${index * 0.1}s` }}
                onClick={() => setSelectedCategory(category.id)}
              >
                <Text className='category-name'>{category.name}</Text>
                <View className='category-count'>
                  <Text className='count-text'>{category.count}</Text>
                </View>
              </View>
            ))}
          </View>
        </ScrollView>
      </View>

      {/* Tools Grid */}
      <View className='tools-content container'>
        <View className='tools-grid'>
          {filteredTools.map((tool, index) => (
            <View
              key={tool.id}
              className={`tool-card card hover-lift animate-scale-in`}
              style={{ animationDelay: `${index * 0.1}s` }}
              onClick={() => handleToolClick(tool)}
            >
              {/* Card Header */}
              <View className='tool-header'>
                <View 
                  className='tool-icon shape-rounded-lg'
                  style={{ backgroundColor: tool.color }}
                >
                  <Text className='icon-emoji'>{tool.icon}</Text>
                </View>

                {/* Badges */}
                <View className='tool-badges'>
                  {tool.isHot && (
                    <View className='badge badge-hot'>
                      <Text className='badge-text'>🔥 热门</Text>
                    </View>
                  )}
                  {tool.isNew && (
                    <View className='badge badge-new'>
                      <Text className='badge-text'>✨ 新品</Text>
                    </View>
                  )}
                </View>
              </View>

              {/* Card Content */}
              <View className='tool-content'>
                <Text className='tool-name font-medium'>{tool.name}</Text>
                <Text className='tool-description text-muted'>{tool.description}</Text>
                
                {/* Tags */}
                <View className='tool-tags flex gap-xs mt-sm'>
                  {tool.tags.slice(0, 2).map((tag, tagIndex) => (
                    <View key={tagIndex} className='tag bg-muted'>
                      <Text className='tag-text'>{tag}</Text>
                    </View>
                  ))}
                  {tool.tags.length > 2 && (
                    <View className='tag bg-muted'>
                      <Text className='tag-text'>+{tool.tags.length - 2}</Text>
                    </View>
                  )}
                </View>
              </View>

              {/* Card Footer */}
              <View className='tool-footer'>
                <View className='tool-stats'>
                  <View className='stat-item'>
                    <Text className='stat-icon'>👥</Text>
                    <Text className='stat-text text-muted'>{tool.usage}</Text>
                  </View>
                  <View className='stat-item'>
                    <Text className='stat-icon'>⭐</Text>
                    <Text className='stat-text text-muted'>{tool.rating}</Text>
                  </View>
                </View>
                
                <View className='tool-action'>
                  <View className='action-btn bg-primary shape-rounded'>
                    <Text className='action-text'>启动</Text>
                  </View>
                </View>
              </View>

              {/* Hover Effect */}
              <View className='card-glow' style={{ backgroundColor: tool.color }}></View>
            </View>
          ))}
        </View>

        {/* Empty State */}
        {filteredTools.length === 0 && (
          <View className='empty-state'>
            <View className='empty-icon'>🔍</View>
            <Text className='empty-title font-medium'>未找到相关工具</Text>
            <Text className='empty-subtitle text-muted'>
              尝试调整搜索关键词或选择其他分类
            </Text>
          </View>
        )}
      </View>

      {/* Quick Actions */}
      <View className='quick-actions container mt-2xl mb-2xl'>
        <View className='actions-card card p-xl'>
          <Text className='actions-title font-medium mb-lg text-center'>
            快捷操作
          </Text>
          <View className='actions-grid'>
            <View className='action-item hover-scale'>
              <View className='action-icon bg-primary shape-circle'>
                <Text className='action-emoji'>➕</Text>
              </View>
              <Text className='action-label text-muted'>自定义工具</Text>
            </View>
            <View className='action-item hover-scale'>
              <View className='action-icon bg-secondary shape-circle'>
                <Text className='action-emoji'>📝</Text>
              </View>
              <Text className='action-label text-muted'>使用记录</Text>
            </View>
            <View className='action-item hover-scale'>
              <View className='action-icon bg-accent shape-circle'>
                <Text className='action-emoji'>⚙️</Text>
              </View>
              <Text className='action-label text-muted'>工具设置</Text>
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  )
}

export default Tools