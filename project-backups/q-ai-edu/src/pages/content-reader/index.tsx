import {
  View,
  Text,
  Image,
  Button,
  ScrollView,
  Audio,
} from "@tarojs/components";
import { useLoad, useRouter } from "@tarojs/taro";
import Taro from "@tarojs/taro";
import { useState, useRef, useEffect } from "react";
import "./index.scss";

export default function ContentReader() {
  const router = useRouter();
  const { contentId, chapterId, isTrial } = router.params;
  const audioRef = useRef<any>(null);
  
  const [contentInfo, setContentInfo] = useState<Course.ContentInfo | null>(null);
  const [currentSection, setCurrentSection] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioProgress, setAudioProgress] = useState(0);
  const [audioDuration, setAudioDuration] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showTrialModal, setShowTrialModal] = useState(false);

  // 模拟内容数据
  const mockContentInfo: Course.ContentInfo = {
    id: contentId || "1",
    title: "React基础入门",
    description: "学习React的基本概念和JSX语法",
    type: "text", // text, image, audio
    sections: [
      {
        id: "s1",
        type: "text",
        title: "什么是React？",
        content: "React是一个用于构建用户界面的JavaScript库。它由Facebook开发，现在由社区维护。React的主要特点是：\n\n1. 组件化开发\n2. 虚拟DOM\n3. 单向数据流\n4. JSX语法\n\nReact让开发者能够构建大型、快速的Web应用，并且能够很好地与其他库或框架配合使用。",
        order: 1
      },
      {
        id: "s2", 
        type: "image",
        title: "React组件结构图",
        content: "https://example.com/react-structure.png",
        description: "这张图展示了React组件的基本结构，包括props、state和生命周期方法。",
        order: 2
      },
      {
        id: "s3",
        type: "audio",
        title: "JSX语法详解",
        content: "https://example.com/jsx-audio.mp3",
        duration: 180, // 3分钟
        description: "音频讲解JSX语法的基本用法和注意事项",
        order: 3
      },
      {
        id: "s4",
        type: "text",
        title: "组件生命周期",
        content: "React组件有三个主要生命周期阶段：\n\n**挂载阶段：**\n- constructor()\n- componentDidMount()\n\n**更新阶段：**\n- componentDidUpdate()\n- shouldComponentUpdate()\n\n**卸载阶段：**\n- componentWillUnmount()\n\n每个生命周期方法都有其特定的用途，理解它们对于编写高效的React应用至关重要。",
        order: 4
      },
      {
        id: "s5",
        type: "image",
        title: "生命周期流程图",
        content: "https://example.com/lifecycle-diagram.png", 
        description: "React组件生命周期的完整流程图，展示了从创建到销毁的整个过程。",
        order: 5
      }
    ],
    isLocked: isTrial === 'true' ? false : true,
    trialSections: 2, // 前2个章节可以试看
    totalSections: 5
  };

  useLoad(() => {
    console.log("内容阅读页加载");
    console.log("参数:", { contentId, chapterId, isTrial });
    loadContentInfo();
  });

  const loadContentInfo = () => {
    setLoading(true);
    setTimeout(() => {
      setContentInfo(mockContentInfo);
      setLoading(false);
    }, 500);
  };

  const handleSectionChange = (index: number) => {
    if (isTrial === 'true' && index >= mockContentInfo.trialSections) {
      setShowTrialModal(true);
      return;
    }
    setCurrentSection(index);
  };

  const handleAudioPlay = () => {
    if (isTrial === 'true' && currentSection >= mockContentInfo.trialSections) {
      setShowTrialModal(true);
      return;
    }
    
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
    }
  };

  const handleAudioTimeUpdate = (e: any) => {
    setAudioProgress(e.detail.currentTime);
  };

  const handleAudioLoadedMetadata = (e: any) => {
    setAudioDuration(e.detail.duration);
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
    setAudioProgress(0);
  };

  const handleBuyCourse = () => {
    Taro.navigateTo({
      url: '/pages/payment/index?courseId=1&price=199'
    });
  };

  const handleBack = () => {
    Taro.navigateBack();
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const renderTextContent = (section: Course.ContentSection) => {
    return (
      <View className="text-content">
        <Text className="section-title">{section.title}</Text>
        <Text className="section-text">{section.content}</Text>
      </View>
    );
  };

  const renderImageContent = (section: Course.ContentSection) => {
    return (
      <View className="image-content">
        <Text className="section-title">{section.title}</Text>
        <Image
          className="content-image"
          src={section.content}
          mode="widthFix"
          onError={() => {
            Taro.showToast({
              title: "图片加载失败",
              icon: "none"
            });
          }}
        />
        {section.description && (
          <Text className="image-description">{section.description}</Text>
        )}
      </View>
    );
  };

  const renderAudioContent = (section: Course.ContentSection) => {
    return (
      <View className="audio-content">
        <Text className="section-title">{section.title}</Text>
        {section.description && (
          <Text className="audio-description">{section.description}</Text>
        )}
        
        <View className="audio-player">
          <Audio
            ref={audioRef}
            src={section.content}
            onTimeUpdate={handleAudioTimeUpdate}
            onLoadedMetadata={handleAudioLoadedMetadata}
            onEnded={handleAudioEnded}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          />
          
          <View className="audio-controls">
            <Button 
              className="play-btn"
              onClick={handleAudioPlay}
            >
              {isPlaying ? '⏸️' : '▶️'}
            </Button>
            
            <View className="progress-info">
              <Text className="time-current">{formatTime(audioProgress)}</Text>
              <View className="progress-bar">
                <View 
                  className="progress-fill"
                  style={{ width: `${(audioProgress / audioDuration) * 100}%` }}
                />
              </View>
              <Text className="time-total">{formatTime(audioDuration)}</Text>
            </View>
          </View>
        </View>
      </View>
    );
  };

  const renderContent = (section: Course.ContentSection) => {
    switch (section.type) {
      case 'text':
        return renderTextContent(section);
      case 'image':
        return renderImageContent(section);
      case 'audio':
        return renderAudioContent(section);
      default:
        return null;
    }
  };

  if (loading) {
    return (
      <View className="content-reader-page">
        <View className="loading-container">
          <Text>内容加载中...</Text>
        </View>
      </View>
    );
  }

  if (!contentInfo) {
    return (
      <View className="content-reader-page">
        <View className="error-container">
          <Text>内容不存在</Text>
        </View>
      </View>
    );
  }

  return (
    <View className="content-reader-page">
      {/* 顶部导航 */}
      <View className="top-nav">
        <Button className="back-btn" onClick={handleBack}>
          ← 返回
        </Button>
        <Text className="page-title">{contentInfo.title}</Text>
        <View className="nav-spacer" />
      </View>

      {/* 内容区域 */}
      <ScrollView className="content-area" scrollY>
        {/* 当前章节内容 */}
        <View className="current-content">
          {renderContent(contentInfo.sections[currentSection])}
        </View>

        {/* 章节导航 */}
        <View className="sections-nav">
          <Text className="nav-title">章节列表</Text>
          <View className="sections-list">
            {contentInfo.sections.map((section, index) => (
              <View
                key={section.id}
                className={`section-item ${currentSection === index ? 'active' : ''} ${isTrial === 'true' && index >= contentInfo.trialSections ? 'locked' : ''}`}
                onClick={() => handleSectionChange(index)}
              >
                <View className="section-info">
                  <Text className="section-number">{index + 1}</Text>
                  <Text className="section-title">{section.title}</Text>
                  <Text className="section-type">
                    {section.type === 'text' ? '📄 文本' : 
                     section.type === 'image' ? '🖼️ 图片' : 
                     '🎵 语音'}
                  </Text>
                </View>
                <View className="section-status">
                  {isTrial === 'true' && index >= contentInfo.trialSections ? (
                    <Text className="lock-icon">🔒</Text>
                  ) : (
                    <Text className="play-icon">▶️</Text>
                  )}
                </View>
              </View>
            ))}
          </View>
        </View>
      </ScrollView>

      {/* 试看结束弹窗 */}
      {showTrialModal && (
        <View className="trial-modal-overlay">
          <View className="trial-modal">
            <Text className="modal-title">试看结束</Text>
            <Text className="modal-content">
              您已看完试看内容，购买完整课程继续学习吧！
            </Text>
            <View className="modal-buttons">
              <Button className="cancel-btn" onClick={handleBack}>
                返回
              </Button>
              <Button className="buy-btn" onClick={handleBuyCourse}>
                立即购买
              </Button>
            </View>
          </View>
        </View>
      )}

      {/* 底部操作栏 */}
      <View className="bottom-bar">
        <View className="progress-info">
          <Text className="progress-text">
            {currentSection + 1} / {contentInfo.totalSections}
          </Text>
          <View className="progress-bar">
            <View 
              className="progress-fill"
              style={{ width: `${((currentSection + 1) / contentInfo.totalSections) * 100}%` }}
            />
          </View>
        </View>
        
        {isTrial === 'true' && (
          <Button className="purchase-btn" onClick={handleBuyCourse}>
            购买完整课程
          </Button>
        )}
      </View>
    </View>
  );
}
