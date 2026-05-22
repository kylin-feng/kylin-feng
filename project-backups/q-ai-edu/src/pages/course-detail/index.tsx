import {
  View,
  Text,
  Image,
  Input,
  Button,
  ScrollView,
} from "@tarojs/components";
import { useLoad, useRouter } from "@tarojs/taro";
import "./index.scss";
import { useState, useEffect } from "react";
import Taro from "@tarojs/taro";

export default function CourseDetail() {
  const router = useRouter();
  const courseId = router.params.id;
  
  const [courseInfo, setCourseInfo] = useState<Course.CourseInfo | null>(null);
  const [activeTab, setActiveTab] = useState<'info' | 'chapters' | 'reviews'>('info');
  const [loading, setLoading] = useState(true);

  // 模拟课程数据
  const mockCourseInfo: Course.CourseInfo = {
    id: courseId || "1",
    title: "React全栈开发实战",
    description: "从零开始学习React，包含Hooks、Redux、Next.js等核心技术，通过实战项目掌握现代前端开发技能。本课程适合有一定JavaScript基础的开发者，将带你深入理解React生态系统。",
    cover: require("../../images/worker.jpg"),
    price: 199,
    originalPrice: 299,
    categoryId: "frontend",
    categoryName: "前端开发",
    teacherInfo: {
      id: "t1",
      name: "张晓明",
      avatar: require("../../images/scott.jpg"),
      title: "高级前端工程师 · 前端架构师",
      introduction: "5年大厂前端开发经验，曾就职于腾讯、字节跳动。精通React、Vue等前端技术栈。",
      experience: 5,
      studentCount: 1200,
      courseCount: 8
    },
    difficulty: "intermediate",
    duration: 480,
    studentCount: 856,
    rating: 4.8,
    tags: ["React", "JavaScript", "前端", "实战项目"],
    chapters: [
      {
        id: "c1",
        courseId: courseId || "1",
        title: "React基础入门",
        description: "学习React的基本概念和JSX语法",
        order: 1,
        duration: 45,
        videoUrl: "https://example.com/video1.mp4",
        isLocked: false,
        isTrial: true,
        isCompleted: false
      },
      {
        id: "c2",
        courseId: courseId || "1",
        title: "组件与Props",
        description: "深入理解React组件化开发",
        order: 2,
        duration: 38,
        videoUrl: "https://example.com/video2.mp4",
        isLocked: false,
        isTrial: true,
        isCompleted: false
      },
      {
        id: "c3",
        courseId: courseId || "1",
        title: "State与事件处理",
        description: "掌握状态管理和用户交互",
        order: 3,
        duration: 42,
        videoUrl: "https://example.com/video3.mp4",
        isLocked: true,
        isTrial: false,
        isCompleted: false
      },
      {
        id: "c4",
        courseId: courseId || "1",
        title: "React Hooks详解",
        description: "深入学习useState、useEffect等Hooks",
        order: 4,
        duration: 55,
        videoUrl: "https://example.com/video4.mp4",
        isLocked: true,
        isTrial: false,
        isCompleted: false
      },
      {
        id: "c5",
        courseId: courseId || "1",
        title: "Redux状态管理",
        description: "学习Redux进行全局状态管理",
        order: 5,
        duration: 68,
        videoUrl: "https://example.com/video5.mp4",
        isLocked: true,
        isTrial: false,
        isCompleted: false
      }
    ],
    createTime: "2024-01-15",
    updateTime: "2024-12-01",
    status: "published",
    highlights: [
      "🎯 5个实战项目从零到一",
      "💼 企业级开发规范和最佳实践",
      "🚀 React18新特性深度解析",
      "📱 响应式设计和移动端适配",
      "🔧 完整的工程化配置方案",
      "👨‍💻 一对一答疑指导"
    ],
    trialVideoUrl: "https://example.com/trial.mp4"
  };

  const mockReviews = [
    {
      id: "r1",
      userName: "前端小白",
      userAvatar: "/images/user1.jpg",
      rating: 5,
      content: "老师讲解非常清晰，项目实战很有帮助，推荐！",
      createTime: "2024-11-20"
    },
    {
      id: "r2", 
      userName: "React爱好者",
      userAvatar: "/images/user2.jpg",
      rating: 5,
      content: "从基础到进阶，内容很全面，值得购买。",
      createTime: "2024-11-18"
    },
    {
      id: "r3",
      userName: "全栈开发者",
      userAvatar: "/images/user3.jpg", 
      rating: 4,
      content: "项目案例很实用，对工作帮助很大。",
      createTime: "2024-11-15"
    }
  ];

  useLoad(() => {
    console.log("课程详情页加载, ID:", courseId);
    loadCourseInfo();
  });

  const loadCourseInfo = () => {
    setLoading(true);
    // 模拟API调用
    setTimeout(() => {
      setCourseInfo(mockCourseInfo);
      setLoading(false);
    }, 500);
  };

  const handleBuyCourse = () => {
    if (!courseInfo) return;
    
    Taro.navigateTo({
      url: `/pages/payment/index?courseId=${courseInfo.id}&price=${courseInfo.price}`
    });
  };

  const handleTrialRead = () => {
    if (!courseInfo) return;
    
    Taro.navigateTo({
      url: `/pages/content-reader/index?contentId=${courseInfo.id}&isTrial=true`
    });
  };

  const handleChapterRead = (chapter: Course.ChapterInfo) => {
    if (chapter.isLocked) {
      Taro.showToast({
        title: "请先购买课程",
        icon: "none"
      });
      return;
    }

    Taro.navigateTo({
      url: `/pages/content-reader/index?contentId=${chapter.id}&chapterId=${chapter.id}&isTrial=${chapter.isTrial}`
    });
  };

  const formatDuration = (minutes: number) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}h ${mins}min` : `${mins}min`;
  };

  const formatPrice = (price: number) => {
    return `¥${price}`;
  };

  const renderStars = (rating: number) => {
    const stars: JSX.Element[] = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;

    for (let i = 0; i < fullStars; i++) {
      stars.push(<Text key={i} className="star full">⭐</Text>);
    }
    if (hasHalfStar) {
      stars.push(<Text key="half" className="star half">⭐</Text>);
    }
    for (let i = stars.length; i < 5; i++) {
      stars.push(<Text key={i} className="star empty">☆</Text>);
    }

    return stars;
  };

  if (loading) {
    return (
      <View className="course-detail-page">
        <View className="loading-container">
          <Text>加载中...</Text>
        </View>
      </View>
    );
  }

  if (!courseInfo) {
    return (
      <View className="course-detail-page">
        <View className="error-container">
          <Text>课程不存在</Text>
        </View>
      </View>
    );
  }

  return (
    <View className="course-detail-page">
      {/* 课程封面区域 */}
      <View className="course-header">
        <Image
          className="course-cover"
          src={courseInfo.cover}
          mode="aspectFill"
        />
        <View className="course-overlay">
          <View className="course-basic-info">
            <Text className="course-title">{courseInfo.title}</Text>
            <View className="course-meta">
              <Text className="student-count">{courseInfo.studentCount}人学习</Text>
              <View className="rating-info">
                {renderStars(courseInfo.rating)}
                <Text className="rating-num">{courseInfo.rating}</Text>
              </View>
            </View>
            <View className="course-tags">
              {courseInfo.tags.slice(0, 4).map(tag => (
                <Text key={tag} className="tag">{tag}</Text>
              ))}
            </View>
          </View>
          <Button className="trial-btn" onClick={handleTrialRead}>
            📖 免费试读
          </Button>
        </View>
      </View>

      {/* 价格信息 */}
      <View className="price-section">
        <View className="price-info">
          <Text className="current-price">{formatPrice(courseInfo.price)}</Text>
          {courseInfo.originalPrice && (
            <Text className="original-price">{formatPrice(courseInfo.originalPrice)}</Text>
          )}
          <View className="discount-info">
            {courseInfo.originalPrice && (
              <Text className="discount-text">
                限时特惠 省{courseInfo.originalPrice - courseInfo.price}元
            </Text>
            )}
          </View>
        </View>
        <Button className="buy-btn" onClick={handleBuyCourse}>
          立即购买
        </Button>
        </View>

      {/* 标签页导航 */}
      <View className="tab-nav">
        <Text
          className={`tab-item ${activeTab === 'info' ? 'active' : ''}`}
          onClick={() => setActiveTab('info')}
        >
          课程介绍
        </Text>
        <Text
          className={`tab-item ${activeTab === 'chapters' ? 'active' : ''}`}
          onClick={() => setActiveTab('chapters')}
        >
          课程目录
        </Text>
        <Text
          className={`tab-item ${activeTab === 'reviews' ? 'active' : ''}`}
          onClick={() => setActiveTab('reviews')}
        >
          学员评价
        </Text>
      </View>

      {/* 标签页内容 */}
      <ScrollView className="tab-content" scrollY>
        {activeTab === 'info' && (
          <View className="info-content">
            {/* 讲师信息 */}
            <View className="teacher-section">
              <Text className="section-title">讲师介绍</Text>
              <View className="teacher-card">
                <Image
                  className="teacher-avatar"
                  src={courseInfo.teacherInfo.avatar}
                  mode="aspectFill"
                />
                <View className="teacher-info">
                  <Text className="teacher-name">{courseInfo.teacherInfo.name}</Text>
                  <Text className="teacher-title">{courseInfo.teacherInfo.title}</Text>
                  <Text className="teacher-intro">{courseInfo.teacherInfo.introduction}</Text>
                  <View className="teacher-stats">
                    <Text className="stat-item">{courseInfo.teacherInfo.experience}年经验</Text>
                    <Text className="stat-item">{courseInfo.teacherInfo.studentCount}学员</Text>
                    <Text className="stat-item">{courseInfo.teacherInfo.courseCount}门课程</Text>
                  </View>
                </View>
              </View>
            </View>

            {/* 课程亮点 */}
            <View className="highlights-section">
              <Text className="section-title">课程亮点</Text>
              <View className="highlights-list">
                {courseInfo.highlights.map((highlight, index) => (
                  <Text key={index} className="highlight-item">{highlight}</Text>
          ))}
        </View>
      </View>

            {/* 课程描述 */}
            <View className="description-section">
              <Text className="section-title">课程详情</Text>
              <Text className="course-description">{courseInfo.description}</Text>
            </View>
          </View>
        )}

        {activeTab === 'chapters' && (
          <View className="chapters-content">
            <View className="chapters-header">
              <Text className="chapters-title">课程目录</Text>
              <Text className="chapters-info">
                共{courseInfo.chapters.length}节课 · {formatDuration(courseInfo.duration)}
              </Text>
            </View>
            <View className="chapters-list">
              {courseInfo.chapters.map((chapter, index) => (
                <View
                  key={chapter.id}
                  className={`chapter-item ${chapter.isLocked ? 'locked' : ''}`}
                  onClick={() => handleChapterRead(chapter)}
                >
                  <View className="chapter-number">
                    {chapter.isLocked ? '🔒' : index + 1}
                  </View>
                  <View className="chapter-info">
                    <Text className="chapter-title">{chapter.title}</Text>
                    <Text className="chapter-desc">{chapter.description}</Text>
                    <View className="chapter-meta">
                      <Text className="chapter-duration">{formatDuration(chapter.duration)}</Text>
                    {chapter.isTrial && (
                      <Text className="trial-badge">试读</Text>
                    )}
                    </View>
                  </View>
                  <View className="chapter-action">
                    {chapter.isLocked ? (
                      <Text className="lock-text">需购买</Text>
                    ) : (
                      <Text className="play-icon">📖</Text>
                    )}
                  </View>
                </View>
              ))}
            </View>
          </View>
        )}

        {activeTab === 'reviews' && (
          <View className="reviews-content">
            <View className="reviews-header">
              <Text className="reviews-title">学员评价</Text>
              <View className="reviews-stats">
                <Text className="reviews-average">{courseInfo.rating}分</Text>
                <View className="reviews-stars">
                  {renderStars(courseInfo.rating)}
                </View>
              </View>
            </View>
            <View className="reviews-list">
              {mockReviews.map(review => (
                <View key={review.id} className="review-item">
                  <Image
                    className="review-avatar"
                    src={review.userAvatar}
                    mode="aspectFill"
                  />
                  <View className="review-content">
                    <View className="review-header">
                      <Text className="review-username">{review.userName}</Text>
                      <View className="review-rating">
                        {renderStars(review.rating)}
                      </View>
                    </View>
                    <Text className="review-text">{review.content}</Text>
                    <Text className="review-time">{review.createTime}</Text>
                  </View>
          </View>
              ))}
            </View>
          </View>
        )}
      </ScrollView>

      {/* 底部购买按钮 */}
      <View className="bottom-bar">
        <View className="bottom-price">
          <Text className="bottom-current-price">{formatPrice(courseInfo.price)}</Text>
          {courseInfo.originalPrice && (
            <Text className="bottom-original-price">{formatPrice(courseInfo.originalPrice)}</Text>
          )}
        </View>
        <Button className="bottom-buy-btn" onClick={handleBuyCourse}>
          立即购买
        </Button>
      </View>
    </View>
  );
}
