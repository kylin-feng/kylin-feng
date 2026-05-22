import { View, Text, Image, Input, ScrollView } from "@tarojs/components";
import { useLoad } from "@tarojs/taro";
import Taro from "@tarojs/taro";
import { useState } from "react";
import "./index.scss";

export default function Index() {
  // const { isLoggedIn, userInfo } = useUser();
  const [searchKeyword, setSearchKeyword] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [sortBy, setSortBy] = useState<Course.SearchParams['sortBy']>("latest");
  const [courses, setCourses] = useState<Course.CourseInfo[]>([]);
  const [categories, setCategories] = useState<Course.CategoryInfo[]>([]);
  const [loading, setLoading] = useState(false);

  // 模拟数据
  const mockCategories: Course.CategoryInfo[] = [
    { id: "all", name: "全部", icon: "", courseCount: 0 },
    { id: "frontend", name: "前端开发", icon: "🎨", courseCount: 25 },
    { id: "backend", name: "后端开发", icon: "⚙️", courseCount: 18 },
    { id: "mobile", name: "移动开发", icon: "📱", courseCount: 12 },
    { id: "ai", name: "人工智能", icon: "🤖", courseCount: 8 },
    { id: "design", name: "UI设计", icon: "🎭", courseCount: 15 },
  ];

  const mockCourses: Course.CourseInfo[] = [
    {
      id: "1",
      title: "React全栈开发实战",
      description: "从零开始学习React，包含Hooks、Redux、Next.js等核心技术",
      cover: require("../../images/worker.jpg"),
      price: 199,
      originalPrice: 299,
      categoryId: "frontend",
      categoryName: "前端开发",
      teacherInfo: {
        id: "t1",
        name: "张老师",
        avatar: require("../../images/scott.jpg"),
        title: "高级前端工程师",
        introduction: "5年前端开发经验",
        experience: 5,
        studentCount: 1200,
        courseCount: 8
      },
      difficulty: "intermediate",
      duration: 480,
      studentCount: 856,
      rating: 4.8,
      tags: ["React", "JavaScript", "前端"],
      chapters: [],
      createTime: "2024-01-15",
      updateTime: "2024-12-01",
      status: "published",
      highlights: ["实战项目", "源码分析", "就业指导"]
    },
    {
      id: "2",
      title: "Vue3 + TypeScript企业级项目",
      description: "深入学习Vue3 Composition API，结合TypeScript开发企业级应用",
      cover: require("../../images/student.jpg"),
      price: 159,
      originalPrice: 239,
      categoryId: "frontend",
      categoryName: "前端开发",
      teacherInfo: {
        id: "t2",
        name: "李老师",
        avatar: require("../../images/worker.jpg"),
        title: "前端架构师",
        introduction: "Vue生态资深专家",
        experience: 6,
        studentCount: 980,
        courseCount: 5
      },
      difficulty: "advanced",
      duration: 520,
      studentCount: 642,
      rating: 4.9,
      tags: ["Vue3", "TypeScript", "前端"],
      chapters: [],
      createTime: "2024-02-20",
      updateTime: "2024-11-25",
      status: "published",
      highlights: ["Vue3新特性", "TypeScript实战", "性能优化"]
    },
    {
      id: "3",
      title: "Node.js后端开发从入门到精通",
      description: "掌握Node.js后端开发，包含Express、MongoDB、Redis等技术栈",
      cover: require("../../images/trans.jpg"),
      price: 179,
      categoryId: "backend",
      categoryName: "后端开发",
      teacherInfo: {
        id: "t3",
        name: "王老师",
        avatar: require("../../images/student.jpg"),
        title: "后端技术专家",
        introduction: "Node.js技术布道师",
        experience: 7,
        studentCount: 760,
        courseCount: 12
      },
      difficulty: "beginner",
      duration: 680,
      studentCount: 423,
      rating: 4.7,
      tags: ["Node.js", "Express", "MongoDB"],
      chapters: [],
      createTime: "2024-03-10",
      updateTime: "2024-12-05",
      status: "published",
      highlights: ["项目实战", "部署上线", "性能调优"]
    }
  ];

  useLoad(() => {
    console.log("首页加载");
    loadCategories();
    loadCourses();
  });

  const loadCategories = () => {
    setCategories(mockCategories);
  };

  const loadCourses = () => {
    setLoading(true);
    // 模拟API调用
    setTimeout(() => {
      setCourses(mockCourses);
      setLoading(false);
    }, 500);
  };

  const handleSearch = (value: string) => {
    setSearchKeyword(value);
    // 这里可以添加搜索逻辑
  };

  const handleCategoryChange = (categoryId: string) => {
    setSelectedCategory(categoryId);
    // 这里可以添加分类筛选逻辑
  };

  const handleSortChange = (sort: Course.SearchParams['sortBy']) => {
    setSortBy(sort);
    // 这里可以添加排序逻辑
  };

  const goToCourseDetail = (courseId: string) => {
    Taro.navigateTo({
      url: `/pages/course-detail/index?id=${courseId}`
    });
  };

  const formatPrice = (price: number) => {
    return `¥${price}`;
  };

  const formatStudentCount = (count: number) => {
    if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}k`;
    }
    return count.toString();
  };

  return (
    <View className="home-page">
      {/* 搜索栏 */}
      <View className="search-bar">
        <View className="search-input-wrapper">
          <Input
            className="search-input"
            placeholder="搜索教程..."
            value={searchKeyword}
            onInput={(e) => handleSearch(e.detail.value)}
          />
          <View className="search-icon">🔍</View>
        </View>
      </View>

      {/* 分类导航 */}
      <ScrollView className="category-nav" scrollX>
        <View className="category-list">
          {categories.map(category => (
            <View
              key={category.id}
              className={`category-item ${selectedCategory === category.id ? 'active' : ''}`}
              onClick={() => handleCategoryChange(category.id)}
            >
              <Text className="category-icon">{category.icon}</Text>
              <Text className="category-name">{category.name}</Text>
            </View>
          ))}
        </View>
      </ScrollView>

      {/* 排序选项 */}
      <View className="sort-bar">
        <View className="sort-options">
          <Text
            className={`sort-option ${sortBy === 'latest' ? 'active' : ''}`}
            onClick={() => handleSortChange('latest')}
          >
            最新
          </Text>
          <Text
            className={`sort-option ${sortBy === 'popular' ? 'active' : ''}`}
            onClick={() => handleSortChange('popular')}
          >
            热门
          </Text>
          <Text
            className={`sort-option ${sortBy === 'rating' ? 'active' : ''}`}
            onClick={() => handleSortChange('rating')}
          >
            好评
          </Text>
          <Text
            className={`sort-option ${sortBy === 'price-asc' ? 'active' : ''}`}
            onClick={() => handleSortChange('price-asc')}
          >
            价格↑
          </Text>
        </View>
      </View>

      {/* 教程列表 */}
      <ScrollView className="course-list" scrollY>
        {loading ? (
          <View className="loading">
            <Text>加载中...</Text>
          </View>
        ) : (
          courses.map(course => (
            <View
              key={course.id}
              className="course-card"
              onClick={() => goToCourseDetail(course.id)}
            >
              <View className="course-cover">
                <Image
                  className="cover-image"
                  src={course.cover}
                  mode="aspectFill"
                />
                {course.originalPrice && (
                  <View className="discount-badge">
                    <Text>特惠</Text>
                  </View>
                )}
              </View>
              
              <View className="course-info">
                <Text className="course-title">{course.title}</Text>
                <Text className="course-desc">{course.description}</Text>
                
                <View className="course-meta">
                  <View className="teacher-info">
                    <Image
                      className="teacher-avatar"
                      src={course.teacherInfo.avatar}
                      mode="aspectFill"
                    />
                    <Text className="teacher-name">{course.teacherInfo.name}</Text>
                  </View>
                  <View className="course-stats">
                    <Text className="student-count">
                      {formatStudentCount(course.studentCount)}人学习
                    </Text>
                    <Text className="rating">⭐ {course.rating}</Text>
                  </View>
                </View>

                <View className="course-tags">
                  {course.tags.slice(0, 3).map(tag => (
                    <Text key={tag} className="tag">{tag}</Text>
                  ))}
                </View>

                <View className="course-footer">
                  <View className="price-info">
                    <Text className="current-price">{formatPrice(course.price)}</Text>
                    {course.originalPrice && (
                      <Text className="original-price">{formatPrice(course.originalPrice)}</Text>
                    )}
                  </View>
                  <View className="course-duration">
                    <Text>{Math.floor(course.duration / 60)}h {course.duration % 60}min</Text>
                  </View>
                </View>
              </View>
            </View>
          ))
        )}
      </ScrollView>
    </View>
  );
}
