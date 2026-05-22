export default defineAppConfig({
  pages: [
    "pages/home/index",           // 首页 - 教程列表
    "pages/course-detail/index",  // 教程详情页
    "pages/content-reader/index",  // 内容阅读页
    "pages/payment/index",        // 支付页面
  ],
  window: {
    backgroundTextStyle: "light",
    navigationBarBackgroundColor: "#667eea",
    navigationBarTitleText: "优学教程",
    navigationBarTextStyle: "white",
  },
  lazyCodeLoading: "requiredComponents",
});
