export default defineAppConfig({
  pages: [
    "pages/home/index",           // 首页 - 个人介绍
    "pages/showcase/index",       // 项目展示
    "pages/services/index",        // 服务介绍
    "pages/contact/index",         // 联系我
  ],
  window: {
    backgroundTextStyle: "light",
    navigationBarBackgroundColor: "#667eea",
    navigationBarTitleText: "小程序开发服务",
    navigationBarTextStyle: "white",
  },
  tabBar: {
    color: "#666666",
    selectedColor: "#667eea",
    backgroundColor: "#ffffff",
    borderStyle: "black",
    list: [
      {
        pagePath: "pages/home/index",
        text: "首页",
        iconPath: "./images/love.png",
        selectedIconPath: "./images/love.png",
      },
      {
        pagePath: "pages/showcase/index",
        text: "项目展示",
        iconPath: "./images/robot.png",
        selectedIconPath: "./images/robot.png",
      },
      {
        pagePath: "pages/services/index",
        text: "服务介绍",
        iconPath: "./images/send.png",
        selectedIconPath: "./images/send.png",
      },
      {
        pagePath: "pages/contact/index",
        text: "联系我",
        iconPath: "./images/love.png",
        selectedIconPath: "./images/love.png",
      },
    ],
  },
  lazyCodeLoading: "requiredComponents",
});
