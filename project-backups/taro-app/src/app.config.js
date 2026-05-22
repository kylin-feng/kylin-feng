export default defineAppConfig({
  pages: [
    'pages/index/index',
    'pages/research/index',
    'pages/tools/index',
    'pages/profile/index'
  ],
  window: {
    backgroundTextStyle: 'light',
    navigationBarBackgroundColor: '#fff',
    navigationBarTitleText: 'Q.AI',
    navigationBarTextStyle: 'black'
  },
  tabBar: {
    custom: false,
    color: '#666',
    selectedColor: '#222',
    backgroundColor: '#fff',
    borderStyle: 'white',
    list: [
      {
        pagePath: 'pages/index/index',
        iconPath: './assets/icons/home.png',
        selectedIconPath: './assets/icons/home-active.png',
        text: '首页'
      },
      {
        pagePath: 'pages/research/index',
        iconPath: './assets/icons/research.png',
        selectedIconPath: './assets/icons/research-active.png',
        text: '研究'
      },
      {
        pagePath: 'pages/tools/index',
        iconPath: './assets/icons/tools.png',
        selectedIconPath: './assets/icons/tools-active.png',
        text: '工具'
      },
      {
        pagePath: 'pages/profile/index',
        iconPath: './assets/icons/profile.png',
        selectedIconPath: './assets/icons/profile-active.png',
        text: '我的'
      }
    ]
  }
})