import router from '@ohos.router';

@Entry
@Component
struct ReaderPage {
  @State content: string = '';
  @State title: string = '';
  @State computingPower: number = 0;
  @State isUnlocked: boolean = false;
  @State currentPage: number = 0;
  @State totalPages: number = 0;
  @State pages: string[] = [];

  aboutToAppear() {
    const params = router.getParams() as Record<string, Object>;
    this.content = params['content'] as string || '';
    this.title = params['title'] as string || '';
    this.computingPower = params['computingPower'] as number || 0;
    
    // 分页处理
    this.splitContentIntoPages();
  }

  private splitContentIntoPages() {
    const wordsPerPage = 200; // 每页200字
    const words = this.content.split('');
    this.pages = [];
    
    for (let i = 0; i < words.length; i += wordsPerPage) {
      const pageContent = words.slice(i, i + wordsPerPage).join('');
      this.pages.push(pageContent);
    }
    
    this.totalPages = this.pages.length;
  }

  build() {
    Column({ space: 0 }) {
      // 顶部导航栏
      Row() {
        Button('返回上层')
          .fontSize(16)
          .fontColor('#007AFF')
          .backgroundColor(Color.Transparent)
          .border({ width: 0 })
          .onClick(() => {
            router.back();
          })
      }
      .width('100%')
      .height(60)
      .padding({ left: 20, right: 20 })
      .justifyContent(FlexAlign.Start)
      .backgroundColor('#F8F9FA')

      // 主要内容区域
      if (this.isUnlocked || this.computingPower === 0) {
        // 已解锁，显示完整内容
        Column({ space: 20 }) {
          Text(this.title)
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor('#333333')
            .textAlign(TextAlign.Center)
            .margin({ top: 20 })

          // 小说内容显示区域
          Scroll() {
            Column({ space: 15 }) {
              Text(this.content)
                .fontSize(16)
                .fontColor('#333333')
                .lineHeight(24)
                .width('100%')
            }
            .padding(20)
          }
          .width('100%')
          .height('100%')
          .backgroundColor('#FFFFFF')
          .border({ width: 1, color: '#E0E0E0' })
          .borderRadius(8)
          .margin({ left: 20, right: 20 })
        }
      } else {
        // 未解锁，显示预览和付费提示
        Column({ space: 20 }) {
          Text(this.title)
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor('#333333')
            .textAlign(TextAlign.Center)
            .margin({ top: 20 })

          // 预览内容区域
          Column() {
            Text('滑动观看')
              .fontSize(16)
              .fontColor('#666666')
              .textAlign(TextAlign.Center)
          }
          .width('100%')
          .height(300)
          .backgroundColor('#FFFFFF')
          .border({ width: 1, color: '#000000' })
          .borderRadius(8)
          .justifyContent(FlexAlign.Center)
          .margin({ left: 20, right: 20 })

          // 付费解锁提示
          Column({ space: 15 }) {
            Text(`刚刚AI消耗了${this.computingPower}算力，付费解锁`)
              .fontSize(16)
              .fontColor('#FF6B6B')
              .textAlign(TextAlign.Center)

            Button('立即解锁')
              .width(200)
              .height(45)
              .fontSize(16)
              .fontColor('#FFFFFF')
              .backgroundColor('#FF6B6B')
              .borderRadius(8)
              .onClick(() => {
                this.unlockContent();
              })
          }
          .margin({ top: 30 })
        }
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F8F9FA')
  }

  private unlockContent() {
    // 模拟付费解锁过程
    this.isUnlocked = true;
    console.log('内容已解锁');
  }
}
