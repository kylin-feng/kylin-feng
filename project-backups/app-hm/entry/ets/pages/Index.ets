import router from '@ohos.router';
import { AIWriterService } from '../services/AIWriterService';

@Entry
@Component
struct Index {
  @State currentBook: string = 'book1';
  @State isGenerating: boolean = false;
  @State generatedContent: string = '';
  @State computingPower: number = 0;

  private aiService: AIWriterService = new AIWriterService();

  build() {
    Column({ space: 20 }) {
      // 顶部标题
      Text('AI小说生成器')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .fontColor('#333333')
        .margin({ top: 60 })

      // 主要内容显示区域
      Column({ space: 15 }) {
        // 小说内容显示框
        Column() {
          Text(this.currentBook)
            .fontSize(18)
            .fontColor('#333333')
            .margin({ bottom: 10 })
          
          Text('滑动观看')
            .fontSize(14)
            .fontColor('#666666')
        }
        .width('100%')
        .height(200)
        .border({ width: 1, color: '#000000' })
        .borderRadius(8)
        .justifyContent(FlexAlign.Center)
        .backgroundColor('#FFFFFF')
        .onClick(() => {
          this.navigateToReader();
        })

        // 开始AI写作按钮
        Button('开启AI写作之旅')
          .width('100%')
          .height(50)
          .fontSize(16)
          .fontColor('#FFFFFF')
          .backgroundColor('#007AFF')
          .borderRadius(8)
          .onClick(() => {
            this.startAIWriting();
          })
          .enabled(!this.isGenerating)
      }
      .width('90%')
      .padding(20)
      .backgroundColor('#F8F9FA')
      .borderRadius(12)
      .margin({ top: 40 })

      // 算力消耗提示
      if (this.computingPower > 0) {
        Column() {
          Text(`刚刚AI消耗了${this.computingPower}算力，付费解锁`)
            .fontSize(14)
            .fontColor('#FF6B6B')
            .textAlign(TextAlign.Center)
            .margin({ top: 20 })
        }
        .width('90%')
      }
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#FFFFFF')
    .padding({ left: 20, right: 20 })
  }

  private async startAIWriting() {
    this.isGenerating = true;
    try {
      // 模拟AI生成过程
      const result = await this.aiService.generateNovel({
        theme: '科幻',
        length: '短篇',
        style: '现代'
      });
      
      this.generatedContent = result.content;
      this.computingPower = result.computingPower;
      this.currentBook = result.title;
    } catch (error) {
      console.error('AI生成失败:', error);
    } finally {
      this.isGenerating = false;
    }
  }

  private navigateToReader() {
    if (this.generatedContent) {
      router.pushUrl({
        url: 'pages/ReaderPage',
        params: {
          content: this.generatedContent,
          title: this.currentBook,
          computingPower: this.computingPower
        }
      });
    }
  }
}
