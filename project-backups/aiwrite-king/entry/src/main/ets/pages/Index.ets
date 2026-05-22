import { AIWritingService } from '../services/AIWritingService';
import { DocumentService, Document } from '../services/DocumentService';
import { Context } from '@kit.AbilityKit';

/**
 * 应用状态枚举
 */
enum AppState {
  IDLE = 'idle',
  GENERATING = 'generating',
  LOADING = 'loading',
  ERROR = 'error'
}

/**
 * 面板类型枚举
 */
enum PanelType {
  AI_ASSISTANT = 'ai_assistant',
  DOCUMENT_LIST = 'document_list',
  SETTINGS = 'settings'
}

/**
 * Select选项接口
 */
interface SelectOption {
  label: string;
  value: string;
}

@Entry
@Component
struct Index {
  @State currentText: string = '';
  @State isAIAssistantVisible: boolean = false;
  @State aiSuggestion: string = '';
  @State appState: AppState = AppState.IDLE;
  @State selectedTemplate: string = '';
  @State documentTitle: string = '';
  @State wordCount: number = 0;
  @State showDocumentList: boolean = false;
  @State documentList: Document[] = [];
  @State currentDocumentId: string = '';
  @State showSettings: boolean = false;
  @State remainingCalls: number = 10;
  @State isRateLimited: boolean = false;
  @State activePanel: PanelType | null = null;

  private aiService: AIWritingService = AIWritingService.getInstance();
  private documentService: DocumentService | null = null;

  aboutToAppear() {
    // 初始化文档服务
    this.documentService = DocumentService.getInstance(getContext(this));
    // 初始化限流服务
    this.aiService.initRateLimit(getContext(this));
    this.loadDocumentList();
    this.updateRateLimitInfo();
  }

  // AI写作功能
  private async generateText(prompt: string): Promise<void> {
    this.appState = AppState.GENERATING;
    try {
      this.aiSuggestion = await this.aiService.generateWritingSuggestion(prompt, this.selectedTemplate);
      // 更新限流信息
      await this.updateRateLimitInfo();
    } catch (error) {
      this.appState = AppState.ERROR;
      if (error instanceof Error && error.message.includes('已达上限')) {
        this.aiSuggestion = `今日AI调用次数已达上限，剩余次数：${this.remainingCalls}`;
      } else {
        this.aiSuggestion = '生成失败，请重试';
      }
    } finally {
      this.appState = AppState.IDLE;
    }
  }

  // 应用AI建议
  private applyAISuggestion(): void {
    this.currentText += '\n\n' + this.aiSuggestion;
    this.aiSuggestion = '';
    this.isAIAssistantVisible = false;
    this.updateWordCount();
  }

  // 优化文本
  private async optimizeText(): Promise<void> {
    if (!this.currentText.trim()) return;
    
    this.appState = AppState.GENERATING;
    try {
      const optimizedText = await this.aiService.optimizeText(this.currentText);
      this.aiSuggestion = optimizedText;
      // 更新限流信息
      await this.updateRateLimitInfo();
    } catch (error) {
      this.appState = AppState.ERROR;
      if (error instanceof Error && error.message.includes('已达上限')) {
        this.aiSuggestion = `今日AI调用次数已达上限，剩余次数：${this.remainingCalls}`;
      } else {
        this.aiSuggestion = '优化失败，请重试';
      }
    } finally {
      this.appState = AppState.IDLE;
    }
  }

  // 续写文本
  private async continueWriting(): Promise<void> {
    console.log('续写功能被调用');
    if (!this.currentText.trim()) {
      console.log('当前文本为空，跳过续写');
      return;
    }
    
    console.log('开始续写，当前文本长度:', this.currentText.length);
    this.appState = AppState.GENERATING;
    try {
      console.log('调用AI服务进行续写');
      const continuedText = await this.aiService.continueWriting(this.currentText);
      console.log('续写完成，结果长度:', continuedText.length);
      this.aiSuggestion = continuedText;
      // 更新限流信息
      await this.updateRateLimitInfo();
    } catch (error) {
      console.error('续写失败:', error);
      this.appState = AppState.ERROR;
      if (error instanceof Error && error.message.includes('已达上限')) {
        this.aiSuggestion = `今日AI调用次数已达上限，剩余次数：${this.remainingCalls}`;
      } else {
        this.aiSuggestion = '续写失败，请重试';
      }
    } finally {
      this.appState = AppState.IDLE;
      console.log('续写流程结束');
    }
  }

  // 改写文本
  private async rewriteText(): Promise<void> {
    if (!this.currentText.trim()) return;
    
    this.appState = AppState.GENERATING;
    try {
      const rewrittenText = await this.aiService.rewriteText(this.currentText);
      this.aiSuggestion = rewrittenText;
      // 更新限流信息
      await this.updateRateLimitInfo();
    } catch (error) {
      this.appState = AppState.ERROR;
      if (error instanceof Error && error.message.includes('已达上限')) {
        this.aiSuggestion = `今日AI调用次数已达上限，剩余次数：${this.remainingCalls}`;
      } else {
        this.aiSuggestion = '改写失败，请重试';
      }
    } finally {
      this.appState = AppState.IDLE;
    }
  }

  // 写作模板
  private getWritingTemplates(): string[] {
    return this.aiService.getWritingTemplates().map(template => template.name);
  }

  // 获取Select选项
  private getSelectOptions(): SelectOption[] {
    const templates = this.aiService.getWritingTemplates();
    const options: SelectOption[] = [];
    for (const template of templates) {
      options.push({
        value: template.name,
        label: template.name
      });
    }
    return options;
  }

  // 保存文档
  private async saveDocument(): Promise<void> {
    if (!this.documentService) return;
    
    const title = this.documentTitle || '未命名文档';
    const success = await this.documentService.saveDocument(title, this.currentText, this.selectedTemplate);
    
    if (success) {
      // 显示保存成功提示
      console.log('文档保存成功');
      this.loadDocumentList();
    } else {
      console.log('文档保存失败');
    }
  }

  // 加载文档列表
  private async loadDocumentList(): Promise<void> {
    if (!this.documentService) return;
    
    this.documentList = await this.documentService.getDocumentList();
  }

  // 加载文档
  private async loadDocument(documentId: string): Promise<void> {
    if (!this.documentService) return;
    
    const document = await this.documentService.loadDocument(documentId);
    if (document) {
      this.currentText = document.content;
      this.documentTitle = document.title;
      this.selectedTemplate = document.template || '';
      this.currentDocumentId = documentId;
      this.updateWordCount();
    }
  }

  // 更新字数统计
  private updateWordCount(): void {
    this.wordCount = Number(this.currentText.length);
  }

  // 更新限流信息
  private async updateRateLimitInfo(): Promise<void> {
    try {
      const rateLimit = await this.aiService.checkRateLimit();
      this.remainingCalls = rateLimit.remaining;
      this.isRateLimited = !rateLimit.canCall;
    } catch (error) {
      console.error('获取限流信息失败:', error);
    }
  }

  // 格式化AI建议文本
  private formatAISuggestion(text: string): string {
    if (!text) return '';
    
    // 清理Markdown格式，转换为更易读的格式
    let formatted = text
      // 移除多余的Markdown标记
      .replace(/###\*\*/g, '')
      .replace(/\*\*/g, '')
      .replace(/###/g, '')
      .replace(/\*\*/g, '')
      // 处理标题
      .replace(/^(\d+\.\s*\*\*.*?\*\*)/gm, '\n$1\n')
      // 处理列表项
      .replace(/^(\s*-\s*)/gm, '• ')
      // 处理引用
      .replace(/^>\s*/gm, '  ')
      // 处理分隔线
      .replace(/^---+$/gm, '─────────────────────')
      // 清理多余的空行
      .replace(/\n{3,}/g, '\n\n')
      // 移除开头和结尾的空白
      .trim();
    
    return formatted;
  }

  // 构建设置项
  @Builder
  buildSettingItem(label: string, value: string, onClick: () => void) {
    Row() {
      Text(label)
        .fontSize(16)
        .fontColor('#333333')
        .layoutWeight(1)
      
      Text(value)
        .fontSize(14)
        .fontColor('#666666')
        .margin({ right: 8 })
      
      Image($r('app.media.startIcon'))
        .width(16)
        .height(16)
        .fillColor('#999999')
    }
    .width('100%')
    .height(40)
    .onClick(onClick)
  }

  build() {
    Column() {
      // 顶部工具栏 - 优化宽度和间距
      Row() {
        Row() {
          Text('✍️')
            .fontSize(24)
            .margin({ right: 8 })
          
          Text('AI写作助手')
            .fontSize(22)
            .fontWeight(FontWeight.Bold)
            .fontColor('#1A1A1A')
            .letterSpacing(0.5)
        }
        .layoutWeight(1)
        
        // 状态指示器
        Row() {
          if (this.isRateLimited) {
            Text('⚠️')
              .fontSize(16)
              .margin({ right: 4 })
            Text('已达上限')
              .fontSize(12)
              .fontColor('#FF3B30')
              .backgroundColor('#FFE5E5')
              .padding({ left: 8, right: 8, top: 4, bottom: 4 })
              .borderRadius(12)
          } else {
            Text('✨')
              .fontSize(16)
              .margin({ right: 4 })
            Text(`剩余 ${this.remainingCalls} 次`)
              .fontSize(12)
              .fontColor('#34C759')
              .backgroundColor('#E8F5E8')
              .padding({ left: 8, right: 8, top: 4, bottom: 4 })
              .borderRadius(12)
          }
        }
        .margin({ right: 16 })
        
        // 功能按钮组
        Row() {
          Button('📄 文档')
            .type(ButtonType.Normal)
            .backgroundColor(this.showDocumentList ? '#007AFF' : '#F2F2F7')
            .fontColor(this.showDocumentList ? Color.White : '#007AFF')
            .borderRadius(20)
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .padding({ left: 16, right: 16, top: 8, bottom: 8 })
            .margin({ right: 8 })
            .onClick(() => {
              this.showDocumentList = !this.showDocumentList;
              this.isAIAssistantVisible = false;
              this.showSettings = false;
            })
          
          Button('🤖 AI助手')
            .type(ButtonType.Normal)
            .backgroundColor(this.isAIAssistantVisible ? '#007AFF' : (this.isRateLimited ? '#8E8E93' : '#F2F2F7'))
            .fontColor(this.isAIAssistantVisible ? Color.White : (this.isRateLimited ? '#8E8E93' : '#007AFF'))
            .borderRadius(20)
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .padding({ left: 16, right: 16, top: 8, bottom: 8 })
            .enabled(!this.isRateLimited)
            .onClick(() => {
              this.isAIAssistantVisible = !this.isAIAssistantVisible;
              this.showDocumentList = false;
              this.showSettings = false;
            })
        }
      }
      .width('100%')
      .height(70)
      .padding({ left: 24, right: 24, top: 12, bottom: 12 })
      .backgroundColor('#FFFFFF')
      .shadow({
        radius: 8,
        color: '#00000010',
        offsetX: 0,
        offsetY: 2
      })

      // 主内容区域 - 优化比例布局
      Row() {
        // 左侧写作区域 - 主要区域
        Column() {
          // 文档信息卡片 - 优化布局
          Column() {
            // 文档标题 - 全宽
            Column() {
              Text('📝 文档标题')
                .fontSize(14)
                .fontWeight(FontWeight.Medium)
                .fontColor('#666666')
                .margin({ bottom: 8 })
              
              TextInput({ placeholder: '请输入文档标题...' })
                .onChange((value: string) => {
                  this.documentTitle = value;
                })
                .width('100%')
                .height(40)
                .fontSize(16)
                .backgroundColor('#F8F9FA')
                .borderRadius(12)
                .padding({ left: 16, right: 16 })
                .border({
                  width: 1,
                  color: '#E5E5E5'
                })
            }
            .width('100%')
            .margin({ bottom: 16 })
            
            // 写作模板 - 全宽
            Column() {
              Text('🎨 写作模板')
                .fontSize(14)
                .fontWeight(FontWeight.Medium)
                .fontColor('#666666')
                .margin({ bottom: 8 })
              
              Select(this.getSelectOptions())
                .selected(this.getWritingTemplates().findIndex(template => template === this.selectedTemplate))
                .onSelect((index: number) => {
                  this.selectedTemplate = this.getWritingTemplates()[index];
                })
                .width('100%')
                .height(40)
                .backgroundColor('#F8F9FA')
                .borderRadius(12)
                .border({
                  width: 1,
                  color: '#E5E5E5'
                })
            }
            .width('100%')
          }
          .width('100%')
          .backgroundColor('#FFFFFF')
          .borderRadius(16)
          .padding(20)
          .margin({ bottom: 20 })
          .shadow({
            radius: 12,
            color: '#00000008',
            offsetX: 0,
            offsetY: 4
          })

          // 文本编辑区域 - 现代化编辑器
          Column() {
            // 智能工具栏 - 优化布局
            Row() {
              Row() {
                Text('📊')
                  .fontSize(16)
                  .margin({ right: 6 })
                Text(`字数: ${this.wordCount}`)
                  .fontSize(14)
                  .fontColor('#666666')
                  .fontWeight(FontWeight.Medium)
              }
              .backgroundColor('#F8F9FA')
              .padding({ left: 12, right: 12, top: 6, bottom: 6 })
              .borderRadius(20)
              .layoutWeight(1)
              
              // AI功能按钮组 - 水平布局
              Row() {
                Button('✨ 优化')
                  .type(ButtonType.Normal)
                  .backgroundColor('#FF9500')
                  .fontColor(Color.White)
                  .fontSize(13)
                  .fontWeight(FontWeight.Medium)
                  .height(32)
                  .borderRadius(16)
                  .padding({ left: 12, right: 12 })
                  .layoutWeight(1)
                  .onClick(() => {
                    this.optimizeText();
                  })
                
                Button('📝 续写')
                  .type(ButtonType.Normal)
                  .backgroundColor('#34C759')
                  .fontColor(Color.White)
                  .fontSize(13)
                  .fontWeight(FontWeight.Medium)
                  .height(32)
                  .borderRadius(16)
                  .padding({ left: 12, right: 12 })
                  .margin({ left: 6 })
                  .layoutWeight(1)
                  .onClick(() => {
                    this.continueWriting();
                  })
                
                Button('🔄 改写')
                  .type(ButtonType.Normal)
                  .backgroundColor('#AF52DE')
                  .fontColor(Color.White)
                  .fontSize(13)
                  .fontWeight(FontWeight.Medium)
                  .height(32)
                  .borderRadius(16)
                  .padding({ left: 12, right: 12 })
                  .margin({ left: 6 })
                  .layoutWeight(1)
                  .onClick(() => {
                    this.rewriteText();
                  })
              }
              .layoutWeight(2)
            }
            .width('100%')
            .margin({ bottom: 16 })

            // 现代化文本编辑器
            TextArea({
              placeholder: '✨ 开始您的创作之旅...\n\n💡 提示：输入一些关键词或句子，然后点击"AI助手"获取智能建议',
              text: this.currentText
            })
              .onChange((value: string) => {
                this.currentText = value;
                this.updateWordCount();
              })
              .backgroundColor('#FFFFFF')
              .borderRadius(16)
              .padding(20)
              .fontSize(16)
              .lineHeight(24)
              .fontColor('#1A1A1A')
              .height('100%')
              .border({
                width: 1,
                color: '#E5E5E5'
              })
              .shadow({
                radius: 8,
                color: '#00000005',
                offsetX: 0,
                offsetY: 2
              })
          }
          .layoutWeight(1)
        }
        .layoutWeight(2)
        .padding(16)

        // 右侧AI助手面板 - 现代化设计
        if (this.isAIAssistantVisible) {
          Column() {
            // AI助手头部
            Row() {
              Row() {
                Text('🤖')
                  .fontSize(20)
                  .margin({ right: 8 })
                Text('AI写作助手')
                  .fontSize(18)
                  .fontWeight(FontWeight.Bold)
                  .fontColor('#1A1A1A')
              }
              
              Blank()
              
              // 状态指示器
              if (this.isRateLimited) {
                Row() {
                  Text('⚠️')
                    .fontSize(14)
                    .margin({ right: 4 })
                  Text('已达上限')
                    .fontSize(12)
                    .fontColor('#FF3B30')
                    .backgroundColor('#FFE5E5')
                    .padding({ left: 8, right: 8, top: 4, bottom: 4 })
                    .borderRadius(12)
                }
              } else {
                Row() {
                  Text('✨')
                    .fontSize(14)
                    .margin({ right: 4 })
                  Text(`剩余 ${this.remainingCalls} 次`)
                    .fontSize(12)
                    .fontColor('#34C759')
                    .backgroundColor('#E8F5E8')
                    .padding({ left: 8, right: 8, top: 4, bottom: 4 })
                    .borderRadius(12)
                }
              }
            }
            .width('100%')
            .margin({ bottom: 20 })

            if (this.appState === AppState.GENERATING) {
              // 加载状态 - 现代化设计
              Column() {
                Row() {
                  LoadingProgress()
                    .width(24)
                    .height(24)
                    .color('#007AFF')
                  
                  Text('AI正在思考中...')
                    .fontSize(16)
                    .fontColor('#666666')
                    .margin({ left: 12 })
                    .fontWeight(FontWeight.Medium)
                }
                .justifyContent(FlexAlign.Center)
                .width('100%')
                .padding(20)
                .backgroundColor('#F8F9FA')
                .borderRadius(16)
                .margin({ bottom: 16 })
                
                Text('💭 正在分析您的文本并生成最佳建议...')
                  .fontSize(14)
                  .fontColor('#999999')
                  .textAlign(TextAlign.Center)
                  .width('100%')
              }
              .width('100%')
              .height(120)
            } else {
              // AI建议区域 - 现代化卡片设计
              Column() {
                Row() {
                  Text('💡')
                    .fontSize(18)
                    .margin({ right: 8 })
                  Text('AI智能建议')
                    .fontSize(16)
                    .fontWeight(FontWeight.Bold)
                    .fontColor('#1A1A1A')
                }
                .width('100%')
                .margin({ bottom: 12 })

                Scroll() {
                  Text(this.formatAISuggestion(this.aiSuggestion) || '🎯 点击下方按钮获取AI智能建议')
                    .fontSize(15)
                    .fontColor('#333333')
                    .lineHeight(22)
                    .textAlign(TextAlign.Start)
                    .width('100%')
                }
                .backgroundColor('#FFFFFF')
                .padding(16)
                .borderRadius(12)
                .width('100%')
                .height(200)
                .border({
                  width: 1,
                  color: '#E5E5E5'
                })
                .shadow({
                  radius: 8,
                  color: '#00000005',
                  offsetX: 0,
                  offsetY: 2
                })

                // AI操作按钮组 - 现代化设计
                Column() {
                  // 主要操作按钮
                  Row() {
                    Button('🚀 生成建议')
                      .type(ButtonType.Normal)
                      .backgroundColor(this.isRateLimited ? '#8E8E93' : '#007AFF')
                      .fontColor(Color.White)
                      .fontSize(14)
                      .fontWeight(FontWeight.Medium)
                      .height(40)
                      .borderRadius(20)
                      .padding({ left: 20, right: 20 })
                      .enabled(!this.isRateLimited)
                      .onClick(() => {
                        if (!this.isRateLimited) {
                          this.generateText(this.currentText);
                        }
                      })

                    if (this.aiSuggestion) {
                      Button('✅ 应用建议')
                        .type(ButtonType.Normal)
                        .backgroundColor('#34C759')
                        .fontColor(Color.White)
                        .fontSize(14)
                        .fontWeight(FontWeight.Medium)
                        .height(40)
                        .borderRadius(20)
                        .padding({ left: 20, right: 20 })
                        .margin({ left: 8 })
                        .onClick(() => {
                          this.applyAISuggestion();
                        })
                    }
                  }
                  .width('100%')
                  .justifyContent(FlexAlign.Start)
                  .margin({ bottom: 16 })

                  // AI功能按钮组
                  Row() {
                    Button('✨ 优化')
                      .type(ButtonType.Normal)
                      .backgroundColor(this.isRateLimited ? '#8E8E93' : '#FF9500')
                      .fontColor(Color.White)
                      .fontSize(13)
                      .fontWeight(FontWeight.Medium)
                      .height(36)
                      .borderRadius(18)
                      .padding({ left: 16, right: 16 })
                      .enabled(!this.isRateLimited)
                      .onClick(() => {
                        if (!this.isRateLimited) {
                          this.optimizeText();
                        }
                      })

                    Button('📝 续写')
                      .type(ButtonType.Normal)
                      .backgroundColor(this.isRateLimited ? '#8E8E93' : '#34C759')
                      .fontColor(Color.White)
                      .fontSize(13)
                      .fontWeight(FontWeight.Medium)
                      .height(36)
                      .borderRadius(18)
                      .padding({ left: 16, right: 16 })
                      .margin({ left: 6 })
                      .enabled(!this.isRateLimited)
                      .onClick(() => {
                        if (!this.isRateLimited) {
                          this.continueWriting();
                        }
                      })

                    Button('🔄 改写')
                      .type(ButtonType.Normal)
                      .backgroundColor(this.isRateLimited ? '#8E8E93' : '#AF52DE')
                      .fontColor(Color.White)
                      .fontSize(13)
                      .fontWeight(FontWeight.Medium)
                      .height(36)
                      .borderRadius(18)
                      .padding({ left: 16, right: 16 })
                      .margin({ left: 6 })
                      .enabled(!this.isRateLimited)
                      .onClick(() => {
                        if (!this.isRateLimited) {
                          this.rewriteText();
                        }
                      })
                  }
                  .width('100%')
                  .justifyContent(FlexAlign.Start)
                }
                .margin({ top: 16 })
              }
            }
          }
          .width(350)
          .height('100%')
          .padding(24)
          .backgroundColor('#FFFFFF')
          .borderRadius(16)
          .margin({ left: 16 })
          .shadow({
            radius: 12,
            color: '#00000008',
            offsetX: 0,
            offsetY: 4
          })
          .border({
            width: 1,
            color: '#E5E5E5'
          })
        }

        // 文档列表面板 - 现代化设计
        if (this.showDocumentList) {
          Column() {
            // 文档列表头部
            Row() {
              Text('📄')
                .fontSize(20)
                .margin({ right: 8 })
              Text('我的文档')
                .fontSize(18)
                .fontWeight(FontWeight.Bold)
                .fontColor('#1A1A1A')
            }
            .width('100%')
            .margin({ bottom: 20 })

            if (this.documentList.length === 0) {
              // 空状态设计
              Column() {
                Text('📝')
                  .fontSize(48)
                  .margin({ bottom: 16 })
                Text('暂无文档')
                  .fontSize(16)
                  .fontColor('#666666')
                  .fontWeight(FontWeight.Medium)
                  .margin({ bottom: 8 })
                Text('点击"新建文档"开始创作')
                  .fontSize(14)
                  .fontColor('#999999')
              }
              .width('100%')
              .height(200)
              .justifyContent(FlexAlign.Center)
            } else {
              // 文档列表
              List() {
                ForEach(this.documentList, (document: Document) => {
                  ListItem() {
                    Column() {
                      Row() {
                        Text('📄')
                          .fontSize(16)
                          .margin({ right: 8 })
                        Text(document.title)
                          .fontSize(16)
                          .fontWeight(FontWeight.Medium)
                          .fontColor('#1A1A1A')
                          .maxLines(1)
                          .textOverflow({ overflow: TextOverflow.Ellipsis })
                          .layoutWeight(1)
                        
                        Text(document.updateTime.split('T')[0])
                          .fontSize(12)
                          .fontColor('#666666')
                          .backgroundColor('#F0F0F0')
                          .padding({ left: 8, right: 8, top: 4, bottom: 4 })
                          .borderRadius(8)
                      }
                      .width('100%')
                      .margin({ bottom: 8 })

                      Row() {
                        Text(`📊 ${document.wordCount} 字`)
                          .fontSize(12)
                          .fontColor('#666666')
                        
                        if (document.template) {
                          Text(`🎨 ${document.template}`)
                            .fontSize(12)
                            .fontColor('#666666')
                            .margin({ left: 12 })
                            .backgroundColor('#F0F0F0')
                            .padding({ left: 6, right: 6, top: 2, bottom: 2 })
                            .borderRadius(4)
                        }
                      }
                      .width('100%')
                    }
                    .width('100%')
                    .padding(16)
                    .backgroundColor('#FFFFFF')
                    .borderRadius(12)
                    .border({
                      width: 1,
                      color: '#E5E5E5'
                    })
                    .shadow({
                      radius: 4,
                      color: '#00000005',
                      offsetX: 0,
                      offsetY: 2
                    })
                    .onClick(() => {
                      this.loadDocument(document.id);
                      this.showDocumentList = false;
                    })
                  }
                  .margin({ bottom: 12 })
                })
              }
              .width('100%')
              .height('100%')
            }
          }
          .width(350)
          .height('100%')
          .padding(24)
          .backgroundColor('#FFFFFF')
          .borderRadius(16)
          .margin({ left: 16 })
          .shadow({
            radius: 12,
            color: '#00000008',
            offsetX: 0,
            offsetY: 4
          })
          .border({
            width: 1,
            color: '#E5E5E5'
          })
        }

        // 设置面板
        if (this.showSettings) {
          Column() {
            Text('应用设置')
              .fontSize(18)
        .fontWeight(FontWeight.Bold)
              .margin({ bottom: 16 })
              .fontColor('#333333')

            Column() {
              // 外观设置
              this.buildSettingItem('字体大小', '16px', () => {
                console.log('调整字体大小');
              })
              
              this.buildSettingItem('主题', '浅色', () => {
                console.log('切换主题');
              })
              
              this.buildSettingItem('自动保存', '开启', () => {
                console.log('切换自动保存');
              })
              
              this.buildSettingItem('AI模型', '默认', () => {
                console.log('选择AI模型');
              })
            }
            .width('100%')
            .backgroundColor('#F8F9FA')
            .borderRadius(8)
            .padding(16)
          }
          .width(300)
          .height('100%')
          .padding(16)
          .backgroundColor('#F8F9FA')
          .borderRadius(8)
          .margin({ left: 8 })
        }
      }
      .height('100%')
      .width('100%')

      // 底部操作栏 - 水平布局
      Row() {
        Button('💾 保存')
          .type(ButtonType.Normal)
          .backgroundColor('#34C759')
          .fontColor(Color.White)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(40)
          .borderRadius(20)
          .padding({ left: 20, right: 20 })
          .onClick(() => {
            this.saveDocument();
          })

        Button('📄 新建')
          .type(ButtonType.Normal)
          .backgroundColor('#007AFF')
          .fontColor(Color.White)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(40)
          .borderRadius(20)
          .padding({ left: 20, right: 20 })
          .margin({ left: 12 })
          .onClick(() => {
            this.currentText = '';
            this.documentTitle = '';
            this.selectedTemplate = '';
            this.currentDocumentId = '';
            this.updateWordCount();
          })

        Button('📤 导出')
          .type(ButtonType.Normal)
          .backgroundColor('#FF9500')
          .fontColor(Color.White)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(40)
          .borderRadius(20)
          .padding({ left: 20, right: 20 })
          .margin({ left: 12 })
          .onClick(() => {
            // 导出功能
            console.log('导出文档');
          })

        Button('⚙️ 设置')
          .type(ButtonType.Normal)
          .backgroundColor('#8E8E93')
          .fontColor(Color.White)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(40)
          .borderRadius(20)
          .padding({ left: 20, right: 20 })
          .margin({ left: 12 })
          .onClick(() => {
            this.showSettings = !this.showSettings;
            this.showDocumentList = false;
            this.isAIAssistantVisible = false;
          })
      }
      .width('100%')
      .height(70)
      .padding({ left: 20, right: 20, top: 12, bottom: 12 })
      .backgroundColor('#FFFFFF')
      .shadow({
        radius: 8,
        color: '#00000010',
        offsetX: 0,
        offsetY: -2
      })
      .justifyContent(FlexAlign.Start)
    }
    .height('100%')
    .width('100%')
    .backgroundColor('#FFFFFF')
  }
}