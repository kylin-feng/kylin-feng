import { preferences } from '@kit.ArkData';
import { Context } from '@kit.AbilityKit';

/**
 * 主题类型枚举
 */
enum ThemeType {
  LIGHT = 'light',
  DARK = 'dark'
}

/**
 * AI模型类型枚举
 */
enum AIModelType {
  DEFAULT = 'default',
  CREATIVE = 'creative',
  FORMAL = 'formal',
  CASUAL = 'casual'
}

/**
 * 语言类型枚举
 */
enum LanguageType {
  ZH_CN = 'zh-CN',
  EN_US = 'en-US',
  JA_JP = 'ja-JP'
}

/**
 * Select选项接口
 */
interface SelectOption {
  value: string;
  label: string;
}

/**
 * 设置页面
 * 提供应用的各种配置选项
 */
@Entry
@Component
struct Settings {
  @State fontSize: number = 16;
  @State theme: ThemeType = ThemeType.LIGHT;
  @State autoSave: boolean = true;
  @State aiModel: AIModelType = AIModelType.DEFAULT;
  @State language: LanguageType = LanguageType.ZH_CN;
  @State showWordCount: boolean = true;
  @State showTemplate: boolean = true;

  private context: Context = getContext(this);

  aboutToAppear() {
    this.loadSettings();
  }

  // 加载设置
  private async loadSettings(): Promise<void> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'app_settings');
      this.fontSize = Number(await dataStore.get('fontSize', 16));
      this.theme = await dataStore.get('theme', ThemeType.LIGHT) as ThemeType;
      this.autoSave = Boolean(await dataStore.get('autoSave', true));
      this.aiModel = await dataStore.get('aiModel', AIModelType.DEFAULT) as AIModelType;
      this.language = await dataStore.get('language', LanguageType.ZH_CN) as LanguageType;
      this.showWordCount = Boolean(await dataStore.get('showWordCount', true));
      this.showTemplate = Boolean(await dataStore.get('showTemplate', true));
    } catch (error) {
      console.error('加载设置失败:', error);
    }
  }

  // 保存设置
  private async saveSettings(): Promise<void> {
    try {
      const dataStore = await preferences.getPreferences(this.context, 'app_settings');
      await dataStore.put('fontSize', this.fontSize);
      await dataStore.put('theme', this.theme);
      await dataStore.put('autoSave', this.autoSave);
      await dataStore.put('aiModel', this.aiModel);
      await dataStore.put('language', this.language);
      await dataStore.put('showWordCount', this.showWordCount);
      await dataStore.put('showTemplate', this.showTemplate);
      await dataStore.flush();
      
      // 显示保存成功提示
      console.log('设置保存成功');
    } catch (error) {
      console.error('保存设置失败:', error);
    }
  }

  build() {
    Column() {
      // 顶部标题栏 - 现代化设计
      Row() {
        Row() {
          Text('⚙️')
            .fontSize(24)
            .margin({ right: 8 })
          Text('应用设置')
            .fontSize(22)
            .fontWeight(FontWeight.Bold)
            .fontColor('#1A1A1A')
            .letterSpacing(0.5)
        }
        
        Blank()
        
        Button('💾 保存')
          .type(ButtonType.Normal)
          .backgroundColor('#007AFF')
          .fontColor(Color.White)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .height(40)
          .borderRadius(20)
          .padding({ left: 20, right: 20 })
          .onClick(() => {
            this.saveSettings();
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
        offsetY: 2
      })

      // 设置内容 - 现代化滚动区域
      Scroll() {
        Column() {
          // 外观设置卡片
          Column() {
            Row() {
              Text('🎨')
                .fontSize(20)
                .margin({ right: 8 })
              Text('外观设置')
                .fontSize(18)
                .fontWeight(FontWeight.Bold)
                .fontColor('#1A1A1A')
            }
            .width('100%')
            .margin({ bottom: 20 })

            Column() {
              this.buildSliderItem('🔤 字体大小', this.fontSize, 12, 24, (value) => {
                this.fontSize = value;
              })
              
              Divider()
                .color('#E5E5E5')
                .margin({ top: 12, bottom: 12 })
              
              this.buildSelectItem('🌓 主题模式', this.theme, [ThemeType.LIGHT, ThemeType.DARK], (value) => {
                this.theme = value as ThemeType;
              })
            }
            .width('100%')
            .backgroundColor('#FFFFFF')
            .borderRadius(16)
            .padding(20)
            .shadow({
              radius: 8,
              color: '#00000005',
              offsetX: 0,
              offsetY: 2
            })
            .border({
              width: 1,
              color: '#E5E5E5'
            })
          }
          .width('100%')
          .margin({ bottom: 24 })

          // 写作设置卡片
          Column() {
            Row() {
              Text('✍️')
                .fontSize(20)
                .margin({ right: 8 })
              Text('写作设置')
                .fontSize(18)
                .fontWeight(FontWeight.Bold)
                .fontColor('#1A1A1A')
            }
            .width('100%')
            .margin({ bottom: 20 })

            Column() {
              this.buildSwitchItem('💾 自动保存', this.autoSave, (value) => {
                this.autoSave = value;
              })
              
              Divider()
                .color('#E5E5E5')
                .margin({ top: 12, bottom: 12 })
              
              this.buildSwitchItem('📊 显示字数统计', this.showWordCount, (value) => {
                this.showWordCount = value;
              })
              
              Divider()
                .color('#E5E5E5')
                .margin({ top: 12, bottom: 12 })
              
              this.buildSwitchItem('🎨 显示模板选择', this.showTemplate, (value) => {
                this.showTemplate = value;
              })
            }
            .width('100%')
            .backgroundColor('#FFFFFF')
            .borderRadius(16)
            .padding(20)
            .shadow({
              radius: 8,
              color: '#00000005',
              offsetX: 0,
              offsetY: 2
            })
            .border({
              width: 1,
              color: '#E5E5E5'
            })
          }
          .width('100%')
          .margin({ bottom: 24 })

          // AI设置卡片
          Column() {
            Row() {
              Text('🤖')
                .fontSize(20)
                .margin({ right: 8 })
              Text('AI设置')
                .fontSize(18)
                .fontWeight(FontWeight.Bold)
                .fontColor('#1A1A1A')
            }
            .width('100%')
            .margin({ bottom: 20 })

            Column() {
              this.buildSelectItem('🧠 AI模型', this.aiModel, [AIModelType.DEFAULT, AIModelType.CREATIVE, AIModelType.FORMAL, AIModelType.CASUAL], (value) => {
                this.aiModel = value as AIModelType;
              })
              
              Divider()
                .color('#E5E5E5')
                .margin({ top: 12, bottom: 12 })
              
              this.buildSelectItem('🌍 语言设置', this.language, [LanguageType.ZH_CN, LanguageType.EN_US, LanguageType.JA_JP], (value) => {
                this.language = value as LanguageType;
              })
            }
            .width('100%')
            .backgroundColor('#FFFFFF')
            .borderRadius(16)
            .padding(20)
            .shadow({
              radius: 8,
              color: '#00000005',
              offsetX: 0,
              offsetY: 2
            })
            .border({
              width: 1,
              color: '#E5E5E5'
            })
          }
          .width('100%')
          .margin({ bottom: 24 })

          // 关于卡片
          Column() {
            Row() {
              Text('ℹ️')
                .fontSize(20)
                .margin({ right: 8 })
              Text('关于应用')
                .fontSize(18)
                .fontWeight(FontWeight.Bold)
                .fontColor('#1A1A1A')
            }
            .width('100%')
            .margin({ bottom: 20 })

            Column() {
              this.buildInfoItem('📱 版本', '1.0.0')
              
              Divider()
                .color('#E5E5E5')
                .margin({ top: 12, bottom: 12 })
              
              this.buildInfoItem('👨‍💻 开发者', 'AI写作助手团队')
              
              Divider()
                .color('#E5E5E5')
                .margin({ top: 12, bottom: 12 })
              
              this.buildInfoItem('📧 联系方式', 'support@aiwriter.com')
            }
            .width('100%')
            .backgroundColor('#FFFFFF')
            .borderRadius(16)
            .padding(20)
            .shadow({
              radius: 8,
              color: '#00000005',
              offsetX: 0,
              offsetY: 2
            })
            .border({
              width: 1,
              color: '#E5E5E5'
            })
          }
          .width('100%')
          .margin({ bottom: 24 })
        }
        .padding(16)
      }
      .layoutWeight(1)
    }
    .height('100%')
    .width('100%')
    .backgroundColor('#FFFFFF')
  }


  // 构建滑块设置项 - 现代化设计
  @Builder
  buildSliderItem(label: string, value: number, min: number, max: number, onChange: (value: number) => void) {
    Row() {
      Text(label)
        .fontSize(16)
        .fontColor('#1A1A1A')
        .fontWeight(FontWeight.Medium)
        .layoutWeight(1)
      
      Row() {
        Text(`${value}`)
          .fontSize(16)
          .fontColor('#007AFF')
          .fontWeight(FontWeight.Bold)
          .margin({ right: 4 })
        Text('px')
          .fontSize(14)
          .fontColor('#666666')
      }
      .backgroundColor('#F0F8FF')
      .padding({ left: 12, right: 12, top: 6, bottom: 6 })
      .borderRadius(12)
    }
    .width('100%')
    .height(50)
    .onClick(() => {
      // 可以添加滑块组件
    })
  }

  // 构建选择器设置项 - 现代化设计
  @Builder
  buildSelectItem(label: string, value: string, options: string[], onChange: (value: string) => void) {
    Row() {
      Text(label)
        .fontSize(16)
        .fontColor('#1A1A1A')
        .fontWeight(FontWeight.Medium)
        .layoutWeight(1)
      
      Select(this.createSelectOptions(options))
        .selected(options.indexOf(value))
        .onSelect((index: number) => {
          onChange(options[index]);
        })
        .width(140)
        .height(36)
        .backgroundColor('#F8F9FA')
        .borderRadius(12)
        .border({
          width: 1,
          color: '#E5E5E5'
        })
    }
    .width('100%')
    .height(50)
  }

  private createSelectOptions(options: string[]): SelectOption[] {
    const selectOptions: SelectOption[] = [];
    for (const option of options) {
      selectOptions.push({
        value: option,
        label: option
      });
    }
    return selectOptions;
  }

  // 构建开关设置项 - 现代化设计
  @Builder
  buildSwitchItem(label: string, value: boolean, onChange: (value: boolean) => void) {
    Row() {
      Text(label)
        .fontSize(16)
        .fontColor('#1A1A1A')
        .fontWeight(FontWeight.Medium)
        .layoutWeight(1)
      
      Toggle({ type: ToggleType.Switch, isOn: value })
        .onChange((isOn: boolean) => {
          onChange(isOn);
        })
        .width(50)
        .height(30)
    }
    .width('100%')
    .height(50)
  }

  // 构建信息项 - 现代化设计
  @Builder
  buildInfoItem(label: string, value: string) {
    Row() {
      Text(label)
        .fontSize(16)
        .fontColor('#1A1A1A')
        .fontWeight(FontWeight.Medium)
        .layoutWeight(1)
      
      Text(value)
        .fontSize(14)
        .fontColor('#666666')
        .backgroundColor('#F8F9FA')
        .padding({ left: 12, right: 12, top: 6, bottom: 6 })
        .borderRadius(8)
    }
    .width('100%')
    .height(50)
  }
}
