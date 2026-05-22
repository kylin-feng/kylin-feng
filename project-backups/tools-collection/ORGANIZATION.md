# 文件组织结构 📁

## 🎯 整理原则
- **用户工具** → 放在 `~/Tools/` 便于访问
- **开发文件** → 隐藏在 `~/.dev_workspace/` 避免混乱
- **常用工具** → 保留在桌面和PATH中

## 📂 最终目录结构

### 🔧 **用户工具** (`~/Tools/`)
```
Tools/
├── OCR/                    # OCR识别工具
│   ├── ocr.py             # 主程序
│   ├── ocr.swift          # 核心引擎
│   └── setup_*.sh         # 配置脚本
├── PlainPaste/            # 纯净粘贴工具
│   ├── PlainPaste.scpt    # 主程序
│   └── setup_*.py         # 配置脚本
└── README.md              # 使用说明
```

### 📄 **生成内容** (`~/Generated_Content/`)
```
Generated_Content/
├── Markdown_Files/        # Markdown源文件
├── PDF_Tutorials/         # PDF教程合集
└── *.html                 # HTML版本教程
```

### 🔒 **开发工作区** (`~/.dev_workspace/`) - 隐藏文件夹
```
.dev_workspace/
├── node_project/          # Node.js依赖和配置
│   ├── node_modules/      # NPM包
│   ├── package.json       # 项目配置
│   └── package-lock.json  # 锁定版本
├── Config_Scripts/        # 配置和生成脚本
│   ├── pdf-styles.css     # PDF样式
│   └── Setup_Scripts/     # 各种设置脚本
└── temp_files/            # 临时文件
```

### 🏠 **桌面保留**
- `PlainPaste.scpt` - 日常使用的纯净粘贴工具

## 🎉 整理完成效果

### ✅ **清洁的主目录**
- 不再有散落的 `.js`, `.css`, `.py` 文件
- 开发依赖隐藏在 `.dev_workspace`
- 用户工具整齐归类在 `Tools`

### ✅ **便于使用**
- 所有工具都有明确的使用说明
- 常用功能保留在桌面和命令行别名
- 开发文件不影响日常使用

### ✅ **便于维护**
- 按功能分类存储
- 隐藏技术细节
- 保留完整的配置能力

## 🚀 快速访问命令

```bash
# 使用工具
ocr image.png              # OCR识别
plainpaste                 # 纯净粘贴

# 查看内容
open ~/Tools/              # 工具箱
open ~/Generated_Content/  # 生成的内容

# 开发需要时访问
ls ~/.dev_workspace/       # 开发文件
```

现在你的主目录整洁了，而且所有功能都保持可用！