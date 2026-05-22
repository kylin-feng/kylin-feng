# 工具箱整理说明 📁

本次整理了以下任务产生的文件，按功能分类存放：

## 📂 目录结构

### 🔍 OCR工具 (`/Users/shixianping/Tools/OCR/`)
- `ocr.py` - OCR主程序（Python包装器）
- `ocr.swift` - OCR核心脚本（Swift实现）
- `ocr_macos.py` - OCR配置脚本
- `setup_ocr_alias.sh` - OCR别名设置脚本
- `setup_shortcuts_ocr.md` - OCR快捷指令设置说明

**使用方法：**
```bash
# 基础用法
python3 /Users/shixianping/Tools/OCR/ocr.py <图片路径>

# 设置别名后（已配置）
ocr <图片路径>
```

### 📋 纯净粘贴工具 (`/Users/shixianping/Tools/PlainPaste/`)
- `PlainPaste.scpt` - 纯净粘贴AppleScript（副本，原件在桌面）
- `plain_paste.sh` - 纯净粘贴Shell脚本
- `setup_plain_paste_shortcut.py` - 纯净粘贴设置脚本

**使用方法：**
- 双击桌面的 `PlainPaste.scpt`
- 终端运行：`plainpaste`（已配置别名）

### 📄 生成的内容 (`/Users/shixianping/Generated_Content/`)

#### Markdown文件 (`Markdown_Files/`)
- `vibe coding实训.md` - 完整的编程实训教程

#### PDF教程 (`PDF_Tutorials/`)
- `vibe-coding实训教程-中文优化版.pdf` - 最终优化版PDF
- `vibe-coding实训教程-简化版.pdf` - 简化版PDF
- `test-chinese.pdf` - 中文测试文件
- 其他版本的PDF文件

#### HTML文件
- `vibe-coding实训教程-修复版.html` - 修复emoji问题的HTML版本
- `vibe-coding实训教程-最终版.html` - 最终版HTML（推荐用于PDF转换）
- `test_ocr.html` - OCR测试页面

### ⚙️ 配置脚本 (`/Users/shixianping/Config_Scripts/`)
- `pdf-styles.css` - PDF样式表
- `Setup_Scripts/` - 各种设置和生成脚本
  - 包含所有JavaScript生成脚本
  - Python配置脚本

## 🎯 快速访问

### 常用工具
```bash
# OCR识别
ocr ~/Desktop/screenshot.png

# 纯净粘贴
plainpaste

# 查看教程
open "/Users/shixianping/Generated_Content/vibe-coding实训教程-最终版.html"
```

### 桌面保留文件
- `PlainPaste.scpt` - 纯净粘贴工具（保持在桌面方便使用）

## 🧹 清理完成项目

1. ✅ **OCR工具配置** - 支持中英文识别
2. ✅ **纯净粘贴工具** - 去除格式的复制粘贴
3. ✅ **编程教程生成** - 完整的Markdown到PDF转换
4. ✅ **文件整理归类** - 按功能分类存储

所有工具已配置完成并可正常使用！