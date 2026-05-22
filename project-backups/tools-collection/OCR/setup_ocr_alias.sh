#!/bin/bash

# 设置OCR工具别名的脚本

echo "🔧 设置OCR工具别名..."

# 检查shell类型
if [[ $SHELL == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ $SHELL == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
else
    SHELL_RC="$HOME/.profile"
fi

# 添加别名到shell配置文件
ALIAS_LINE="alias ocr='python3 /Users/shixianping/ocr.py'"

if ! grep -q "alias ocr=" "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo "# OCR工具别名" >> "$SHELL_RC"
    echo "$ALIAS_LINE" >> "$SHELL_RC"
    echo "✅ 别名已添加到 $SHELL_RC"
else
    echo "ℹ️  别名已存在于 $SHELL_RC"
fi

echo ""
echo "🎉 设置完成！"
echo ""
echo "📖 使用方法:"
echo "  1. 重新打开终端，或运行: source $SHELL_RC"
echo "  2. 使用命令: ocr <图片路径>"
echo ""
echo "📝 示例:"
echo "  ocr screenshot.png"
echo "  ocr ~/Desktop/image.jpg"
echo ""
echo "📋 快速截图并OCR:"
echo "  1. Cmd+Shift+4 截图"
echo "  2. ocr ~/Desktop/Screen\\ Shot\\ *.png"