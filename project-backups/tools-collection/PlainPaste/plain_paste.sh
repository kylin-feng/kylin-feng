#!/bin/bash
# 纯净粘贴脚本 - 清除剪贴板格式

# 获取剪贴板内容并重新设置为纯文本
pbpaste | pbcopy

echo "✅ 剪贴板内容已转换为纯文本"