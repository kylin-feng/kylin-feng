#!/bin/bash

# 快速构建脚本 - 在SDK安装完成后使用

echo "🚀 快速构建鸿蒙AI写作助手..."

# 检查SDK环境
if [ -z "$DEVECO_SDK_HOME" ]; then
    echo "⚠️  未设置DEVECO_SDK_HOME环境变量"
    echo "💡 请先在DevEco Studio中安装SDK，然后设置环境变量"
    echo "   例如: export DEVECO_SDK_HOME=\"/path/to/your/sdk\""
    exit 1
fi

echo "✅ SDK路径: $DEVECO_SDK_HOME"

# 停止现有daemon
echo "🛑 停止现有daemon进程..."
/Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw --stop-daemon 2>/dev/null

# 清理项目
echo "🧹 清理项目..."
rm -rf entry/build 2>/dev/null
rm -rf .hvigor 2>/dev/null

# 构建debug版本
echo "📦 构建debug版本..."
/Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw assembleApp --debug

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 构建成功！"
    echo "📱 应用包位置:"
    echo "   entry/build/default/outputs/default/"
    echo ""
    echo "📋 构建产物:"
    ls -la entry/build/default/outputs/default/ 2>/dev/null || echo "   构建输出目录不存在"
    echo ""
    echo "🚀 下一步:"
    echo "   1. 在DevEco Studio中运行应用"
    echo "   2. 或使用模拟器/真机测试"
    echo "   3. 准备发布到应用市场"
else
    echo ""
    echo "❌ 构建失败"
    echo "💡 请检查:"
    echo "   1. SDK是否正确安装"
    echo "   2. 环境变量是否正确设置"
    echo "   3. 网络连接是否正常"
    echo ""
    echo "🔧 调试命令:"
    echo "   /Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw assembleApp --debug --stacktrace"
fi
