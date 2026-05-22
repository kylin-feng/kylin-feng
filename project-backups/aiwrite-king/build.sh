#!/bin/bash

# 鸿蒙应用构建脚本
echo "🚀 开始构建鸿蒙AI写作助手应用..."

# 设置环境变量
export DEVECO_SDK_HOME="/Applications/DevEco-Studio.app/Contents/sdk/default"

# 检查DevEco Studio是否安装
if [ ! -d "/Applications/DevEco-Studio.app" ]; then
    echo "❌ 错误: 未找到DevEco Studio，请先安装DevEco Studio"
    exit 1
fi

echo "✅ 找到DevEco Studio: /Applications/DevEco-Studio.app"

# 检查项目结构
if [ ! -f "hvigorfile.ts" ]; then
    echo "❌ 错误: 未找到hvigorfile.ts，请确保在项目根目录运行"
    exit 1
fi

echo "✅ 项目结构检查通过"

# 尝试使用DevEco Studio的hvigor工具
echo "🔧 尝试使用hvigor构建..."

# 停止现有的daemon进程
/Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw --stop-daemon 2>/dev/null

# 尝试构建
echo "📦 开始构建应用..."
/Applications/DevEco-Studio.app/Contents/tools/hvigor/bin/hvigorw assembleApp --debug

if [ $? -eq 0 ]; then
    echo "✅ 构建成功！"
    echo "📱 应用包位置: entry/build/default/outputs/default/"
    ls -la entry/build/default/outputs/default/ 2>/dev/null || echo "构建输出目录不存在"
else
    echo "❌ 构建失败"
    echo "💡 建议:"
    echo "   1. 请确保已安装完整的HarmonyOS SDK"
    echo "   2. 在DevEco Studio中打开项目进行构建"
    echo "   3. 检查项目配置是否正确"
fi
