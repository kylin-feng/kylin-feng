#!/bin/bash

# PetCompanion 启动脚本
echo "🐾 欢迎使用 PetCompanion - macOS桌面宠物应用！"
echo ""

# 检查是否安装了Xcode
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ 错误：未找到Xcode。请先安装Xcode。"
    echo "   可以从App Store下载Xcode，或访问：https://developer.apple.com/xcode/"
    exit 1
fi

echo "✅ 检测到Xcode，版本信息："
xcodebuild -version
echo ""

# 检查项目文件
if [ ! -f "PetCompanion.xcodeproj/project.pbxproj" ]; then
    echo "❌ 错误：未找到项目文件 PetCompanion.xcodeproj"
    echo "   请确保在正确的目录中运行此脚本。"
    exit 1
fi

echo "✅ 项目文件检查通过"
echo ""

# 尝试编译项目
echo "🔨 正在编译项目..."
if xcodebuild -project PetCompanion.xcodeproj -scheme PetCompanion -configuration Debug build; then
    echo ""
    echo "✅ 编译成功！"
    echo ""
    echo "🚀 启动应用..."
    echo "   应用将在新窗口中打开。"
    echo ""
    echo "📖 使用说明："
    echo "   • 点击播放按钮控制宠物动画"
    echo "   • 点击说话按钮让宠物说话"
    echo "   • 点击心情按钮切换宠物心情"
    echo "   • 点击设置按钮自定义宠物"
    echo ""
    echo "🎉 享受与你的数字宠物共度的美好时光！"
    
    # 尝试运行应用
    open PetCompanion.xcodeproj
else
    echo ""
    echo "❌ 编译失败！"
    echo "   请检查错误信息并修复问题。"
    echo "   常见问题："
    echo "   • 确保macOS版本 >= 14.0"
    echo "   • 确保Xcode版本 >= 15.0"
    echo "   • 检查代码语法错误"
    exit 1
fi
