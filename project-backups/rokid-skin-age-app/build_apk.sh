#!/bin/bash

echo "🚀 开始构建Rokid皮肤年龄分析应用APK..."
echo "📂 工作目录: $(pwd)"

# 设置Flutter环境
export PATH="$PATH:/Users/shixianping/Desktop/flutter/bin"

# 清理项目
echo "🧹 清理项目缓存..."
flutter clean

# 获取依赖
echo "📦 获取Flutter依赖..."
flutter pub get

# 代码分析
echo "🔍 代码质量检查..."
flutter analyze --no-fatal-infos

# 确保Android权限配置
echo "⚙️ 检查Android配置..."
echo "✅ 网络权限: $(grep -q 'android.permission.INTERNET' android/app/src/main/AndroidManifest.xml && echo '已配置' || echo '未配置')"
echo "✅ 相机权限: $(grep -q 'android.permission.CAMERA' android/app/src/main/AndroidManifest.xml && echo '已配置' || echo '未配置')"

# 构建Debug APK（更快，用于测试）
echo "🏗️ 构建Debug版APK..."
flutter build apk --debug --verbose

if [ $? -eq 0 ]; then
    echo "✅ Debug APK构建成功！"
    echo "📱 文件位置: build/app/outputs/flutter-apk/app-debug.apk"
    
    # 显示文件信息
    if [ -f "build/app/outputs/flutter-apk/app-debug.apk" ]; then
        echo "📊 APK信息："
        ls -lh build/app/outputs/flutter-apk/app-debug.apk
        echo "📱 安装命令: adb install -r build/app/outputs/flutter-apk/app-debug.apk"
    fi
else
    echo "❌ APK构建失败"
    echo "🔧 尝试以下解决方案："
    echo "1. 检查网络连接"
    echo "2. 清理Gradle缓存: cd android && ./gradlew clean"
    echo "3. 重新同步项目: flutter pub get"
fi

echo "🏁 构建流程完成"