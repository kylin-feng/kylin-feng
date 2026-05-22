# Rokid皮肤年龄分析应用 - 打包部署指南

## 📦 APK打包流程

### 🛠️ 环境准备

1. **检查Flutter环境**
   ```bash
   flutter doctor -v
   ```
   
   确保以下组件正常：
   - ✅ Flutter SDK
   - ✅ Android toolchain
   - ✅ Android Studio
   - ✅ 已接受Android许可证

2. **项目清理和依赖安装**
   ```bash
   cd /Users/shixianping/2025-10/app-rokid
   flutter clean
   flutter pub get
   ```

### 🔧 Gradle配置优化

#### 1. 更新 `android/settings.gradle`
```gradle
pluginManagement {
    def flutterSdkPath = {
        def properties = new Properties()
        file("local.properties").withInputStream { properties.load(it) }
        def flutterSdkPath = properties.getProperty("flutter.sdk")
        assert flutterSdkPath != null, "flutter.sdk not set in local.properties"
        return flutterSdkPath
    }
    settings.ext.flutterSdkPath = flutterSdkPath()

    includeBuild("${settings.ext.flutterSdkPath}/packages/flutter_tools/gradle")

    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id "dev.flutter.flutter-plugin-loader" version "1.0.0"
    id "com.android.application" version "7.4.2" apply false
    id "org.jetbrains.kotlin.android" version "1.7.10" apply false
}

include ":app"
```

#### 2. 更新 `android/app/build.gradle`
```gradle
plugins {
    id "com.android.application"
    id "kotlin-android"
    id "dev.flutter.flutter-gradle-plugin"
}
```

#### 3. 优化 `android/gradle.properties`
```properties
org.gradle.jvmargs=-Xmx4G -XX:MaxMetaspaceSize=2G -XX:+HeapDumpOnOutOfMemoryError
android.useAndroidX=true
android.enableJetifier=true
org.gradle.configureondemand=true
org.gradle.daemon=true
org.gradle.parallel=true
android.bundle.enableUncompressedNativeLibs=false
```

### 🏗️ APK构建命令

#### Debug版本（用于开发测试）
```bash
flutter build apk --debug
```

#### Release版本（用于发布）
```bash
flutter build apk --release
```

#### 多架构APK（推荐）
```bash
flutter build apk --release --split-per-abi
```

### 📍 APK输出位置

构建成功后，APK文件位于：

- **Debug版本**: `build/app/outputs/flutter-apk/app-debug.apk`
- **Release版本**: `build/app/outputs/flutter-apk/app-release.apk`
- **分架构版本**:
  - `build/app/outputs/flutter-apk/app-arm64-v8a-release.apk`
  - `build/app/outputs/flutter-apk/app-armeabi-v7a-release.apk`
  - `build/app/outputs/flutter-apk/app-x86_64-release.apk`

### 📱 安装到设备

#### 通过ADB安装
```bash
# 安装Debug版本
adb install build/app/outputs/flutter-apk/app-debug.apk

# 安装Release版本
adb install build/app/outputs/flutter-apk/app-release.apk

# 强制重新安装（覆盖已有应用）
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

#### 直接传输安装
1. 将APK文件传输到Android设备
2. 在设备上打开文件管理器
3. 找到APK文件并点击安装
4. 允许来源未知的应用安装（如需要）

### 🔍 构建故障排除

#### 常见问题及解决方案

##### 1. Gradle Plugin错误
**错误信息**: `Flutter's app_plugin_loader Gradle plugin imperatively`

**解决方案**: 更新到新的插件语法（已在上述配置中修复）

##### 2. 网络连接问题
**错误信息**: `Connection terminated during handshake`

**解决方案**:
- 检查网络连接
- 配置Gradle镜像源
- 使用VPN或代理

##### 3. 内存不足
**错误信息**: `OutOfMemoryError`

**解决方案**:
- 增加Gradle JVM内存：`org.gradle.jvmargs=-Xmx4G`
- 关闭其他占用内存的应用

##### 4. Android SDK问题
**错误信息**: `Android SDK not found`

**解决方案**:
```bash
flutter config --android-sdk /path/to/android/sdk
flutter doctor --android-licenses
```

##### 5. 构建缓存问题
**解决方案**:
```bash
flutter clean
cd android
./gradlew clean
cd ..
flutter pub get
```

### 🎯 真机调试步骤

#### 1. 启用开发者选项
1. 设置 → 关于手机
2. 连续点击"版本号"7次
3. 返回设置，找到"开发者选项"

#### 2. 开启USB调试
1. 进入开发者选项
2. 开启"USB调试"
3. 开启"安装未知来源的应用"

#### 3. 连接设备
```bash
# 检查设备连接
adb devices

# 查看设备信息
flutter devices

# 运行到指定设备
flutter run -d [设备ID]
```

#### 4. 实时调试
```bash
# 热重载模式运行
flutter run --debug

# 发布模式运行
flutter run --release
```

### 🛡️ 权限配置

#### AndroidManifest.xml必要权限
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

### 📊 性能优化建议

#### 1. APK大小优化
```bash
# 启用代码混淆
flutter build apk --release --obfuscate --split-debug-info=debug-info/

# 分架构构建
flutter build apk --release --split-per-abi
```

#### 2. 启动性能优化
- 减少启动时的初始化操作
- 使用SplashScreen延迟重型操作
- 优化图片资源大小

#### 3. 运行时性能
- 使用适当的图片格式和尺寸
- 避免在UI线程执行网络请求
- 合理使用状态管理

### 🔐 签名配置（生产环境）

#### 1. 生成签名密钥
```bash
keytool -genkey -v -keystore key.jks -keyalg RSA -keysize 2048 -validity 10000 -alias key
```

#### 2. 配置签名文件
在 `android/app/build.gradle` 中添加：
```gradle
signingConfigs {
    release {
        keyAlias 'key'
        keyPassword 'your_password'
        storeFile file('key.jks')
        storePassword 'your_password'
    }
}
buildTypes {
    release {
        signingConfig signingConfigs.release
        minifyEnabled true
        proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
    }
}
```

### 🚀 自动化构建脚本

创建 `build.sh` 脚本：
```bash
#!/bin/bash
echo "🚀 开始构建Rokid皮肤年龄分析应用..."

# 清理项目
echo "🧹 清理项目..."
flutter clean

# 获取依赖
echo "📦 获取依赖..."
flutter pub get

# 代码分析
echo "🔍 代码分析..."
flutter analyze

# 构建APK
echo "🏗️ 构建Release APK..."
flutter build apk --release --split-per-abi

# 显示构建结果
echo "✅ 构建完成！APK文件位置："
echo "📱 ARM64: build/app/outputs/flutter-apk/app-arm64-v8a-release.apk"
echo "📱 ARMv7: build/app/outputs/flutter-apk/app-armeabi-v7a-release.apk"
echo "📱 x86_64: build/app/outputs/flutter-apk/app-x86_64-release.apk"

# 文件大小统计
echo "📊 APK文件大小："
ls -lh build/app/outputs/flutter-apk/*.apk
```

### 📱 Rokid设备特殊配置

#### 1. 横屏适配
```xml
<!-- AndroidManifest.xml -->
<activity
    android:screenOrientation="landscape"
    android:configChanges="orientation|keyboardHidden|keyboard|screenSize">
```

#### 2. 硬件加速
```xml
<application
    android:hardwareAccelerated="true">
```

#### 3. 性能配置
- 目标设备：Rokid智能眼镜
- 分辨率适配：横屏布局优化
- 触控交互：考虑设备特殊交互方式

### 📋 部署清单

部署前检查清单：

- [ ] ✅ 代码编译无错误
- [ ] ✅ 权限配置正确
- [ ] ✅ 网络连接测试通过
- [ ] ✅ 相机功能正常
- [ ] ✅ AI分析接口可用
- [ ] ✅ 横屏显示适配
- [ ] ✅ APK签名配置
- [ ] ✅ 性能测试通过
- [ ] ✅ 真机测试验证

---

### 🚨 构建状态更新 (2025-10-13)

#### 当前构建问题
- ✅ Gradle配置已更新到最新版本 (8.7, AGP 8.6.0, Kotlin 1.9.10)
- ✅ 已清理所有缓存和临时文件释放磁盘空间
- ✅ 项目依赖配置正确
- ❌ APK构建超时，网络下载依赖过慢

#### 当前解决方案
1. **简化构建命令**:
   ```bash
   flutter build apk --debug --target-platform=android-arm64
   ```

2. **使用Android Studio**:
   - 打开 `android/` 文件夹
   - 在IDE中执行Gradle构建

3. **网络优化建议**:
   - 配置国内Maven镜像源
   - 使用VPN或代理加速

**项目已准备就绪，需要优化网络环境完成APK构建！** 🔧