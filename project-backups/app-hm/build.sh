#\!/bin/bash

# 华为鸿蒙项目构建脚本

echo "🚀 开始构建华为鸿蒙项目..."

# 设置环境变量
export HARMONY_HOME="/Users/shixianping/Library/Huawei/Sdk"
export NODE_HOME="/Applications/DevEco-Studio.app/Contents/tools/node"
export HVIGOR_HOME="/Applications/DevEco-Studio.app/Contents/tools/hvigor"

# 使用DevEco Studio的工具构建
echo "📱 项目路径: $(pwd)"
echo "🔧 工具路径: $HVIGOR_HOME"

# 在DevEco Studio中构建的步骤
echo "
请在DevEco Studio中完成构建:

1. 确保项目已在DevEco Studio中打开
2. 点击菜单: Build → Build Hap(s)/App(s)
3. 等待构建完成
4. 构建成功后，HAP文件将在: build/default/outputs/default/

如果构建遇到问题:
- 尝试 Build → Clean Project
- 检查 File → Project Structure → SDK Location
- 确保签名配置正确
"

echo "✅ 构建准备完成！请在DevEco Studio中执行构建操作。"

