#!/usr/bin/env python3
"""
设置纯净粘贴快捷键
一键设置，永久使用
"""

import subprocess
import os

def create_automator_service():
    """创建Automator服务"""
    
    # 创建AppleScript内容
    applescript_content = '''
on run {input, parameters}
    -- 获取剪贴板内容并转换为纯文本
    set the clipboard to (the clipboard as string)
    return input
end run
'''
    
    # 创建Automator workflow
    workflow_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>AMApplicationBuild</key>
    <string>519</string>
    <key>AMApplicationVersion</key>
    <string>2.10</string>
    <key>AMDocumentVersion</key>
    <string>2</string>
    <key>actions</key>
    <array>
        <dict>
            <key>action</key>
            <dict>
                <key>AMAccepts</key>
                <dict>
                    <key>Container</key>
                    <string>List</string>
                    <key>Optional</key>
                    <true/>
                    <key>Types</key>
                    <array>
                        <string>com.apple.applescript.object</string>
                    </array>
                </dict>
                <key>AMActionVersion</key>
                <string>1.0.2</string>
                <key>AMApplication</key>
                <array>
                    <string>Automator</string>
                </array>
                <key>AMParameterProperties</key>
                <dict>
                    <key>source</key>
                    <dict>
                        <key>tokenizedValue</key>
                        <array>
                            <string>{applescript_content}</string>
                        </array>
                    </dict>
                </dict>
                <key>AMProvides</key>
                <dict>
                    <key>Container</key>
                    <string>List</string>
                    <key>Types</key>
                    <array>
                        <string>com.apple.applescript.object</string>
                    </array>
                </dict>
                <key>ActionBundlePath</key>
                <string>/System/Library/Automator/Run AppleScript.action</string>
                <key>ActionName</key>
                <string>Run AppleScript</string>
                <key>ActionParameters</key>
                <dict>
                    <key>source</key>
                    <string>{applescript_content}</string>
                </dict>
                <key>BundleIdentifier</key>
                <string>com.apple.Automator.RunScript</string>
                <key>CFBundleVersion</key>
                <string>1.0.2</string>
                <key>CanShowSelectedItemsWhenRun</key>
                <false/>
                <key>CanShowWhenRun</key>
                <true/>
                <key>Category</key>
                <array>
                    <string>AMCategoryUtilities</string>
                </array>
                <key>Class Name</key>
                <string>RunScriptAction</string>
                <key>InputUUID</key>
                <string>12345678-1234-1234-1234-123456789012</string>
                <key>Keywords</key>
                <array>
                    <string>Run</string>
                </array>
                <key>OutputUUID</key>
                <string>12345678-1234-1234-1234-123456789013</string>
                <key>UUID</key>
                <string>12345678-1234-1234-1234-123456789014</string>
                <key>UnlocalizedApplications</key>
                <array>
                    <string>Automator</string>
                </array>
                <key>arguments</key>
                <dict>
                    <key>0</key>
                    <dict>
                        <key>default value</key>
                        <string>on run {{input, parameters}}
	
	return input
end run</string>
                        <key>name</key>
                        <string>source</string>
                        <key>required</key>
                        <string>0</string>
                        <key>type</key>
                        <string>0</string>
                        <key>uuid</key>
                        <string>0</string>
                    </dict>
                </dict>
                <key>isViewVisible</key>
                <true/>
                <key>location</key>
                <string>449.000000:316.000000</string>
                <key>nibPath</key>
                <string>/System/Library/Automator/Run AppleScript.action/Contents/Resources/Base.lproj/main.nib</string>
            </dict>
        </dict>
    </array>
    <key>connectors</key>
    <dict/>
    <key>workflowMetaData</key>
    <dict>
        <key>workflowTypeIdentifier</key>
        <string>com.apple.Automator.servicesMenu</string>
    </dict>
</dict>
</plist>'''
    
    # 创建服务目录
    services_dir = os.path.expanduser("~/Library/Services")
    if not os.path.exists(services_dir):
        os.makedirs(services_dir)
    
    service_path = f"{services_dir}/Plain Paste.workflow"
    if not os.path.exists(service_path):
        os.makedirs(service_path)
    
    # 写入workflow文件
    with open(f"{service_path}/Contents.plist", 'w') as f:
        f.write(workflow_content)
    
    return service_path

def create_simple_app():
    """创建简单的App来实现纯净粘贴"""
    
    # 创建AppleScript应用
    applescript = '''
-- 纯净粘贴工具
try
    set currentClipboard to (the clipboard as string)
    set the clipboard to currentClipboard
    display notification "✅ 剪贴板已转换为纯文本" with title "Plain Paste"
on error
    display notification "❌ 剪贴板转换失败" with title "Plain Paste"
end try
'''
    
    # 保存AppleScript
    script_path = "/Users/shixianping/PlainPaste.scpt"
    with open("/tmp/plain_paste.applescript", 'w') as f:
        f.write(applescript)
    
    # 编译AppleScript
    try:
        subprocess.run([
            'osacompile', 
            '-o', script_path,
            '/tmp/plain_paste.applescript'
        ], check=True)
        
        print(f"✅ AppleScript已创建: {script_path}")
        return script_path
    except subprocess.CalledProcessError:
        print("❌ AppleScript编译失败")
        return None

def show_setup_instructions():
    """显示设置说明"""
    print("""
🎉 纯净粘贴工具设置完成！

📖 使用方法：

方法1️⃣ - 双击运行 (最简单)
  双击桌面上的 PlainPaste.scpt
  
方法2️⃣ - 终端运行
  osascript /Users/shixianping/PlainPaste.scpt

方法3️⃣ - 设置快捷键 (推荐)
  1. 打开 系统偏好设置 → 键盘 → 快捷键
  2. 选择 App快捷键
  3. 点击 + 添加
  4. 应用程序: 选择 所有应用程序
  5. 菜单标题: Plain Paste  
  6. 键盘快捷键: Cmd+Shift+Option+V
  
🚀 工作流程：
  1. 复制带格式的内容 (Cmd+C)
  2. 运行纯净粘贴工具 (双击或快捷键)
  3. 粘贴纯文本 (Cmd+V)

💡 小贴士：
  可以把PlainPaste.scpt拖到Dock上方便使用！
""")

def main():
    print("🔧 设置超方便的纯净粘贴工具...")
    print()
    
    # 创建AppleScript应用
    script_path = create_simple_app()
    
    if script_path:
        # 复制到桌面方便使用
        desktop_path = os.path.expanduser("~/Desktop/PlainPaste.scpt")
        try:
            subprocess.run(['cp', script_path, desktop_path], check=True)
            print(f"✅ 已复制到桌面: {desktop_path}")
        except:
            pass
        
        show_setup_instructions()
    else:
        print("❌ 设置失败")

if __name__ == "__main__":
    main()