#!/usr/bin/env python3
"""
macOS OCR工具 - 使用系统内置的Vision框架
无需安装额外依赖，支持中英文识别
"""

import subprocess
import sys
import os
from pathlib import Path

def create_swift_ocr_script():
    """创建Swift OCR脚本"""
    swift_script = '''
import Vision
import AppKit
import Foundation

func performOCR(on imagePath: String) {
    guard let image = NSImage(contentsOfFile: imagePath),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("Error: 无法加载图片 \\(imagePath)")
        return
    }
    
    let request = VNRecognizeTextRequest { request, error in
        if let error = error {
            print("OCR错误: \\(error.localizedDescription)")
            return
        }
        
        guard let observations = request.results as? [VNRecognizedTextObservation] else {
            print("没有识别到文字")
            return
        }
        
        print("=== OCR识别结果 ===")
        for observation in observations {
            guard let topCandidate = observation.topCandidates(1).first else { continue }
            print(topCandidate.string)
        }
        print("=== 识别完成 ===")
    }
    
    // 设置识别语言（支持中英文）
    request.recognitionLanguages = ["zh-Hans", "zh-Hant", "en"]
    request.recognitionLevel = .accurate
    
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    
    do {
        try handler.perform([request])
    } catch {
        print("处理请求时出错: \\(error.localizedDescription)")
    }
}

// 检查命令行参数
guard CommandLine.arguments.count > 1 else {
    print("使用方法: swift ocr.swift <图片路径>")
    print("支持格式: JPG, PNG, TIFF, BMP")
    exit(1)
}

let imagePath = CommandLine.arguments[1]

// 检查文件是否存在
guard FileManager.default.fileExists(atPath: imagePath) else {
    print("错误: 文件不存在 \\(imagePath)")
    exit(1)
}

// 执行OCR
performOCR(on: imagePath)

// 等待异步完成
RunLoop.main.run(until: Date(timeIntervalSinceNow: 5))
'''
    
    script_path = "/Users/shixianping/ocr.swift"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(swift_script)
    
    return script_path

def create_python_wrapper():
    """创建Python包装器"""
    wrapper_script = '''#!/usr/bin/env python3
import subprocess
import sys
import os
from pathlib import Path

def ocr_image(image_path):
    """使用macOS Vision框架进行OCR识别"""
    if not os.path.exists(image_path):
        print(f"错误: 图片文件不存在 {image_path}")
        return
    
    swift_script = "/Users/shixianping/ocr.swift"
    if not os.path.exists(swift_script):
        print("错误: OCR脚本未找到，请重新运行配置")
        return
    
    try:
        # 使用swift命令运行OCR
        result = subprocess.run([
            'swift', swift_script, image_path
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"OCR执行失败: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print("OCR超时，请检查图片文件")
    except FileNotFoundError:
        print("错误: Swift编译器未找到，请确保Xcode Command Line Tools已安装")
        print("安装命令: xcode-select --install")
    except Exception as e:
        print(f"执行OCR时出错: {e}")

def main():
    if len(sys.argv) != 2:
        print("macOS OCR工具")
        print("使用方法: python3 ocr.py <图片路径>")
        print("支持格式: JPG, PNG, TIFF, BMP")
        print("支持语言: 中文、英文")
        print()
        print("示例:")
        print("  python3 ocr.py screenshot.png")
        print("  python3 ocr.py ~/Documents/image.jpg")
        return
    
    image_path = sys.argv[1]
    
    # 转换为绝对路径
    image_path = os.path.abspath(image_path)
    
    ocr_image(image_path)

if __name__ == "__main__":
    main()
'''
    
    wrapper_path = "/Users/shixianping/ocr.py"
    with open(wrapper_path, 'w', encoding='utf-8') as f:
        f.write(wrapper_script)
    
    # 添加执行权限
    os.chmod(wrapper_path, 0o755)
    
    return wrapper_path

def check_xcode_tools():
    """检查Xcode Command Line Tools是否安装"""
    try:
        result = subprocess.run(['swift', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def main():
    print("🔧 配置macOS OCR工具...")
    print()
    
    # 检查Swift编译器
    if not check_xcode_tools():
        print("❌ Swift编译器未找到")
        print("请先安装Xcode Command Line Tools:")
        print("  xcode-select --install")
        print()
        print("安装完成后重新运行此脚本")
        return
    
    print("✅ Swift编译器已安装")
    
    # 创建OCR脚本
    print("📝 创建Swift OCR脚本...")
    swift_script = create_swift_ocr_script()
    print(f"✅ Swift脚本已创建: {swift_script}")
    
    # 创建Python包装器
    print("🐍 创建Python包装器...")
    python_script = create_python_wrapper()
    print(f"✅ Python脚本已创建: {python_script}")
    
    print()
    print("🎉 OCR工具配置完成！")
    print()
    print("📖 使用方法:")
    print("  python3 /Users/shixianping/ocr.py <图片路径>")
    print()
    print("🌟 特性:")
    print("  ✅ 使用macOS内置Vision框架")
    print("  ✅ 支持中英文识别")
    print("  ✅ 无需安装额外依赖")
    print("  ✅ 支持JPG、PNG、TIFF、BMP格式")
    print()
    print("💡 提示: 可以将此工具添加到PATH中方便使用")

if __name__ == "__main__":
    main()