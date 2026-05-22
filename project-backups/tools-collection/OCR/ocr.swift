
import Vision
import AppKit
import Foundation

func performOCR(on imagePath: String) {
    guard let image = NSImage(contentsOfFile: imagePath),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("Error: 无法加载图片 \(imagePath)")
        return
    }
    
    let request = VNRecognizeTextRequest { request, error in
        if let error = error {
            print("OCR错误: \(error.localizedDescription)")
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
        print("处理请求时出错: \(error.localizedDescription)")
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
    print("错误: 文件不存在 \(imagePath)")
    exit(1)
}

// 执行OCR
performOCR(on: imagePath)

// 等待异步完成
RunLoop.main.run(until: Date(timeIntervalSinceNow: 5))
