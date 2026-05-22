import AVFoundation
import Foundation

class SpeechManager: NSObject, ObservableObject {
    private let synthesizer = AVSpeechSynthesizer()
    @Published var isSpeaking = false
    @Published var speechRate: Float = 0.5
    @Published var speechPitch: Float = 1.0
    @Published var selectedVoice: String = "com.apple.voice.compact.zh-CN.TingTing"
    
    private let availableVoices = [
        ("TingTing", "com.apple.voice.compact.zh-CN.TingTing"),
        ("Sin-ji", "com.apple.voice.compact.zh-CN.Sin-ji"),
        ("Yu-shu", "com.apple.voice.compact.zh-CN.Yu-shu"),
        ("Alex", "com.apple.speech.voice.Alex"),
        ("Samantha", "com.apple.speech.voice.Samantha")
    ]
    
    override init() {
        super.init()
        setupAudioSession()
    }
    
    private func setupAudioSession() {
        // macOS不需要设置AVAudioSession
        // 语音合成在macOS上可以直接使用
    }
    
    func speak(text: String) {
        // 停止当前播放
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        
        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = speechRate
        utterance.pitchMultiplier = speechPitch
        utterance.voice = AVSpeechSynthesisVoice(identifier: selectedVoice)
        
        // 设置代理来跟踪播放状态
        synthesizer.delegate = self
        
        isSpeaking = true
        synthesizer.speak(utterance)
    }
    
    func stopSpeaking() {
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
    }
    
    func getAvailableVoices() -> [(String, String)] {
        return availableVoices
    }
}

extension SpeechManager: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        DispatchQueue.main.async {
            self.isSpeaking = true
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        DispatchQueue.main.async {
            self.isSpeaking = false
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        DispatchQueue.main.async {
            self.isSpeaking = false
        }
    }
}
