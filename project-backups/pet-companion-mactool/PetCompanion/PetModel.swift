import Foundation
import SwiftUI

class PetModel: ObservableObject {
    @Published var isAnimating = false
    @Published var currentMood: PetMood = .happy
    @Published var petName = "小可爱"
    @Published var selectedPetType: PetType = .cat
    
    enum PetMood: String, CaseIterable {
        case happy = "开心"
        case sad = "难过"
        case excited = "兴奋"
        case sleepy = "困倦"
        case angry = "生气"
        
        var emoji: String {
            switch self {
            case .happy: return "😊"
            case .sad: return "😢"
            case .excited: return "🤩"
            case .sleepy: return "😴"
            case .angry: return "😠"
            }
        }
        
        var color: Color {
            switch self {
            case .happy: return .yellow
            case .sad: return .blue
            case .excited: return .orange
            case .sleepy: return .purple
            case .angry: return .red
            }
        }
    }
    
    enum PetType: String, CaseIterable {
        case cat = "猫咪"
        case dog = "小狗"
        case rabbit = "兔子"
        case bird = "小鸟"
        
        var emoji: String {
            switch self {
            case .cat: return "🐱"
            case .dog: return "🐶"
            case .rabbit: return "🐰"
            case .bird: return "🐦"
            }
        }
    }
    
    private let messages = [
        "你好！我是你的小宠物！",
        "今天天气真不错呢！",
        "陪我玩一会儿吧！",
        "我有点饿了...",
        "主人，你工作辛苦了！",
        "我想睡觉了...",
        "我们一起听音乐吧！",
        "今天心情很好！",
        "你能摸摸我吗？",
        "我学会了新技能！"
    ]
    
    func toggleAnimation() {
        withAnimation(.easeInOut(duration: 0.3)) {
            isAnimating.toggle()
        }
    }
    
    func changeMood() {
        let moods = PetMood.allCases
        if let currentIndex = moods.firstIndex(of: currentMood) {
            let nextIndex = (currentIndex + 1) % moods.count
            withAnimation(.easeInOut(duration: 0.5)) {
                currentMood = moods[nextIndex]
            }
        }
    }
    
    func getRandomMessage() -> String {
        return messages.randomElement() ?? "你好！"
    }
    
    func getMoodMessage() -> String {
        switch currentMood {
        case .happy:
            return "我很开心！"
        case .sad:
            return "我有点难过..."
        case .excited:
            return "太兴奋了！"
        case .sleepy:
            return "我想睡觉..."
        case .angry:
            return "我生气了！"
        }
    }
}
