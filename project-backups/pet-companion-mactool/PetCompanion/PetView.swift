import SwiftUI

struct PetView: View {
    @ObservedObject var petModel: PetModel
    @ObservedObject var speechManager: SpeechManager
    @State private var bounceOffset: CGFloat = 0
    @State private var rotationAngle: Double = 0
    @State private var scale: CGFloat = 1.0
    
    var body: some View {
        VStack(spacing: 20) {
            // 宠物显示区域 - 悬浮样式
            ZStack {
                // 背景装饰 - 更小的圆形
                Circle()
                    .fill(
                        RadialGradient(
                            gradient: Gradient(colors: [
                                petModel.currentMood.color.opacity(0.2),
                                petModel.currentMood.color.opacity(0.05)
                            ]),
                            center: .center,
                            startRadius: 30,
                            endRadius: 80
                        )
                    )
                    .frame(width: 120, height: 120)
                    .scaleEffect(scale)
                    .animation(
                        petModel.isAnimating ? 
                        .easeInOut(duration: 1.0).repeatForever(autoreverses: true) : 
                        .easeInOut(duration: 0.3),
                        value: scale
                    )
                
                // 宠物主体
                VStack(spacing: 6) {
                    // 宠物图标 - 更小更精致
                    Text(petModel.selectedPetType.emoji)
                        .font(.system(size: 50))
                        .scaleEffect(scale)
                        .rotationEffect(.degrees(rotationAngle))
                        .offset(y: bounceOffset)
                        .animation(
                            petModel.isAnimating ? 
                            .easeInOut(duration: 0.8).repeatForever(autoreverses: true) : 
                            .easeInOut(duration: 0.3),
                            value: bounceOffset
                        )
                        .animation(
                            petModel.isAnimating ? 
                            .linear(duration: 2.0).repeatForever(autoreverses: false) : 
                            .easeInOut(duration: 0.3),
                            value: rotationAngle
                        )
                    
                    // 心情显示 - 简化版
                    Text(petModel.currentMood.emoji)
                        .font(.title3)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(
                            Capsule()
                                .fill(petModel.currentMood.color.opacity(0.2))
                        )
                }
            }
            
            // 宠物信息 - 简化版
            VStack(spacing: 4) {
                Text(petModel.petName)
                    .font(.caption)
                    .foregroundColor(.primary)
                
                if speechManager.isSpeaking {
                    HStack(spacing: 3) {
                        ForEach(0..<3) { index in
                            Circle()
                                .fill(Color.blue)
                                .frame(width: 4, height: 4)
                                .scaleEffect(speechManager.isSpeaking ? 1.2 : 0.8)
                                .animation(
                                    .easeInOut(duration: 0.6)
                                    .repeatForever(autoreverses: true)
                                    .delay(Double(index) * 0.2),
                                    value: speechManager.isSpeaking
                                )
                        }
                    }
                }
            }
        }
        .onAppear {
            startAnimations()
        }
        .onChange(of: petModel.isAnimating) { _, newValue in
            if newValue {
                startAnimations()
            } else {
                stopAnimations()
            }
        }
        .onChange(of: petModel.currentMood) { _, _ in
            // 心情改变时的特殊动画
            withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) {
                scale = 1.2
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                withAnimation(.spring(response: 0.6, dampingFraction: 0.8)) {
                    scale = 1.0
                }
            }
        }
    }
    
    private func startAnimations() {
        guard petModel.isAnimating else { return }
        
        withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
            bounceOffset = -10
        }
        
        withAnimation(.linear(duration: 2.0).repeatForever(autoreverses: false)) {
            rotationAngle = 360
        }
        
        withAnimation(.easeInOut(duration: 1.0).repeatForever(autoreverses: true)) {
            scale = 1.1
        }
    }
    
    private func stopAnimations() {
        withAnimation(.easeInOut(duration: 0.3)) {
            bounceOffset = 0
            rotationAngle = 0
            scale = 1.0
        }
    }
}

#Preview {
    PetView(
        petModel: PetModel(),
        speechManager: SpeechManager()
    )
    .frame(width: 400, height: 400)
}
