import SwiftUI

struct ContentView: View {
    @StateObject private var petModel = PetModel()
    @StateObject private var speechManager = SpeechManager()
    @State private var showingSettings = false
    @State private var dragOffset = CGSize.zero
    @State private var isDragging = false
    
    var body: some View {
        ZStack {
            // 背景渐变
            LinearGradient(
                gradient: Gradient(colors: [Color.blue.opacity(0.1), Color.purple.opacity(0.1)]),
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()
            
            VStack(spacing: 8) {
                // 宠物视图 - 主要显示区域
                PetView(petModel: petModel, speechManager: speechManager)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                
                // 简化的控制按钮
                HStack(spacing: 12) {
                    Button(action: {
                        petModel.toggleAnimation()
                    }) {
                        Image(systemName: petModel.isAnimating ? "pause.circle.fill" : "play.circle.fill")
                            .font(.title3)
                            .foregroundColor(.blue)
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    Button(action: {
                        let randomMessage = petModel.getRandomMessage()
                        speechManager.speak(text: randomMessage)
                    }) {
                        Image(systemName: "speaker.wave.2.fill")
                            .font(.title3)
                            .foregroundColor(.green)
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    Button(action: {
                        petModel.changeMood()
                    }) {
                        Image(systemName: "heart.fill")
                            .font(.title3)
                            .foregroundColor(.pink)
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    Button(action: {
                        showingSettings = true
                    }) {
                        Image(systemName: "gearshape.fill")
                            .font(.title3)
                            .foregroundColor(.gray)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 8)
            }
        }
        .frame(width: 200, height: 250)
        .background(
            RoundedRectangle(cornerRadius: 15)
                .fill(.ultraThinMaterial)
                .opacity(isDragging ? 0.8 : 0.6)
        )
        .cornerRadius(15)
        .shadow(color: .black.opacity(0.3), radius: 10, x: 0, y: 5)
        .offset(dragOffset)
        .scaleEffect(isDragging ? 1.05 : 1.0)
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: isDragging)
        .gesture(
            DragGesture()
                .onChanged { value in
                    isDragging = true
                    dragOffset = value.translation
                }
                .onEnded { value in
                    isDragging = false
                    withAnimation(.spring()) {
                        dragOffset = .zero
                    }
                }
        )
        .onTapGesture {
            // 点击宠物时的随机反应
            let reactions = [
                { petModel.changeMood() },
                { 
                    let message = petModel.getRandomMessage()
                    speechManager.speak(text: message)
                },
                { petModel.toggleAnimation() }
            ]
            reactions.randomElement()?()
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView(petModel: petModel, speechManager: speechManager)
        }
    }
}

#Preview {
    ContentView()
}
