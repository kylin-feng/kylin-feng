import SwiftUI

struct SettingsView: View {
    @ObservedObject var petModel: PetModel
    @ObservedObject var speechManager: SpeechManager
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            Form {
                Section("宠物设置") {
                    HStack {
                        Text("宠物名称")
                        Spacer()
                        TextField("输入宠物名称", text: $petModel.petName)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .frame(width: 150)
                    }
                    
                    Picker("宠物类型", selection: $petModel.selectedPetType) {
                        ForEach(PetModel.PetType.allCases, id: \.self) { type in
                            HStack {
                                Text(type.emoji)
                                Text(type.rawValue)
                            }
                            .tag(type)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                Section("语音设置") {
                    HStack {
                        Text("语速")
                        Spacer()
                        Slider(value: $speechManager.speechRate, in: 0.1...1.0, step: 0.1)
                            .frame(width: 150)
                        Text("\(Int(speechManager.speechRate * 100))%")
                            .frame(width: 40)
                    }
                    
                    HStack {
                        Text("音调")
                        Spacer()
                        Slider(value: $speechManager.speechPitch, in: 0.5...2.0, step: 0.1)
                            .frame(width: 150)
                        Text("\(Int(speechManager.speechPitch * 100))%")
                            .frame(width: 40)
                    }
                    
                    Picker("语音", selection: $speechManager.selectedVoice) {
                        ForEach(speechManager.getAvailableVoices(), id: \.1) { voice in
                            Text(voice.0).tag(voice.1)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                }
                
                Section("心情设置") {
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 10) {
                        ForEach(PetModel.PetMood.allCases, id: \.self) { mood in
                            Button(action: {
                                petModel.currentMood = mood
                            }) {
                                VStack {
                                    Text(mood.emoji)
                                        .font(.title2)
                                    Text(mood.rawValue)
                                        .font(.caption)
                                }
                                .padding(8)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                        .fill(petModel.currentMood == mood ? mood.color.opacity(0.3) : Color.gray.opacity(0.1))
                                )
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                }
                
                Section("测试语音") {
                    HStack {
                        Button("测试当前设置") {
                            speechManager.speak(text: "你好！我是\(petModel.petName)，今天心情\(petModel.currentMood.rawValue)！")
                        }
                        .buttonStyle(.borderedProminent)
                        
                        if speechManager.isSpeaking {
                            Button("停止") {
                                speechManager.stopSpeaking()
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                }
            }
            .navigationTitle("设置")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button("完成") {
                        dismiss()
                    }
                }
            }
        }
        .frame(width: 500, height: 600)
    }
}

#Preview {
    SettingsView(
        petModel: PetModel(),
        speechManager: SpeechManager()
    )
}
