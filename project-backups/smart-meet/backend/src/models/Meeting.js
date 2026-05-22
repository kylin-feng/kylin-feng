// 会议模型
export class Meeting {
  constructor(data) {
    this.id = data.id;
    this.title = data.title;
    this.date = data.date;
    this.duration = data.duration || 0;
    this.participants = data.participants || [];
    this.status = data.status || 'upcoming'; // upcoming, live, completed
    this.recording = data.recording || null;
    this.transcription = data.transcription || [];
    this.audioData = data.audioData || [];
    this.createdAt = data.createdAt || new Date().toISOString();
    this.updatedAt = data.updatedAt || new Date().toISOString();
  }

  startRecording() {
    this.status = 'live';
    this.date = new Date().toISOString();
    this.updatedAt = new Date().toISOString();
  }

  stopRecording() {
    this.status = 'completed';
    this.duration = Date.now() - new Date(this.date).getTime();
    this.updatedAt = new Date().toISOString();
  }

  addTranscription(transcription) {
    this.transcription.push({
      id: Date.now().toString(),
      text: transcription.text,
      speaker: transcription.speaker,
      timestamp: transcription.timestamp || Date.now(),
      confidence: transcription.confidence || 1,
      createdAt: new Date().toISOString()
    });
    this.updatedAt = new Date().toISOString();
  }

  toJSON() {
    return {
      id: this.id,
      title: this.title,
      date: this.date,
      duration: this.duration,
      participants: this.participants,
      status: this.status,
      recording: this.recording,
      transcription: this.transcription,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt
    };
  }
}

export default Meeting;