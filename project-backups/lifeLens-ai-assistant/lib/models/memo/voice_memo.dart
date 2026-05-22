import 'package:json_annotation/json_annotation.dart';

part 'voice_memo.g.dart';

@JsonSerializable()
class VoiceMemo {
  final String id;
  final String content;
  final String? originalAudio;
  final MemoCategory category;
  final DateTime createdAt;
  final DateTime? reminderTime;
  final bool isCompleted;
  final List<String> tags;
  final LocationInfo? location;
  final Priority priority;

  VoiceMemo({
    required this.id,
    required this.content,
    this.originalAudio,
    required this.category,
    required this.createdAt,
    this.reminderTime,
    this.isCompleted = false,
    required this.tags,
    this.location,
    this.priority = Priority.medium,
  });

  factory VoiceMemo.fromJson(Map<String, dynamic> json) =>
      _$VoiceMemoFromJson(json);

  Map<String, dynamic> toJson() => _$VoiceMemoToJson(this);
}

@JsonEnum()
enum MemoCategory {
  @JsonValue('todo')
  todo,
  @JsonValue('reminder')
  reminder,
  @JsonValue('note')
  note,
  @JsonValue('idea')
  idea,
  @JsonValue('shopping')
  shopping,
  @JsonValue('work')
  work,
  @JsonValue('personal')
  personal,
}

@JsonEnum()
enum Priority {
  @JsonValue('low')
  low,
  @JsonValue('medium')
  medium,
  @JsonValue('high')
  high,
  @JsonValue('urgent')
  urgent,
}

extension MemoCategoryExtension on MemoCategory {
  String get displayName {
    switch (this) {
      case MemoCategory.todo:
        return '待办事项';
      case MemoCategory.reminder:
        return '提醒';
      case MemoCategory.note:
        return '笔记';
      case MemoCategory.idea:
        return '想法';
      case MemoCategory.shopping:
        return '购物';
      case MemoCategory.work:
        return '工作';
      case MemoCategory.personal:
        return '个人';
    }
  }
}

extension PriorityExtension on Priority {
  String get displayName {
    switch (this) {
      case Priority.low:
        return '低';
      case Priority.medium:
        return '中';
      case Priority.high:
        return '高';
      case Priority.urgent:
        return '紧急';
    }
  }
}

@JsonSerializable()
class LocationInfo {
  final double latitude;
  final double longitude;
  final String? address;

  LocationInfo({
    required this.latitude,
    required this.longitude,
    this.address,
  });

  factory LocationInfo.fromJson(Map<String, dynamic> json) =>
      _$LocationInfoFromJson(json);

  Map<String, dynamic> toJson() => _$LocationInfoToJson(this);
}