import 'package:json_annotation/json_annotation.dart';

part 'recognition_result.g.dart';

@JsonSerializable()
class RecognitionResult {
  final String objectName;
  final String category;
  final double confidence;
  final String description;
  final List<String> tags;
  final Map<String, dynamic>? metadata;
  final DateTime timestamp;

  RecognitionResult({
    required this.objectName,
    required this.category,
    required this.confidence,
    required this.description,
    required this.tags,
    this.metadata,
    required this.timestamp,
  });

  factory RecognitionResult.fromJson(Map<String, dynamic> json) =>
      _$RecognitionResultFromJson(json);

  Map<String, dynamic> toJson() => _$RecognitionResultToJson(this);
}

@JsonSerializable()
class ObjectInfo {
  final String name;
  final String category;
  final String description;
  final List<String> features;
  final String? wikipediaUrl;
  final String? imageUrl;
  final bool isPoisonous;
  final bool isEdible;
  final Map<String, dynamic>? additionalInfo;

  ObjectInfo({
    required this.name,
    required this.category,
    required this.description,
    required this.features,
    this.wikipediaUrl,
    this.imageUrl,
    this.isPoisonous = false,
    this.isEdible = false,
    this.additionalInfo,
  });

  factory ObjectInfo.fromJson(Map<String, dynamic> json) =>
      _$ObjectInfoFromJson(json);

  Map<String, dynamic> toJson() => _$ObjectInfoToJson(this);
}