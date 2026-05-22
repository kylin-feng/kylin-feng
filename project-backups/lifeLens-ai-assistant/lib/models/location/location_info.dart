import 'package:json_annotation/json_annotation.dart';

part 'location_info.g.dart';

@JsonSerializable()
class LocationInfo {
  final double latitude;
  final double longitude;
  final String address;
  final String? district;
  final String? city;
  final String? province;
  final String? country;
  final DateTime timestamp;

  LocationInfo({
    required this.latitude,
    required this.longitude,
    required this.address,
    this.district,
    this.city,
    this.province,
    this.country,
    required this.timestamp,
  });

  factory LocationInfo.fromJson(Map<String, dynamic> json) =>
      _$LocationInfoFromJson(json);

  Map<String, dynamic> toJson() => _$LocationInfoToJson(this);
}

@JsonSerializable()
class NavigationInfo {
  final LocationInfo origin;
  final LocationInfo destination;
  final double distance;
  final int duration;
  final String mode;
  final List<RouteStep> steps;
  final String overview;

  NavigationInfo({
    required this.origin,
    required this.destination,
    required this.distance,
    required this.duration,
    required this.mode,
    required this.steps,
    required this.overview,
  });

  factory NavigationInfo.fromJson(Map<String, dynamic> json) =>
      _$NavigationInfoFromJson(json);

  Map<String, dynamic> toJson() => _$NavigationInfoToJson(this);

  String get distanceDisplay => distance < 1000
      ? '${distance.toInt()}米'
      : '${(distance / 1000).toStringAsFixed(1)}公里';

  String get durationDisplay => duration < 60
      ? '${duration}分钟'
      : '${(duration / 60).toInt()}小时${duration % 60}分钟';
}

@JsonSerializable()
class RouteStep {
  final String instruction;
  final double distance;
  final int duration;
  final String direction;
  final LocationInfo startLocation;
  final LocationInfo endLocation;

  RouteStep({
    required this.instruction,
    required this.distance,
    required this.duration,
    required this.direction,
    required this.startLocation,
    required this.endLocation,
  });

  factory RouteStep.fromJson(Map<String, dynamic> json) =>
      _$RouteStepFromJson(json);

  Map<String, dynamic> toJson() => _$RouteStepToJson(this);
}

@JsonSerializable()
class POI {
  final String id;
  final String name;
  final String type;
  final LocationInfo location;
  final double distance;
  final String? phone;
  final String? address;
  final double? rating;
  final bool isOpen;
  final String? openingHours;

  POI({
    required this.id,
    required this.name,
    required this.type,
    required this.location,
    required this.distance,
    this.phone,
    this.address,
    this.rating,
    this.isOpen = true,
    this.openingHours,
  });

  factory POI.fromJson(Map<String, dynamic> json) =>
      _$POIFromJson(json);

  Map<String, dynamic> toJson() => _$POIToJson(this);
}