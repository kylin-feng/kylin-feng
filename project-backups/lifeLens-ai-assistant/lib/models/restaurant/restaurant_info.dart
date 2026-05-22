import 'package:json_annotation/json_annotation.dart';

part 'restaurant_info.g.dart';

@JsonSerializable()
class RestaurantInfo {
  final String id;
  final String name;
  final double rating;
  final int reviewCount;
  final String address;
  final String phone;
  final double averagePrice;
  final String cuisine;
  final List<String> specialDishes;
  final String openingHours;
  final double distance;
  final List<String> tags;
  final String? imageUrl;
  final List<Review> reviews;

  RestaurantInfo({
    required this.id,
    required this.name,
    required this.rating,
    required this.reviewCount,
    required this.address,
    required this.phone,
    required this.averagePrice,
    required this.cuisine,
    required this.specialDishes,
    required this.openingHours,
    required this.distance,
    required this.tags,
    this.imageUrl,
    required this.reviews,
  });

  factory RestaurantInfo.fromJson(Map<String, dynamic> json) =>
      _$RestaurantInfoFromJson(json);

  Map<String, dynamic> toJson() => _$RestaurantInfoToJson(this);

  String get ratingDisplay => rating.toStringAsFixed(1);
  String get priceDisplay => '¥${averagePrice.toInt()}/人';
  String get distanceDisplay => distance < 1000 
      ? '${distance.toInt()}m' 
      : '${(distance / 1000).toStringAsFixed(1)}km';
}

@JsonSerializable()
class Review {
  final String userId;
  final String userName;
  final double rating;
  final String comment;
  final DateTime date;
  final List<String>? images;

  Review({
    required this.userId,
    required this.userName,
    required this.rating,
    required this.comment,
    required this.date,
    this.images,
  });

  factory Review.fromJson(Map<String, dynamic> json) =>
      _$ReviewFromJson(json);

  Map<String, dynamic> toJson() => _$ReviewToJson(this);
}