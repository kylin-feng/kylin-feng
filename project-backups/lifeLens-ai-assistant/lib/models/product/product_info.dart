import 'package:json_annotation/json_annotation.dart';

part 'product_info.g.dart';

@JsonSerializable()
class ProductInfo {
  final String id;
  final String name;
  final String brand;
  final String category;
  final String barcode;
  final double currentPrice;
  final double originalPrice;
  final String currency;
  final double rating;
  final int reviewCount;
  final String description;
  final List<String> images;
  final List<String> specifications;
  final List<PriceComparison> priceComparisons;
  final List<ProductReview> reviews;
  final bool isRecommended;
  final String? recommendation;

  ProductInfo({
    required this.id,
    required this.name,
    required this.brand,
    required this.category,
    required this.barcode,
    required this.currentPrice,
    required this.originalPrice,
    this.currency = 'CNY',
    required this.rating,
    required this.reviewCount,
    required this.description,
    required this.images,
    required this.specifications,
    required this.priceComparisons,
    required this.reviews,
    this.isRecommended = false,
    this.recommendation,
  });

  factory ProductInfo.fromJson(Map<String, dynamic> json) =>
      _$ProductInfoFromJson(json);

  Map<String, dynamic> toJson() => _$ProductInfoToJson(this);

  String get priceDisplay => '¥${currentPrice.toStringAsFixed(2)}';
  String get originalPriceDisplay => '¥${originalPrice.toStringAsFixed(2)}';
  bool get isOnSale => currentPrice < originalPrice;
  double get discountPercentage => 
      ((originalPrice - currentPrice) / originalPrice * 100);
  String get discountDisplay => 
      isOnSale ? '${discountPercentage.toInt()}% OFF' : '';
}

@JsonSerializable()
class PriceComparison {
  final String platform;
  final String platformName;
  final double price;
  final String url;
  final bool isAvailable;
  final String? shipping;

  PriceComparison({
    required this.platform,
    required this.platformName,
    required this.price,
    required this.url,
    this.isAvailable = true,
    this.shipping,
  });

  factory PriceComparison.fromJson(Map<String, dynamic> json) =>
      _$PriceComparisonFromJson(json);

  Map<String, dynamic> toJson() => _$PriceComparisonToJson(this);

  String get priceDisplay => '¥${price.toStringAsFixed(2)}';
}

@JsonSerializable()
class ProductReview {
  final String userId;
  final String userName;
  final double rating;
  final String comment;
  final DateTime date;
  final List<String>? images;
  final List<String>? pros;
  final List<String>? cons;

  ProductReview({
    required this.userId,
    required this.userName,
    required this.rating,
    required this.comment,
    required this.date,
    this.images,
    this.pros,
    this.cons,
  });

  factory ProductReview.fromJson(Map<String, dynamic> json) =>
      _$ProductReviewFromJson(json);

  Map<String, dynamic> toJson() => _$ProductReviewToJson(this);
}