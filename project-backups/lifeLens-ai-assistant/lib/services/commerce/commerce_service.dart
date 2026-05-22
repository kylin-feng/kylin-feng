import '../../models/product/product_info.dart';

abstract class CommerceService {
  Future<ProductInfo?> getProductByBarcode(String barcode);
  Future<List<PriceComparison>> comparePrices(String productId);
  Future<List<ProductInfo>> searchProducts(String query);
  Future<List<ProductReview>> getProductReviews(String productId);
  Future<String> generatePurchaseRecommendation(ProductInfo product);
}

class MultiPlatformCommerceService implements CommerceService {
  final List<PlatformService> platforms;
  
  MultiPlatformCommerceService({required this.platforms});

  @override
  Future<ProductInfo?> getProductByBarcode(String barcode) async {
    try {
      // 尝试从各个平台获取商品信息
      for (final platform in platforms) {
        try {
          final product = await platform.getProductByBarcode(barcode);
          if (product != null) {
            // 获取其他平台的价格对比
            final comparisons = await comparePrices(product.id);
            return product.copyWith(priceComparisons: comparisons);
          }
        } catch (e) {
          continue; // 如果一个平台失败，尝试下一个
        }
      }
      return null;
    } catch (e) {
      throw CommerceServiceException('获取商品信息失败: $e');
    }
  }

  @override
  Future<List<PriceComparison>> comparePrices(String productId) async {
    final comparisons = <PriceComparison>[];
    
    for (final platform in platforms) {
      try {
        final price = await platform.getProductPrice(productId);
        if (price != null) {
          comparisons.add(price);
        }
      } catch (e) {
        // 忽略单个平台的错误
        continue;
      }
    }
    
    // 按价格排序
    comparisons.sort((a, b) => a.price.compareTo(b.price));
    return comparisons;
  }

  @override
  Future<List<ProductInfo>> searchProducts(String query) async {
    final allProducts = <ProductInfo>[];
    
    for (final platform in platforms) {
      try {
        final products = await platform.searchProducts(query);
        allProducts.addAll(products);
      } catch (e) {
        continue;
      }
    }
    
    return allProducts;
  }

  @override
  Future<List<ProductReview>> getProductReviews(String productId) async {
    final allReviews = <ProductReview>[];
    
    for (final platform in platforms) {
      try {
        final reviews = await platform.getProductReviews(productId);
        allReviews.addAll(reviews);
      } catch (e) {
        continue;
      }
    }
    
    return allReviews;
  }

  @override
  Future<String> generatePurchaseRecommendation(ProductInfo product) async {
    try {
      final lowestPrice = product.priceComparisons.isNotEmpty
          ? product.priceComparisons.first
          : null;
      
      final isGoodDeal = product.isOnSale && product.discountPercentage > 20;
      final hasGoodRating = product.rating >= 4.0;
      
      if (isGoodDeal && hasGoodRating) {
        return '推荐购买！现在是近期低价，商品评价良好(${product.rating}分)。';
      } else if (lowestPrice != null && lowestPrice.price < product.currentPrice) {
        return '建议在${lowestPrice.platformName}购买，价格便宜¥${(product.currentPrice - lowestPrice.price).toStringAsFixed(2)}。';
      } else if (hasGoodRating) {
        return '商品质量不错(${product.rating}分)，价格合理，可以考虑购买。';
      } else {
        return '建议再看看其他选择，或等待更好的价格。';
      }
    } catch (e) {
      return '暂时无法生成购买建议，请稍后再试。';
    }
  }
}

abstract class PlatformService {
  Future<ProductInfo?> getProductByBarcode(String barcode);
  Future<PriceComparison?> getProductPrice(String productId);
  Future<List<ProductInfo>> searchProducts(String query);
  Future<List<ProductReview>> getProductReviews(String productId);
}

class JDPlatformService implements PlatformService {
  final String apiKey;
  final String secretKey;
  
  JDPlatformService({required this.apiKey, required this.secretKey});

  @override
  Future<ProductInfo?> getProductByBarcode(String barcode) async {
    // 模拟京东API调用
    await Future.delayed(Duration(seconds: 1));
    
    return ProductInfo(
      id: 'jd_$barcode',
      name: '苹果iPhone 15 Pro',
      brand: '苹果',
      category: '手机',
      barcode: barcode,
      currentPrice: 8999.0,
      originalPrice: 9999.0,
      rating: 4.6,
      reviewCount: 15680,
      description: '苹果iPhone 15 Pro，钛金属设计',
      images: [],
      specifications: ['6.1英寸', '钛金属', 'A17 Pro芯片'],
      priceComparisons: [],
      reviews: [],
    );
  }

  @override
  Future<PriceComparison?> getProductPrice(String productId) async {
    await Future.delayed(Duration(milliseconds: 500));
    
    return PriceComparison(
      platform: 'jd',
      platformName: '京东',
      price: 8999.0,
      url: 'https://item.jd.com/$productId',
    );
  }

  @override
  Future<List<ProductInfo>> searchProducts(String query) async {
    await Future.delayed(Duration(seconds: 1));
    return [];
  }

  @override
  Future<List<ProductReview>> getProductReviews(String productId) async {
    await Future.delayed(Duration(seconds: 1));
    return [];
  }
}

class TaobaoPlatformService implements PlatformService {
  final String apiKey;
  final String secretKey;
  
  TaobaoPlatformService({required this.apiKey, required this.secretKey});

  @override
  Future<ProductInfo?> getProductByBarcode(String barcode) async {
    // 模拟淘宝API调用
    await Future.delayed(Duration(seconds: 1));
    return null; // 淘宝可能没有这个商品
  }

  @override
  Future<PriceComparison?> getProductPrice(String productId) async {
    await Future.delayed(Duration(milliseconds: 500));
    
    return PriceComparison(
      platform: 'taobao',
      platformName: '淘宝',
      price: 8799.0,
      url: 'https://item.taobao.com/$productId',
    );
  }

  @override
  Future<List<ProductInfo>> searchProducts(String query) async {
    await Future.delayed(Duration(seconds: 1));
    return [];
  }

  @override
  Future<List<ProductReview>> getProductReviews(String productId) async {
    await Future.delayed(Duration(seconds: 1));
    return [];
  }
}

extension ProductInfoExtension on ProductInfo {
  ProductInfo copyWith({
    List<PriceComparison>? priceComparisons,
    String? recommendation,
    bool? isRecommended,
  }) {
    return ProductInfo(
      id: id,
      name: name,
      brand: brand,
      category: category,
      barcode: barcode,
      currentPrice: currentPrice,
      originalPrice: originalPrice,
      currency: currency,
      rating: rating,
      reviewCount: reviewCount,
      description: description,
      images: images,
      specifications: specifications,
      priceComparisons: priceComparisons ?? this.priceComparisons,
      reviews: reviews,
      isRecommended: isRecommended ?? this.isRecommended,
      recommendation: recommendation ?? this.recommendation,
    );
  }
}

class CommerceServiceException implements Exception {
  final String message;
  CommerceServiceException(this.message);
  
  @override
  String toString() => 'CommerceServiceException: $message';
}