import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import '../../models/product/product_info.dart';
import '../../services/commerce/commerce_service.dart';
import '../../services/ai_vision/vision_service.dart';
import '../../services/ar_display/ar_display_service.dart';
import '../../services/voice/voice_service.dart';

class ShoppingController extends ChangeNotifier {
  final CommerceService _commerceService;
  final VisionService _visionService;
  final ARDisplayService _arDisplayService;
  final VoiceService _voiceService;

  ShoppingController({
    required CommerceService commerceService,
    required VisionService visionService,
    required ARDisplayService arDisplayService,
    required VoiceService voiceService,
  })  : _commerceService = commerceService,
        _visionService = visionService,
        _arDisplayService = arDisplayService,
        _voiceService = voiceService;

  ProductInfo? _currentProduct;
  List<PriceComparison> _priceComparisons = [];
  String? _purchaseRecommendation;
  bool _isScanning = false;
  String? _errorMessage;

  ProductInfo? get currentProduct => _currentProduct;
  List<PriceComparison> get priceComparisons => _priceComparisons;
  String? get purchaseRecommendation => _purchaseRecommendation;
  bool get isScanning => _isScanning;
  String? get errorMessage => _errorMessage;

  Future<void> scanProductBarcode() async {
    try {
      _setScanning(true);
      _clearError();

      // 获取相机图像并识别条码
      final imageData = await _getCameraImage();
      final barcode = await _extractBarcode(imageData);
      
      if (barcode.isNotEmpty) {
        // 根据条码查询商品信息
        final product = await _commerceService.getProductByBarcode(barcode);
        
        if (product != null) {
          _currentProduct = product;
          
          // 获取价格比较
          _priceComparisons = await _commerceService.comparePrices(product.id);
          
          // 生成购买建议
          _purchaseRecommendation = await _commerceService.generatePurchaseRecommendation(product);
          
          // 显示商品信息
          await _displayProductInfo(product);
          
          // 语音播报
          await _speakProductInfo(product);
        } else {
          await _voiceService.textToSpeech('未找到该商品信息，请尝试其他商品');
        }
      } else {
        await _voiceService.textToSpeech('未识别到条码，请对准商品条码重试');
      }

      notifyListeners();
    } catch (e) {
      _setError('扫描商品失败: $e');
    } finally {
      _setScanning(false);
    }
  }

  Future<void> handleVoiceCommand(String command) async {
    try {
      final lowerCommand = command.toLowerCase();
      
      if (lowerCommand.contains('这个值得买') || lowerCommand.contains('扫描商品')) {
        await scanProductBarcode();
      } else if (lowerCommand.contains('哪里更便宜') && _currentProduct != null) {
        await _showPriceComparison();
      } else if (lowerCommand.contains('购买建议') && _currentProduct != null) {
        await _speakPurchaseRecommendation();
      } else if (lowerCommand.contains('商品详情') && _currentProduct != null) {
        await _showProductDetails();
      } else {
        await _voiceService.textToSpeech('请说"这个值得买吗"来扫描商品');
      }
    } catch (e) {
      _setError('处理语音命令失败: $e');
    }
  }

  Future<void> _displayProductInfo(ProductInfo product) async {
    // 显示商品名称
    await _arDisplayService.showText(
      product.name,
      position: ARPosition.centerTop(),
      style: ARStyle(fontSize: 18.0, fontWeight: FontWeight.bold),
    );

    // 显示品牌和价格
    await _arDisplayService.showText(
      '${product.brand} - ${product.priceDisplay}',
      position: ARPosition(x: 0.5, y: 0.25),
      style: ARStyle(fontSize: 16.0),
    );

    // 显示评分
    await _arDisplayService.showText(
      '⭐ ${product.rating} (${product.reviewCount}评价)',
      position: ARPosition(x: 0.5, y: 0.35),
      style: ARStyle(fontSize: 14.0, textColor: const Color(0xFFFFEB3B)),
    );

    // 显示折扣信息
    if (product.isOnSale) {
      await _arDisplayService.showText(
        '🎉 ${product.discountDisplay} 原价: ${product.originalPriceDisplay}',
        position: ARPosition(x: 0.5, y: 0.45),
        style: ARStyle(
          fontSize: 14.0,
          textColor: const Color(0xFF4CAF50),
          backgroundColor: const Color(0x80F44336),
        ),
      );
    }

    // 显示购买建议
    if (_purchaseRecommendation != null) {
      await _arDisplayService.showText(
        _purchaseRecommendation!,
        position: ARPosition(x: 0.5, y: 0.55),
        style: ARStyle(
          fontSize: 12.0,
          textColor: product.isRecommended ? const Color(0xFF4CAF50) : const Color(0xFFFF9800),
        ),
      );
    }

    // 显示最低价格平台
    if (_priceComparisons.isNotEmpty) {
      final lowestPrice = _priceComparisons.first;
      await _arDisplayService.showText(
        '💰 最低价: ${lowestPrice.platformName} ${lowestPrice.priceDisplay}',
        position: ARPosition(x: 0.5, y: 0.65),
        style: ARStyle(fontSize: 12.0, textColor: const Color(0xFF2196F3)),
      );
    }
  }

  Future<void> _speakProductInfo(ProductInfo product) async {
    String spokenText = '${product.brand}${product.name}';
    
    spokenText += '，当前价格${product.currentPrice.toInt()}元';
    
    if (product.isOnSale) {
      spokenText += '，正在打折，比原价便宜${product.discountPercentage.toInt()}%';
    }
    
    if (product.rating >= 4.0) {
      spokenText += '，用户评价${product.rating}分，评价不错';
    }
    
    if (_purchaseRecommendation != null) {
      spokenText += '。购买建议：$_purchaseRecommendation';
    }

    await _voiceService.textToSpeech(spokenText);
  }

  Future<void> _showPriceComparison() async {
    if (_priceComparisons.isEmpty) {
      await _voiceService.textToSpeech('暂时没有找到其他平台的价格信息');
      return;
    }

    await _arDisplayService.hideAllOverlays();
    
    // 显示价格比较标题
    await _arDisplayService.showText(
      '${_currentProduct!.name} - 价格比较',
      position: ARPosition(x: 0.5, y: 0.15),
      style: ARStyle(fontSize: 16.0, fontWeight: FontWeight.bold),
    );

    // 显示各平台价格
    for (int i = 0; i < _priceComparisons.length && i < 4; i++) {
      final comparison = _priceComparisons[i];
      final yPosition = 0.3 + (i * 0.1);
      
      await _arDisplayService.showText(
        '${comparison.platformName}: ${comparison.priceDisplay}',
        position: ARPosition(x: 0.5, y: yPosition),
        style: ARStyle(
          fontSize: 14.0,
          textColor: i == 0 ? const Color(0xFF4CAF50) : const Color(0xFFFFFFFF),
        ),
      );
    }

    // 语音播报价格比较
    final lowestPrice = _priceComparisons.first;
    final highestPrice = _priceComparisons.last;
    final priceDifference = highestPrice.price - lowestPrice.price;
    
    await _voiceService.textToSpeech(
      '价格比较结果：${lowestPrice.platformName}最便宜，价格${lowestPrice.priceDisplay}，'
      '比最高价便宜${priceDifference.toStringAsFixed(2)}元'
    );
  }

  Future<void> _speakPurchaseRecommendation() async {
    if (_purchaseRecommendation != null) {
      await _voiceService.textToSpeech(_purchaseRecommendation!);
    } else {
      await _voiceService.textToSpeech('暂时无法生成购买建议');
    }
  }

  Future<void> _showProductDetails() async {
    if (_currentProduct == null) return;

    await _arDisplayService.hideAllOverlays();
    
    final product = _currentProduct!;
    
    // 显示详细信息
    await _arDisplayService.showText(
      '${product.name} - 详细信息',
      position: ARPosition(x: 0.5, y: 0.15),
      style: ARStyle(fontSize: 16.0, fontWeight: FontWeight.bold),
    );
    
    await _arDisplayService.showText(
      '品牌: ${product.brand}',
      position: ARPosition(x: 0.5, y: 0.25),
      style: ARStyle(fontSize: 12.0),
    );
    
    await _arDisplayService.showText(
      '类别: ${product.category}',
      position: ARPosition(x: 0.5, y: 0.35),
      style: ARStyle(fontSize: 12.0),
    );
    
    await _arDisplayService.showText(
      product.description.length > 60
          ? '${product.description.substring(0, 57)}...'
          : product.description,
      position: ARPosition(x: 0.5, y: 0.45),
      style: ARStyle(fontSize: 11.0),
    );

    // 语音播报详细信息
    await _voiceService.textToSpeech(
      '${product.name}，${product.brand}品牌，属于${product.category}类别。${product.description}'
    );
  }

  Future<String> _extractBarcode(Uint8List imageData) async {
    // 模拟条码识别
    // 实际实现中应该使用专门的条码识别库
    await Future.delayed(const Duration(milliseconds: 800));
    
    // 模拟返回条码
    final mockBarcodes = [
      '6901028089296', // 可口可乐
      '6920146160227', // 农夫山泉
      '6902083191128', // 金龙鱼油
      '6921168518483', // 德芙巧克力
      '6925303711009', // 旺旺仙贝
    ];
    
    return mockBarcodes[DateTime.now().millisecond % mockBarcodes.length];
  }

  Future<Uint8List> _getCameraImage() async {
    // 模拟从相机获取图像数据
    await Future.delayed(const Duration(milliseconds: 500));
    return Uint8List.fromList([1, 2, 3, 4, 5]);
  }

  void _setScanning(bool scanning) {
    _isScanning = scanning;
    notifyListeners();
  }

  void _setError(String error) {
    _errorMessage = error;
    notifyListeners();
  }

  void _clearError() {
    _errorMessage = null;
    notifyListeners();
  }

  @override
  void dispose() {
    super.dispose();
  }
}