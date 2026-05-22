import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'shopping_controller.dart';
import '../../models/product/product_info.dart';

class ShoppingPage extends StatefulWidget {
  const ShoppingPage({Key? key}) : super(key: key);

  @override
  State<ShoppingPage> createState() => _ShoppingPageState();
}

class _ShoppingPageState extends State<ShoppingPage> {
  late ShoppingController _controller;

  @override
  void initState() {
    super.initState();
    _controller = ShoppingController(
      commerceService: context.read(),
      visionService: context.read(),
      arDisplayService: context.read(),
      voiceService: context.read(),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider.value(
      value: _controller,
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Consumer<ShoppingController>(
          builder: (context, controller, child) {
            return Stack(
              children: [
                // 相机预览区域
                const CameraPreviewWidget(),
                
                // 商品信息AR叠加
                if (controller.currentProduct != null)
                  ProductAROverlayWidget(
                    product: controller.currentProduct!,
                    priceComparisons: controller.priceComparisons,
                    recommendation: controller.purchaseRecommendation,
                  ),
                
                // 加载指示器
                if (controller.isScanning)
                  const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(color: Colors.green),
                        SizedBox(height: 16),
                        Text(
                          '正在扫描商品...',
                          style: TextStyle(color: Colors.white),
                        ),
                      ],
                    ),
                  ),
                
                // 错误信息
                if (controller.errorMessage != null)
                  Positioned(
                    bottom: 100,
                    left: 20,
                    right: 20,
                    child: Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.red.withOpacity(0.8),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        controller.errorMessage!,
                        style: const TextStyle(color: Colors.white),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                
                // 控制按钮
                Positioned(
                  bottom: 30,
                  left: 0,
                  right: 0,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      FloatingActionButton(
                        onPressed: controller.isScanning 
                            ? null 
                            : controller.scanProductBarcode,
                        backgroundColor: Colors.green,
                        child: const Icon(Icons.qr_code_scanner),
                      ),
                      if (controller.currentProduct != null)
                        FloatingActionButton(
                          onPressed: () => _showProductDetails(controller.currentProduct!),
                          backgroundColor: Colors.blue,
                          child: const Icon(Icons.info),
                        ),
                      if (controller.priceComparisons.isNotEmpty)
                        FloatingActionButton(
                          onPressed: () => _showPriceComparison(controller.priceComparisons),
                          backgroundColor: Colors.orange,
                          child: const Icon(Icons.compare_arrows),
                        ),
                    ],
                  ),
                ),
                
                // 语音指示
                const Positioned(
                  top: 50,
                  left: 0,
                  right: 0,
                  child: ShoppingVoiceIndicator(),
                ),
                
                // 扫描指示
                if (!controller.isScanning)
                  const Positioned(
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    child: ScanOverlayWidget(),
                  ),
              ],
            );
          },
        ),
      ),
    );
  }

  void _showProductDetails(ProductInfo product) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.black.withOpacity(0.9),
      isScrollControlled: true,
      builder: (context) => ProductDetailSheet(product: product),
    );
  }

  void _showPriceComparison(List<PriceComparison> comparisons) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.black.withOpacity(0.9),
      builder: (context) => PriceComparisonSheet(comparisons: comparisons),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}

class CameraPreviewWidget extends StatelessWidget {
  const CameraPreviewWidget({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: double.infinity,
      color: Colors.grey[900],
      child: const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.qr_code_scanner,
              size: 100,
              color: Colors.grey,
            ),
            SizedBox(height: 20),
            Text(
              '商品扫描预览',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 18,
              ),
            ),
            SizedBox(height: 10),
            Text(
              '对准商品条码或说"这个值得买吗"',
              style: TextStyle(
                color: Colors.grey,
                fontSize: 14,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class ScanOverlayWidget extends StatefulWidget {
  const ScanOverlayWidget({Key? key}) : super(key: key);

  @override
  State<ScanOverlayWidget> createState() => _ScanOverlayWidgetState();
}

class _ScanOverlayWidgetState extends State<ScanOverlayWidget>
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
    _animationController.repeat(reverse: true);
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Stack(
          children: [
            // 扫描框
            Center(
              child: Container(
                width: 250,
                height: 250,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.green, width: 2),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Stack(
                  children: [
                    // 角落装饰
                    Positioned(
                      top: 10,
                      left: 10,
                      child: Container(
                        width: 20,
                        height: 20,
                        decoration: const BoxDecoration(
                          border: Border(
                            top: BorderSide(color: Colors.green, width: 4),
                            left: BorderSide(color: Colors.green, width: 4),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      top: 10,
                      right: 10,
                      child: Container(
                        width: 20,
                        height: 20,
                        decoration: const BoxDecoration(
                          border: Border(
                            top: BorderSide(color: Colors.green, width: 4),
                            right: BorderSide(color: Colors.green, width: 4),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      bottom: 10,
                      left: 10,
                      child: Container(
                        width: 20,
                        height: 20,
                        decoration: const BoxDecoration(
                          border: Border(
                            bottom: BorderSide(color: Colors.green, width: 4),
                            left: BorderSide(color: Colors.green, width: 4),
                          ),
                        ),
                      ),
                    ),
                    Positioned(
                      bottom: 10,
                      right: 10,
                      child: Container(
                        width: 20,
                        height: 20,
                        decoration: const BoxDecoration(
                          border: Border(
                            bottom: BorderSide(color: Colors.green, width: 4),
                            right: BorderSide(color: Colors.green, width: 4),
                          ),
                        ),
                      ),
                    ),
                    // 扫描线
                    Positioned(
                      top: 20 + (_animation.value * 190),
                      left: 20,
                      right: 20,
                      child: Container(
                        height: 2,
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              Colors.transparent,
                              Colors.green.withOpacity(0.8),
                              Colors.transparent,
                            ],
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.green.withOpacity(0.5),
                              blurRadius: 10,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            // 扫描提示
            Positioned(
              bottom: 150,
              left: 0,
              right: 0,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: const Text(
                  '将商品条码对准扫描框',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class ProductAROverlayWidget extends StatelessWidget {
  final ProductInfo product;
  final List<PriceComparison> priceComparisons;
  final String? recommendation;
  
  const ProductAROverlayWidget({
    Key? key,
    required this.product,
    required this.priceComparisons,
    this.recommendation,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 120,
      left: 20,
      right: 20,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.8),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.green, width: 2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // 商品名称和品牌
            Text(
              product.name,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
            const SizedBox(height: 4),
            Text(
              product.brand,
              style: const TextStyle(
                color: Colors.grey,
                fontSize: 14,
              ),
            ),
            const SizedBox(height: 12),
            
            // 价格和评分
            Row(
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          product.priceDisplay,
                          style: const TextStyle(
                            color: Colors.green,
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        if (product.isOnSale) ...[
                          const SizedBox(width: 8),
                          Text(
                            product.originalPriceDisplay,
                            style: const TextStyle(
                              color: Colors.grey,
                              fontSize: 14,
                              decoration: TextDecoration.lineThrough,
                            ),
                          ),
                        ],
                      ],
                    ),
                    if (product.isOnSale)
                      Container(
                        margin: const EdgeInsets.only(top: 4),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.red,
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          product.discountDisplay,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                  ],
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: _getRatingColor(product.rating),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.star,
                        color: Colors.white,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        product.rating.toString(),
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            
            // 评价数量
            Text(
              '${product.reviewCount}条评价',
              style: const TextStyle(
                color: Colors.grey,
                fontSize: 12,
              ),
            ),
            
            // 购买建议
            if (recommendation != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: product.isRecommended 
                      ? Colors.green.withOpacity(0.3) 
                      : Colors.orange.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(
                      product.isRecommended ? Icons.thumb_up : Icons.info,
                      color: product.isRecommended ? Colors.green : Colors.orange,
                      size: 16,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        recommendation!,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
            
            // 价格比较预览
            if (priceComparisons.isNotEmpty) ...[
              const SizedBox(height: 8),
              const Text(
                '价格比较:',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 4),
              Row(
                children: [
                  Icon(
                    Icons.trending_down,
                    color: Colors.green,
                    size: 16,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    '${priceComparisons.first.platformName}: ${priceComparisons.first.priceDisplay}',
                    style: const TextStyle(
                      color: Colors.green,
                      fontSize: 11,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    '共${priceComparisons.length}个平台',
                    style: const TextStyle(
                      color: Colors.grey,
                      fontSize: 10,
                    ),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Color _getRatingColor(double rating) {
    if (rating >= 4.5) return Colors.green;
    if (rating >= 4.0) return Colors.orange;
    if (rating >= 3.5) return Colors.amber;
    return Colors.red;
  }
}

class ShoppingVoiceIndicator extends StatefulWidget {
  const ShoppingVoiceIndicator({Key? key}) : super(key: key);

  @override
  State<ShoppingVoiceIndicator> createState() => _ShoppingVoiceIndicatorState();
}

class _ShoppingVoiceIndicatorState extends State<ShoppingVoiceIndicator>
    with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    );
    _animation = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(parent: _animationController, curve: Curves.easeInOut),
    );
    _animationController.repeat(reverse: true);
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _animation,
      builder: (context, child) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Colors.green.withOpacity(_animation.value * 0.7),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      Icons.mic,
                      color: Colors.white,
                      size: 16,
                    ),
                    SizedBox(width: 8),
                    Text(
                      '说"这个值得买吗"扫描商品',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class ProductDetailSheet extends StatelessWidget {
  final ProductInfo product;
  
  const ProductDetailSheet({
    Key? key,
    required this.product,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  product.name,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.close, color: Colors.white),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          Text(
            product.description,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 14,
              height: 1.5,
            ),
          ),
          const SizedBox(height: 16),
          
          _buildDetailRow('品牌', product.brand),
          _buildDetailRow('类别', product.category),
          _buildDetailRow('当前价格', product.priceDisplay),
          if (product.isOnSale)
            _buildDetailRow('原价', product.originalPriceDisplay),
          _buildDetailRow('评分', '${product.rating} (${product.reviewCount}条评价)'),
          
          if (product.specifications.isNotEmpty) ...[
            const SizedBox(height: 16),
            const Text(
              '规格参数:',
              style: TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            ...product.specifications.map((spec) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Row(
                  children: [
                    const Icon(Icons.check, color: Colors.green, size: 16),
                    const SizedBox(width: 8),
                    Text(
                      spec,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 14,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ],
          
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 80,
            child: Text(
              '$label:',
              style: const TextStyle(
                color: Colors.white70,
                fontSize: 14,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 14,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class PriceComparisonSheet extends StatelessWidget {
  final List<PriceComparison> comparisons;
  
  const PriceComparisonSheet({
    Key? key,
    required this.comparisons,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              const Text(
                '价格比较',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Spacer(),
              IconButton(
                onPressed: () => Navigator.pop(context),
                icon: const Icon(Icons.close, color: Colors.white),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          ...comparisons.asMap().entries.map((entry) {
            final index = entry.key;
            final comparison = entry.value;
            final isLowest = index == 0;
            
            return Container(
              margin: const EdgeInsets.only(bottom: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: isLowest 
                    ? Colors.green.withOpacity(0.2) 
                    : Colors.grey.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
                border: isLowest 
                    ? Border.all(color: Colors.green) 
                    : null,
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          comparison.platformName,
                          style: TextStyle(
                            color: isLowest ? Colors.green : Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        if (comparison.shipping != null)
                          Text(
                            comparison.shipping!,
                            style: const TextStyle(
                              color: Colors.grey,
                              fontSize: 12,
                            ),
                          ),
                      ],
                    ),
                  ),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        comparison.priceDisplay,
                        style: TextStyle(
                          color: isLowest ? Colors.green : Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      if (isLowest)
                        const Text(
                          '最低价',
                          style: TextStyle(
                            color: Colors.green,
                            fontSize: 10,
                          ),
                        ),
                    ],
                  ),
                ],
              ),
            );
          }).toList(),
          
          const SizedBox(height: 20),
        ],
      ),
    );
  }
}