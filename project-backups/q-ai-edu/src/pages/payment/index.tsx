import { View, Text, Image, Button } from "@tarojs/components";
import { useLoad, useRouter } from "@tarojs/taro";
import Taro from "@tarojs/taro";
import { useState } from "react";
import "./index.scss";

export default function Payment() {
  const router = useRouter();
  const { courseId, price } = router.params;
  
  const [paymentMethod, setPaymentMethod] = useState<'wechat' | 'alipay'>('wechat');
  const [orderInfo, setOrderInfo] = useState<Course.OrderInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);

  // 模拟订单信息
  const mockOrderInfo: Course.OrderInfo = {
    id: `order_${Date.now()}`,
    userId: "user123",
    courseId: courseId || "1",
    courseTitle: "React全栈开发实战",
    courseCover: require("../../images/worker.jpg"),
    originalPrice: 299,
    actualPrice: Number(price) || 199,
    discountAmount: 100,
    paymentMethod: paymentMethod,
    status: "pending",
    createTime: new Date().toISOString(),
    expireTime: new Date(Date.now() + 30 * 60 * 1000).toISOString() // 30分钟后过期
  };

  useLoad(() => {
    console.log("支付页面加载");
    console.log("参数:", { courseId, price });
    loadOrderInfo();
  });

  const loadOrderInfo = () => {
    setLoading(true);
    // 模拟API调用
    setTimeout(() => {
      setOrderInfo(mockOrderInfo);
      setLoading(false);
    }, 500);
  };

  const handlePaymentMethodChange = (method: 'wechat' | 'alipay') => {
    setPaymentMethod(method);
    if (orderInfo) {
      setOrderInfo({
        ...orderInfo,
        paymentMethod: method
      });
    }
  };

  const handlePay = async () => {
    if (!orderInfo) return;

    setProcessing(true);
    
    try {
      // 模拟支付流程
      if (paymentMethod === 'wechat') {
        // 调用微信支付
        await Taro.requestPayment({
          timeStamp: String(Date.now()),
          nonceStr: 'test_nonce_str',
          package: 'prepay_id=test_prepay_id',
          signType: 'MD5',
          paySign: 'test_pay_sign'
        });
      } else {
        // 模拟支付宝支付
        await new Promise(resolve => setTimeout(resolve, 2000));
      }

      // 支付成功
      Taro.showToast({
        title: "支付成功",
        icon: "success"
      });

      // 跳转到成功页面
      setTimeout(() => {
        Taro.redirectTo({
          url: `/pages/course-detail/index?id=${orderInfo.courseId}&purchased=true`
        });
      }, 1500);

    } catch (error) {
      console.error("支付失败:", error);
      Taro.showToast({
        title: "支付失败，请重试",
        icon: "none"
      });
    } finally {
      setProcessing(false);
    }
  };

  const handleCancel = () => {
    Taro.showModal({
      title: "取消支付",
      content: "确定要取消支付吗？",
      success: (res) => {
        if (res.confirm) {
          Taro.navigateBack();
        }
      }
    });
  };

  const formatPrice = (price: number) => {
    return `¥${price}`;
  };

  const getDiscountPercent = () => {
    if (!orderInfo) return 0;
    return Math.round((orderInfo.discountAmount / orderInfo.originalPrice) * 100);
  };

  if (loading) {
    return (
      <View className="payment-page">
        <View className="loading">
          <Text>订单生成中...</Text>
        </View>
      </View>
    );
  }

  if (!orderInfo) {
    return (
      <View className="payment-page">
        <View className="error">
          <Text>订单信息异常</Text>
          <Button onClick={() => Taro.navigateBack()}>返回</Button>
        </View>
      </View>
    );
  }

  return (
    <View className="payment-page">
      {/* 订单信息 */}
      <View className="order-section">
        <Text className="section-title">订单信息</Text>
        <View className="order-card">
          <Image
            className="course-cover"
            src={orderInfo.courseCover}
            mode="aspectFill"
          />
          <View className="course-info">
            <Text className="course-title">{orderInfo.courseTitle}</Text>
            <View className="price-info">
              <Text className="current-price">{formatPrice(orderInfo.actualPrice)}</Text>
              <Text className="original-price">{formatPrice(orderInfo.originalPrice)}</Text>
              <View className="discount-badge">
                <Text>省{getDiscountPercent()}%</Text>
              </View>
            </View>
          </View>
        </View>

        <View className="order-details">
          <View className="detail-item">
            <Text className="label">订单号</Text>
            <Text className="value">{orderInfo.id}</Text>
          </View>
          <View className="detail-item">
            <Text className="label">原价</Text>
            <Text className="value">{formatPrice(orderInfo.originalPrice)}</Text>
          </View>
          <View className="detail-item">
            <Text className="label">优惠</Text>
            <Text className="value discount">-{formatPrice(orderInfo.discountAmount)}</Text>
          </View>
          <View className="detail-item total">
            <Text className="label">实付金额</Text>
            <Text className="value">{formatPrice(orderInfo.actualPrice)}</Text>
          </View>
        </View>
      </View>

      {/* 支付方式 */}
      <View className="payment-section">
        <Text className="section-title">支付方式</Text>
        <View className="payment-methods">
          <View 
            className={`payment-method ${paymentMethod === 'wechat' ? 'active' : ''}`}
            onClick={() => handlePaymentMethodChange('wechat')}
          >
            <View className="method-info">
              <Image className="method-icon" src={require("../../images/send.png")} />
              <Text className="method-name">微信支付</Text>
            </View>
            <View className={`radio ${paymentMethod === 'wechat' ? 'checked' : ''}`} />
          </View>
          
          <View 
            className={`payment-method ${paymentMethod === 'alipay' ? 'active' : ''}`}
            onClick={() => handlePaymentMethodChange('alipay')}
          >
            <View className="method-info">
              <Image className="method-icon" src={require("../../images/love.png")} />
              <Text className="method-name">支付宝支付</Text>
            </View>
            <View className={`radio ${paymentMethod === 'alipay' ? 'checked' : ''}`} />
          </View>
        </View>
      </View>

      {/* 优惠信息 */}
      <View className="discount-section">
        <View className="discount-info">
          <Text className="discount-title">🎉 限时特惠</Text>
          <Text className="discount-desc">
            立减{formatPrice(orderInfo.discountAmount)}，错过再等一年！
          </Text>
        </View>
      </View>

      {/* 支付协议 */}
      <View className="agreement-section">
        <Text className="agreement-text">
          点击"立即支付"即表示您同意
          <Text className="link">《用户服务协议》</Text>
          和
          <Text className="link">《隐私政策》</Text>
        </Text>
      </View>

      {/* 底部支付栏 */}
      <View className="payment-bar">
        <View className="price-summary">
          <Text className="total-label">实付金额</Text>
          <Text className="total-price">{formatPrice(orderInfo.actualPrice)}</Text>
        </View>
        <View className="payment-actions">
          <Button 
            className="cancel-btn" 
            onClick={handleCancel}
            disabled={processing}
          >
            取消
          </Button>
          <Button 
            className="pay-btn" 
            onClick={handlePay}
            loading={processing}
            disabled={processing}
          >
            {processing ? "支付中..." : "立即支付"}
          </Button>
        </View>
      </View>
    </View>
  );
} 