// 全局错误处理中间件
export const errorHandler = (err, req, res, next) => {
  console.error('Error:', {
    message: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method,
    timestamp: new Date().toISOString()
  });

  // 默认错误状态码
  let statusCode = err.statusCode || 500;
  let message = err.message || '服务器内部错误';

  // 特定错误类型处理
  if (err.name === 'ValidationError') {
    statusCode = 400;
    message = '请求参数验证失败';
  } else if (err.name === 'UnauthorizedError') {
    statusCode = 401;
    message = '未授权访问';
  } else if (err.name === 'CastError') {
    statusCode = 400;
    message = '无效的ID格式';
  }

  // 生产环境中不暴露具体错误信息
  if (process.env.NODE_ENV === 'production' && statusCode === 500) {
    message = '服务器内部错误';
  }

  res.status(statusCode).json({
    success: false,
    data: null,
    message,
    error: process.env.NODE_ENV === 'development' ? err.stack : undefined,
    timestamp: new Date().toISOString()
  });
};

// 404处理中间件
export const notFoundHandler = (req, res) => {
  res.status(404).json({
    success: false,
    data: null,
    message: `请求的路径 ${req.originalUrl} 不存在`,
    timestamp: new Date().toISOString()
  });
};

export default { errorHandler, notFoundHandler };