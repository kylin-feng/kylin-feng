# 使用官方Python基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（不需要gcc，因为使用JSON文件存储）
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data

# 暴露应用端口
EXPOSE 7860

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 运行应用
CMD ["python", "app.py"]
