# AI-Between-Us Docker 部署说明

## 环境要求
- Ubuntu 18.04 或更高版本
- Docker 20.10 或更高版本
- Docker Compose 2.0 或更高版本

## 安装 Docker 和 Docker Compose

### 在 Ubuntu 上安装 Docker
```bash
# 更新软件包列表
sudo apt update

# 安装依赖包
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common

# 添加 Docker GPG 密钥
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# 添加 Docker APT 仓库
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 更新软件包列表
sudo apt update

# 安装 Docker
sudo apt install -y docker-ce docker-ce-cli containerd.io

# 启动 Docker 服务
sudo systemctl start docker

# 设置 Docker 开机自启
sudo systemctl enable docker

# 添加当前用户到 docker 组（避免每次使用 sudo）
sudo usermod -aG docker $USER
```

### 安装 Docker Compose
```bash
# 下载 Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# 赋予执行权限
chmod +x /usr/local/bin/docker-compose

# 验证安装
docker-compose --version
```

## 部署步骤

### 1. 克隆项目
```bash
git clone https://github.com/jueyunai/AI-Between-Us.git
cd AI-Between-Us
```

### 2. 配置环境变量

#### 创建 .env 文件
```bash
cp .env.example .env
```

#### 编辑 .env 文件
使用文本编辑器打开 .env 文件，配置以下内容：

```bash
# Coze API 配置
COZE_API_KEY=你的Coze API密钥
COZE_BOT_ID_COACH=你的疗愈师Bot ID
COZE_BOT_ID_LOUNGE=你的聊天室Bot ID
```

> **注意事项**：
> - 请确保替换所有占位符为实际的值
> - 不要在生产环境中暴露你的API密钥
> - 这个文件会被挂载到Docker容器中

### 3. 构建并运行 Docker 容器

#### 使用 Docker Compose
```bash
docker-compose up -d
```

#### 使用 Docker 命令
```bash
# 构建镜像
docker build -t ai-between-us .

# 运行容器
docker run -d \
  --name ai-between-us \
  -p 7860:7860 \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/data:/app/data \
  -e FLASK_APP=app.py \
  -e FLASK_ENV=production \
  ai-between-us
```

### 4. 访问应用

应用启动后，可以通过以下地址访问：
```
http://localhost:7860
```

## 管理 Docker 容器

### 查看容器状态
```bash
docker-compose ps
```

### 查看容器日志
```bash
docker-compose logs -f
```

### 停止容器
```bash
docker-compose down
```

### 重启容器
```bash
docker-compose restart
```

## 数据持久化

项目使用 JSON 文件存储数据，数据文件位于 `data/` 目录下：
- `users.json` - 用户数据
- `coach_chats.json` - 个人教练聊天记录

这个目录已经通过 Docker 卷挂载到容器中，确保数据在容器重启后不会丢失。

## 故障排除

### 容器无法启动
1. 检查环境变量配置是否正确：
   ```bash
   cat .env
   ```

2. 查看容器日志：
   ```bash
   docker-compose logs
   ```

### API 密钥无效
1. 确保 Coze API 密钥正确
2. 确保 Bot ID 正确
3. 检查网络连接

### 数据丢失
1. 确保数据目录已经正确挂载
2. 检查容器运行状态

## 安全建议

1. **不要将 .env 文件提交到版本控制**：
   - `.env` 文件包含敏感信息，已经被添加到 `.gitignore`

2. **定期备份数据**：
   - 定期备份 `data/` 目录中的文件

3. **使用 HTTPS**：
   - 在生产环境中，建议使用反向代理（如 Nginx）配置 HTTPS

4. **限制容器权限**：
   - 在生产环境中，避免使用 root 用户运行容器

## 更新应用

```bash
# 拉取最新代码
git pull

# 重新构建并运行容器
docker-compose up -d --build
```

---

如果在部署过程中遇到问题，请参考 Docker 和 Docker Compose 的官方文档，或提交 issue 到项目仓库。