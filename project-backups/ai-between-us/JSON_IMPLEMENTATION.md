# JSON 结构实现方案

以下是如何使用写死的 JSON 结构来实现 AI 聊天应用的各种功能：

## 1. 用户数据管理

```json
{
  "next_id": 6,
  "data": [
    {
      "id": 1,
      "phone": "example-phone-number",
      "password": "hashed_password",
      "binding_code": "D9E9FA",
      "partner_id": null,
      "unbind_at": null,
      "created_at": "2026-01-17T16:40:06.787888"
    },
    {
      "id": 2,
      "phone": "example-phone-number",
      "password": "hashed_password123",
      "binding_code": "829D5E",
      "partner_id": 3,
      "unbind_at": null,
      "created_at": "2026-01-17T16:44:48.384240"
    }
  ]
}
```

## 2. 伴侣关系管理

```json
{
  "next_id": 3,
  "data": [
    {
      "id": 1,
      "user1_id": 2,
      "user2_id": 3,
      "room_id": "lounge_room_123",
      "is_active": true,
      "created_at": "2026-01-17T16:45:00.000000"
    },
    {
      "id": 2,
      "user1_id": 4,
      "user2_id": 5,
      "room_id": "lounge_room_456",
      "is_active": true,
      "created_at": "2026-01-17T16:46:00.000000"
    }
  ]
}
```

## 3. 个人教练聊天记录

```json
{
  "next_id": 10,
  "data": [
    {
      "id": 1,
      "user_id": 2,
      "role": "user",
      "content": "你好，我最近压力很大，应该怎么办？",
      "reasoning_content": null,
      "created_at": "2026-01-17T17:05:12.035401"
    },
    {
      "id": 2,
      "user_id": 2,
      "role": "assistant",
      "content": "你好呀～最近压力像块沉甸甸的石头压在心里，一定特别累吧？",
      "reasoning_content": "用户表示压力大，需要共情并引导进一步倾诉...",
      "created_at": "2026-01-17T17:05:23.024847"
    }
  ]
}
```

## 4. 情感客厅聊天记录

```json
{
  "next_id": 20,
  "data": [
    {
      "id": 1,
      "room_id": "lounge_room_123",
      "user_id": 2,
      "role": "user",
      "content": "今天天气真好，心情也跟着变好了",
      "created_at": "2026-01-17T18:00:00.000000"
    },
    {
      "id": 2,
      "room_id": "lounge_room_123",
      "user_id": 3,
      "role": "user",
      "content": "是的呢，我们一起出去走走吧",
      "created_at": "2026-01-17T18:01:00.000000"
    },
    {
      "id": 3,
      "room_id": "lounge_room_123",
      "user_id": null,
      "role": "assistant",
      "content": "看到你们这么开心，我也感到很温暖呢～",
      "created_at": "2026-01-17T18:02:00.000000"
    }
  ]
}
```

## 5. 系统配置

```json
{
  "coze": {
    "api_url": "https://api.coze.cn/v3/chat",
    "api_key": "your_api_key_here",
    "bot_ids": {
      "coach": "75957503example-phone-number",
      "lounge": "7596example-phone-number8699"
    }
  },
  "server": {
    "host": "0.0.0.0",
    "port": 7860,
    "debug": true
  },
  "socketio": {
    "async_mode": "threading",
    "cors_allowed_origins": "*"
  }
}
```

## 6. 实现思路

1. **数据存储**：使用多个 JSON 文件分别存储不同类型的数据（用户、关系、聊天记录等）

2. **数据访问层**：创建一个数据访问模块，提供统一的接口来读取和写入 JSON 文件：
   ```python
   def load_json(file_path):
       with open(file_path, 'r', encoding='utf-8') as f:
           return json.load(f)

   def save_json(file_path, data):
       with open(file_path, 'w', encoding='utf-8') as f:
           json.dump(data, f, ensure_ascii=False, indent=2)
   ```

3. **业务逻辑层**：基于数据访问层实现各种业务功能：
   - 用户认证：从 JSON 文件中查找匹配的用户信息
   - 伴侣绑定：更新用户和关系的 JSON 数据
   - 聊天记录：在 JSON 文件中添加新的聊天消息

4. **配置管理**：使用一个单独的 JSON 文件存储应用配置，在启动时加载

## 7. 优缺点

**优点**：
- 简单易用，不需要额外的数据库服务
- 开发效率高，适合快速原型开发
- 数据结构清晰，易于理解和调试
- 便于版本控制（JSON 文件可以直接提交到代码仓库）

**缺点**：
- 性能有限，不适合高并发场景
- 数据一致性需要手动维护
- 缺乏高级查询功能，需要自己实现过滤和排序
- 不支持事务，可能会出现数据损坏

## 8. 适用场景

- 开发阶段的原型验证
- 小型应用或内部工具
- 演示或教学目的
- 数据量小且访问频率低的场景

这种写死的 JSON 结构实现方式在当前的项目中已经部分采用，比如 `data/users.json`、`data/coach_chats.json` 等文件。可以继续扩展和完善这种实现方式，或者在未来需要时迁移到数据库系统。