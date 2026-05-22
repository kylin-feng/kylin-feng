# 不使用 Socket 实现聊天室功能

以下是如何使用 HTTP 轮询和 JSON 存储来实现聊天室功能，完全不依赖 WebSocket：

## 1. 实现思路

- **数据存储**：使用 JSON 文件存储聊天消息
- **发送消息**：通过 HTTP POST 请求将消息保存到 JSON 文件
- **获取消息**：通过 HTTP GET 请求定期轮询获取最新消息
- **客户端轮询**：前端定期发送请求检查新消息

## 2. JSON 数据结构

### 2.1 聊天室消息存储 (`data/lounge_chat.json`)

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

## 3. 后端实现

### 3.1 数据访问层

```python
# data_manager.py
import json
import os
from datetime import datetime

def load_json(file_path):
    """加载 JSON 文件"""
    if not os.path.exists(file_path):
        return {
            "next_id": 1,
            "data": []
        }
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(file_path, data):
    """保存数据到 JSON 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_message(file_path, room_id, user_id, role, content):
    """添加新消息到聊天室"""
    data = load_json(file_path)
    
    new_message = {
        "id": data["next_id"],
        "room_id": room_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "created_at": datetime.now().isoformat()
    }
    
    data["data"].append(new_message)
    data["next_id"] += 1
    
    save_json(file_path, data)
    return new_message

def get_messages(file_path, room_id, since=None):
    """获取聊天室消息，支持按时间过滤"""
    data = load_json(file_path)
    
    # 过滤指定房间的消息
    room_messages = [msg for msg in data["data"] if msg["room_id"] == room_id]
    
    # 如果指定了时间，只返回该时间之后的消息
    if since:
        room_messages = [msg for msg in room_messages if msg["created_at"] > since]
    
    # 按时间排序
    room_messages.sort(key=lambda x: x["created_at"])
    
    return room_messages
```

### 3.2 API 接口层

```python
# app.py
from flask import Flask, request, jsonify, session
from data_manager import add_message, get_messages
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# 聊天室文件路径
LOUNGE_CHAT_FILE = "data/lounge_chat.json"

@app.route('/api/lounge/send', methods=['POST'])
def send_lounge_message():
    """发送消息到情感客厅"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    data = request.get_json()
    room_id = data.get('room_id')
    content = data.get('content')
    
    if not room_id or not content:
        return jsonify({'success': False, 'message': '参数错误'}), 400
    
    # 添加消息到 JSON 文件
    new_message = add_message(LOUNGE_CHAT_FILE, room_id, user_id, 'user', content)
    
    # 可以在这里添加 AI 回复的逻辑
    
    return jsonify({
        'success': True,
        'message': new_message
    })

@app.route('/api/lounge/messages', methods=['GET'])
def get_lounge_messages():
    """获取情感客厅消息"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    room_id = request.args.get('room_id')
    since = request.args.get('since')  # 用于轮询的时间戳
    
    if not room_id:
        return jsonify({'success': False, 'message': '参数错误'}), 400
    
    # 从 JSON 文件获取消息
    messages = get_messages(LOUNGE_CHAT_FILE, room_id, since)
    
    return jsonify({
        'success': True,
        'messages': messages
    })

@app.route('/api/lounge/room')
def get_lounge_room():
    """获取用户的情感客厅房间"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    # 这里可以根据用户 ID 获取对应的房间 ID
    # 示例：简单返回一个固定的房间 ID
    return jsonify({
        'success': True,
        'room_id': 'lounge_room_123'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)
```

## 4. 前端实现

### 4.1 HTML 结构

```html
<!-- templates/lounge.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>情感客厅</title>
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div id="input-area">
            <input type="text" id="message-input" placeholder="输入消息...">
            <button id="send-button">发送</button>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="/static/js/lounge.js"></script>
</body>
</html>
```

### 4.2 JavaScript 实现（轮询方式）

```javascript
// static/js/lounge.js
let roomId = null;
let lastMessageTime = null;
let pollingInterval = null;

// 初始化聊天室
function initChat() {
    // 获取房间 ID
    $.get('/api/lounge/room', function(response) {
        if (response.success) {
            roomId = response.room_id;
            console.log('进入房间:', roomId);
            
            // 获取历史消息
            loadMessages();
            
            // 开始轮询新消息
            startPolling();
        } else {
            alert('获取房间失败: ' + response.message);
        }
    });
}

// 加载消息
function loadMessages() {
    let url = `/api/lounge/messages?room_id=${roomId}`;
    if (lastMessageTime) {
        url += `&since=${lastMessageTime}`;
    }
    
    $.get(url, function(response) {
        if (response.success) {
            displayMessages(response.messages);
            
            // 更新最后一条消息的时间
            if (response.messages.length > 0) {
                lastMessageTime = response.messages[response.messages.length - 1].created_at;
            }
        }
    });
}

// 显示消息
function displayMessages(messages) {
    const messagesContainer = $('#messages');
    
    messages.forEach(function(message) {
        const messageDiv = $('<div>').addClass('message');
        const role = message.role === 'user' ? '我' : 'AI';
        
        messageDiv.html(`
            <div class="message-header">
                <span class="role">${role}</span>
                <span class="time">${formatTime(message.created_at)}</span>
            </div>
            <div class="message-content">${message.content}</div>
        `);
        
        messagesContainer.append(messageDiv);
    });
    
    // 滚动到底部
    messagesContainer.scrollTop(messagesContainer[0].scrollHeight);
}

// 发送消息
function sendMessage() {
    const content = $('#message-input').val().trim();
    if (!content || !roomId) return;
    
    $.ajax({
        url: '/api/lounge/send',
        type: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            room_id: roomId,
            content: content
        }),
        success: function(response) {
            if (response.success) {
                $('#message-input').val('');
                // 直接显示新消息，不需要等下一次轮询
                displayMessages([response.message]);
            }
        }
    });
}

// 开始轮询
function startPolling() {
    // 每 2 秒检查一次新消息
    pollingInterval = setInterval(loadMessages, 2000);
}

// 格式化时间
function formatTime(timeString) {
    const date = new Date(timeString);
    return date.toLocaleTimeString();
}

// 页面加载完成后初始化
$(document).ready(function() {
    initChat();
    
    // 绑定发送按钮
    $('#send-button').click(sendMessage);
    
    // 回车发送
    $('#message-input').keypress(function(e) {
        if (e.which === 13) {
            sendMessage();
        }
    });
});
```

## 5. 优化建议

1. **长轮询优化**：将普通轮询改为长轮询，减少请求次数：
   - 客户端发送请求后，服务器保持连接直到有新消息
   - 有新消息时立即返回，客户端处理后立即发起下一次请求

2. **消息批量处理**：
   - 批量发送和接收消息，减少 HTTP 请求次数
   - 可以在前端缓存多条消息，定时批量发送

3. **本地存储**：
   - 使用 localStorage 缓存历史消息
   - 减少重复请求，提高加载速度

4. **消息分页**：
   - 实现分页加载历史消息
   - 避免一次性加载过多消息导致性能问题

5. **连接状态管理**：
   - 检测网络连接状态，网络断开时暂停轮询
   - 网络恢复时自动重新连接和同步消息

## 6. 优缺点分析

### 优点
- **实现简单**：不需要复杂的 WebSocket 配置和维护
- **兼容性好**：几乎所有浏览器都支持 HTTP 请求
- **部署方便**：不需要特殊的服务器配置
- **调试容易**：可以通过浏览器开发者工具直接查看请求和响应

### 缺点
- **实时性差**：轮询间隔决定了消息延迟
- **服务器压力大**：大量客户端同时轮询会产生较多请求
- **资源浪费**：轮询请求可能包含大量空响应
- **用户体验**：消息延迟可能影响用户体验

## 7. 适用场景

- **小型聊天室**：用户数量少，对实时性要求不高
- **原型开发**：快速实现聊天功能进行验证
- **网络环境受限**：WebSocket 连接不稳定或被防火墙阻止
- **资源有限**：服务器资源有限，无法处理大量并发 WebSocket 连接

这种基于 HTTP 轮询和 JSON 存储的实现方式虽然不如 WebSocket 高效，但在特定场景下是一种简单有效的解决方案，完全避免了使用 Socket 技术。