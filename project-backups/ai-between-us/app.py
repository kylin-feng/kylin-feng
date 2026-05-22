# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template, session, Response, stream_with_context
from flask_cors import CORS
from storage import User, Relationship, CoachChat, LoungeChat
from datetime import datetime, timedelta
import secrets
import os
import requests
import json
from dotenv import load_dotenv
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

# 加载环境变量

# 优先从 ms_deploy.json 文件读取环境变量
ms_deploy_path = 'ms_deploy.json'
if os.path.exists(ms_deploy_path):
    print(f"[Env Debug] 从 ms_deploy.json 读取环境变量", flush=True)
    with open(ms_deploy_path, 'r') as f:
        deploy_config = json.load(f)
    
    # 设置环境变量
    for env_var in deploy_config.get('environment_variables', []):
        env_name = env_var.get('name')
        env_value = env_var.get('value')
        if env_name and env_value:
            os.environ[env_name] = env_value
            print(f"[Env Debug] 设置环境变量: {env_name} = {'***' if 'KEY' in env_name or 'TOKEN' in env_name else env_value}", flush=True)
else:
    # 如果没有 ms_deploy.json 文件，则从 .env 文件读取
    print(f"[Env Debug] 从 .env 文件读取环境变量", flush=True)
    load_dotenv(override=True)  # 确保覆盖系统环境变量

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)

app.config['JSON_AS_ASCII'] = False  # 支持中文 JSON 响应


CORS(app)

# Coze API 配置
COZE_API_URL = "https://api.coze.cn/v3/chat"
# 从环境变量读取 API 密钥，设置默认值作为备选
DEFAULT_COZE_API_KEY = "your-coze-api-token"
COZE_API_KEY = os.getenv("COZE_API_KEY", DEFAULT_COZE_API_KEY)
COZE_BOT_ID_COACH = os.getenv("COZE_BOT_ID_COACH", "75957503example-phone-number")
COZE_BOT_ID_LOUNGE = os.getenv("COZE_BOT_ID_LOUNGE", "7596example-phone-number8699")

# API 配置检测
print(f"\n{'='*60}")
print("[Config Check] Coze API 配置检测:")
print(f"[Config Check] API URL: {COZE_API_URL}")
print(f"[Config Check] API Key: {'***' if COZE_API_KEY else '未配置'}")
print(f"[Config Check] Coach Bot ID: {COZE_BOT_ID_COACH if COZE_BOT_ID_COACH else '未配置'}")
print(f"[Config Check] Lounge Bot ID: {COZE_BOT_ID_LOUNGE if COZE_BOT_ID_LOUNGE else '未配置'}")

# 检查是否有配置缺失
config_issues = []
if not COZE_API_KEY:
    config_issues.append("COZE_API_KEY 未配置")
if not COZE_BOT_ID_COACH:
    config_issues.append("COZE_BOT_ID_COACH 未配置")
if not COZE_BOT_ID_LOUNGE:
    config_issues.append("COZE_BOT_ID_LOUNGE 未配置")

if config_issues:
    print(f"[Config Check] ⚠️  配置警告: {', '.join(config_issues)}")
else:
    print(f"[Config Check] ✅ 所有配置均已完成")
print(f"{'='*60}\n")

def get_coze_response(message, api_key, bot_id):
    """
    调用 Coze API 获取响应
    :param message: 用户消息
    :param api_key: Coze API 密钥
    :param bot_id: Bot ID
    :return: AI 回复内容字典
    """
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "bot_id": bot_id,
        "user_id": "lounge_user",
        "stream": False,
        "auto_save_history": True,
        "query": message  # 使用简单的 query 参数，而不是 additional_messages
    }
    
    try:
        logging.info(f"调用 Coze API: bot_id={bot_id}, message={message[:100]}...")
        response = requests.post(COZE_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        logging.info(f"Coze API 响应: {json.dumps(result, ensure_ascii=False)}")
        
        if result.get("code") != 0:
            error_msg = f"Coze API 错误: {result.get('msg', '未知错误')} (代码: {result.get('code')})"
            logging.error(error_msg)
            return {"content": error_msg, "reasoning_content": ""}
        
        data = result.get("data", {})
        messages = data.get("messages", [])
        
        # 检查是否有assistant消息
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content.strip():
                    # 返回包含正文和思考过程的字典
                    return {
                        "content": content,
                        "reasoning_content": msg.get("reasoning_content", "")
                    }
                else:
                    logging.warning("找到 assistant 消息但内容为空")
        
        # 尝试直接从data中获取回复
        if "content" in data and data["content"].strip():
            logging.info("从 data.content 中获取回复")
            return {
                "content": data["content"],
                "reasoning_content": data.get("reasoning_content", "")
            }
        
        logging.warning("未找到有效回复")
        return {"content": "AI 未返回有效回复", "reasoning_content": ""}
    except Exception as e:
        logging.error(f"调用 Coze API 失败: {e}")
        return {"content": "AI 调用失败，请稍后重试", "reasoning_content": ""}

def call_coze_api(user_phone, message, bot_id, conversation_history=None):
    """
    调用 Coze API（使用流式响应）
    :param user_phone: 用户手机号（作为 user_id）
    :param message: 用户消息
    :param bot_id: Bot ID
    :param conversation_history: 对话历史（可选）
    :return: AI 回复内容
    """
    # API 配置检测
    if not COZE_API_KEY:
        error_msg = "AI 服务未配置: COZE_API_KEY 缺失"
        print(f"[Coze API] ERROR: {error_msg}", flush=True)
        return error_msg
    
    if not bot_id:
        error_msg = "AI 服务未配置: Bot ID 缺失"
        print(f"[Coze API] ERROR: {error_msg}", flush=True)
        return error_msg

    try:
        import json
        
        # 网络连接检测
        try:
            import socket
            # 测试 Coze API 服务器连接
            socket.create_connection(("api.coze.cn", 443), timeout=5)
            print(f"[Coze API] 网络连接: ✅ 成功连接到 api.coze.cn", flush=True)
        except socket.error as e:
            error_msg = f"网络连接失败: 无法连接到 Coze API 服务器 ({str(e)})"
            print(f"[Coze API] ERROR: {error_msg}", flush=True)
            return error_msg

        headers = {
            'Authorization': f'Bearer {COZE_API_KEY}',
            'Content-Type': 'application/json'
        }

        # 构建消息列表
        messages = []
        if conversation_history:
            for msg in conversation_history:
                msg_type = "question" if msg["role"] == "user" else "answer"
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                    "content_type": "text",
                    "type": msg_type
                })

        # 添加当前消息
        messages.append({
            "role": "user",
            "content": message,
            "content_type": "text",
            "type": "question"
        })

        payload = {
            "bot_id": bot_id,
            "user_id": user_phone,
            "stream": True,  # 使用流式响应
            "auto_save_history": True,
            "additional_messages": messages
        }

        print(f"\n{'='*60}", flush=True)
        print(f"[Coze API] 发送请求到: {COZE_API_URL}", flush=True)
        print(f"[Coze API] Bot ID: {bot_id}", flush=True)
        print(f"[Coze API] User ID: {user_phone}", flush=True)
        print(f"[Coze API] 请求头: Authorization={headers['Authorization'][:20]}...", flush=True)
        
        try:
            response = requests.post(COZE_API_URL, headers=headers, json=payload, timeout=60, stream=True)
            
            # 检查 HTTP 状态码
            print(f"[Coze API] 响应状态码: {response.status_code}", flush=True)
            
            if response.status_code != 200:
                error_msg = f"API 请求失败: HTTP {response.status_code} {response.reason}"
                print(f"[Coze API] ERROR: {error_msg}", flush=True)
                return error_msg
            
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_msg = f"API 请求错误: {str(e)}"
            print(f"[Coze API] ERROR: {error_msg}", flush=True)
            return error_msg

        # 检查响应内容类型
        content_type = response.headers.get('Content-Type', '')
        print(f"[Coze API] 响应 Content-Type: {content_type}", flush=True)

        # 处理流式响应
        completed_content = None
        has_stream_data = False
        current_event = None  # 跟踪 SSE event 类型
        
        for line in response.iter_lines():
            if line:
                has_stream_data = True
                try:
                    line_text = line.decode('utf-8')
                    print(f"[Coze Stream] {line_text}", flush=True)

                    # 处理 SSE 格式 - event: 行
                    if line_text.startswith('event:'):
                        current_event = line_text[6:].strip()
                        continue

                    # 处理 SSE 格式 - data: 行
                    if line_text.startswith('data:'):
                        json_str = line_text[5:].strip()
                        if json_str == '[DONE]' or json_str == '"[DONE]"':
                            break
                        
                        if not json_str:
                            continue

                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            print(f"[Coze API] JSON 解析错误: {e}", flush=True)
                            continue
                        
                        if not isinstance(data, dict):
                            continue
                        
                        # 检查 API 返回的错误
                        if data.get('code') and data.get('code') != 0:
                            error_msg = f"Coze API 错误: {data.get('msg', '未知错误')} (代码: {data.get('code')})"
                            print(f"[Coze API] ERROR: {error_msg}", flush=True)
                            return error_msg
                        
                        # 过滤掉所有元数据消息
                        msg_type = data.get('msg_type')
                        if msg_type:
                            continue

                        # 只处理完成事件，使用之前记录的 event 类型
                        if current_event == 'conversation.message.completed':
                            # Coze API 返回的数据格式：role 和 content 直接在 data 中
                            role = data.get('role')
                            msg_type_field = data.get('type')  # answer, follow_up, verbose 等
                            content = data.get('content', '')
                            
                            print(f"[Coze API] 完成事件: role={role}, type={msg_type_field}, content_len={len(content) if content else 0}", flush=True)
                            
                            # 跳过 verbose 类型（内部日志）
                            if msg_type_field == 'verbose':
                                continue
                            
                            if role == 'assistant' and isinstance(content, str) and content:
                                # 优先使用 answer 类型的回复，follow_up 作为备选
                                if msg_type_field == 'answer':
                                    completed_content = content
                                    print(f"[Coze API] 收到 answer 回复，内容长度: {len(content)}", flush=True)
                                elif msg_type_field == 'follow_up' and not completed_content:
                                    # 如果还没有 answer，暂存 follow_up
                                    completed_content = content
                                    print(f"[Coze API] 收到 follow_up 回复，内容长度: {len(content)}", flush=True)

                except UnicodeDecodeError as e:
                    print(f"[Coze API] 解码错误: {e}", flush=True)
                    continue
                except Exception as e:
                    print(f"[Coze API] 处理流式数据异常: {type(e).__name__}: {e}", flush=True)
                    continue
        
        # 如果没有收到流式数据
        if not has_stream_data:
            try:
                result = response.json()
                print(f"[Coze API] 非流式响应: {result}", flush=True)
                
                if isinstance(result, dict):
                    if result.get("code") != 0:
                        error_msg = f"Coze API 错误: {result.get('msg', '未知错误')} (代码: {result.get('code')})"
                        print(f"[Coze API] ERROR: {error_msg}", flush=True)
                        return error_msg
                    
                    data = result.get("data", {})
                    if isinstance(data, dict):
                        messages = data.get("messages", [])
                        if isinstance(messages, list):
                            for msg in messages:
                                if isinstance(msg, dict) and msg.get("role") == "assistant":
                                    content = msg.get("content", "")
                                    if isinstance(content, str) and content:
                                        completed_content = content
                                        break
            except Exception as e:
                print(f"[Coze API] 解析非流式响应失败: {e}", flush=True)
                print(f"[Coze API] 响应内容: {response.text[:200]}...", flush=True)

        # 检查是否获取到内容
        if not completed_content:
            error_msg = "AI 未返回有效回复，可能是网络问题或 API 配置错误"
            print(f"[Coze API] ERROR: {error_msg}", flush=True)
            return error_msg

        # 清理文本：移除可能混入的JSON字符串和重复内容
        final_content = completed_content
        if final_content:
            import re
            # 移除所有JSON格式的字符串（包括嵌套的）
            while True:
                # 移除简单的JSON对象
                new_content = re.sub(r'\{[^\{\}]*"msg_type"[^\{\}]*\}', '', final_content)
                # 移除嵌套的JSON对象（处理转义的情况）
                new_content = re.sub(r'\{[^\{\}]*"data"[^\{\}]*"[^\{\}]*"[^\{\}]*\}', '', new_content)
                if new_content == final_content:
                    break
                final_content = new_content
            
            # 移除多余的空白字符，但保留换行符
            final_content = re.sub(r'[ ]+', ' ', final_content).strip()
            
            # 移除行首和行尾的空格
            final_content = '\n'.join(line.strip() for line in final_content.split('\n'))
            
            # 智能去重：检测并移除重复的句子或段落
            half_len = len(final_content) // 2
            if half_len > 20:  # 至少20个字符才判断重复
                if final_content[:half_len] == final_content[half_len:]:
                    final_content = final_content[:half_len]
                elif len(final_content) > 40:
                    # 尝试找到重复的句子
                    sentences = re.split(r'[。！？\n]', final_content)
                    if len(sentences) > 2:
                        # 检查是否有连续重复的句子
                        cleaned_sentences = []
                        for i, sent in enumerate(sentences):
                            if i == 0 or sent != sentences[i-1]:
                                cleaned_sentences.append(sent)
                        if len(cleaned_sentences) < len(sentences):
                            final_content = '。'.join(cleaned_sentences)
        
        print(f"[Coze API] 完成事件内容长度: {len(completed_content) if completed_content else 0}", flush=True)
        print(f"[Coze API] 清理后内容长度: {len(final_content) if final_content else 0}", flush=True)
        print(f"[Coze API] 最终回复: {final_content[:200] if final_content else 'None'}...", flush=True)
        print(f"{'='*60}\n", flush=True)

        # 保存详细调试信息
        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "api_key_configured": bool(COZE_API_KEY),
            "bot_id": bot_id,
            "user_id": user_phone,
            "request_message": message,
            "completed_content": completed_content,
            "final_content": final_content,
            "content_length": len(final_content) if final_content else 0
        }
        with open('coze_debug.json', 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, ensure_ascii=False, indent=2)

        return final_content

    except requests.exceptions.Timeout:
        error_msg = "AI 响应超时，请检查网络连接"
        print(f"[Coze API] ERROR: {error_msg}", flush=True)
        return error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"AI 调用失败: {str(e)}"
        print(f"[Coze API] ERROR: {error_msg}", flush=True)
        return error_msg
    except Exception as e:
        error_msg = f"AI 处理异常: {type(e).__name__}: {str(e)}"
        print(f"[Coze API] ERROR: {error_msg}", flush=True)
        import traceback
        print(f"[Coze API] 详细错误: {traceback.format_exc()}", flush=True)
        return error_msg

# ==================== 用户认证 API ====================
@app.route('/api/register', methods=['POST'])
def register():
    """用户注册"""
    data = request.json
    phone = data.get('phone')
    password = data.get('password')

    if not phone or not password:
        return jsonify({'success': False, 'message': '手机号和密码不能为空'}), 400

    # 检查用户是否已存在
    existing_users = User.filter(phone=phone)
    if existing_users:
        return jsonify({'success': False, 'message': '该手机号已注册'}), 400

    # 创建新用户
    user = User(phone=phone, password=password)
    user.generate_binding_code()
    user.save()

    return jsonify({
        'success': True,
        'message': '注册成功',
        'user': user.to_dict()
    })


@app.route('/api/login', methods=['POST'])
def login():
    """用户登录"""
    data = request.json
    phone = data.get('phone')
    password = data.get('password')

    users = User.filter(phone=phone, password=password)
    user = users[0] if users else None
    if not user:
        return jsonify({'success': False, 'message': '手机号或密码错误'}), 401

    # 设置 session
    session['user_id'] = user.id

    return jsonify({
        'success': True,
        'message': '登录成功',
        'user': user.to_dict()
    })


@app.route('/api/logout', methods=['POST'])
def logout():
    """用户登出"""
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': '登出成功'})


@app.route('/api/user/info', methods=['GET'])
def get_user_info():
    """获取用户信息"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    user = User.get(user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在'}), 404

    return jsonify({
        'success': True,
        'user': user.to_dict()
    })


@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user_by_id(user_id):
    """根据ID获取用户信息（只返回基本信息）"""
    current_user_id = session.get('user_id')
    if not current_user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    user = User.get(user_id)
    if not user:
        return jsonify({'success': False, 'message': '用户不存在'}), 404

    # 只返回基本信息，保护隐私
    return jsonify({
        'success': True,
        'user': {
            'id': user.id,
            'phone': user.phone  # 用于显示昵称
        }
    })


# ==================== 伴侣绑定 API ====================
@app.route('/api/binding/code', methods=['GET'])
def get_binding_code():
    """获取绑定码"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    user = User.get(user_id)
    if not user.binding_code:
        user.generate_binding_code()
        user.save()

    return jsonify({
        'success': True,
        'binding_code': user.binding_code
    })


@app.route('/api/binding/bind', methods=['POST'])
def bind_partner():
    """使用绑定码绑定伴侣"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    data = request.json
    binding_code = data.get('binding_code')

    user = User.get(user_id)
    partners = User.filter(binding_code=binding_code)
    partner = partners[0] if partners else None

    if not partner:
        return jsonify({'success': False, 'message': '绑定码无效'}), 400

    if partner.id == user.id:
        return jsonify({'success': False, 'message': '不能绑定自己'}), 400

    if user.partner_id or partner.partner_id:
        return jsonify({'success': False, 'message': '您或对方已有伴侣'}), 400

    # 建立绑定关系
    user.partner_id = partner.id
    partner.partner_id = user.id

    # 创建情感客厅房间
    room_id = f"room_{min(user.id, partner.id)}_{max(user.id, partner.id)}"
    relationship = Relationship(
        user1_id=min(user.id, partner.id),
        user2_id=max(user.id, partner.id),
        room_id=room_id
    )
    relationship.save()
    user.save()
    partner.save()

    return jsonify({
        'success': True,
        'message': '绑定成功！',
        'room_id': room_id
    })


@app.route('/api/binding/unbind', methods=['POST'])
def unbind_partner():
    """解绑伴侣"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    user = User.get(user_id)
    if not user.partner_id:
        return jsonify({'success': False, 'message': '您还没有绑定伴侣'}), 400

    partner = User.get(user.partner_id)

    # 设置解绑时间（1个月冷静期）
    unbind_time = datetime.now()
    user.unbind_at = unbind_time
    partner.unbind_at = unbind_time

    # 停用关系
    relationships = Relationship.filter(
        user1_id=min(user.id, partner.id),
        user2_id=max(user.id, partner.id)
    )
    if relationships:
        relationship = relationships[0]
        relationship.is_active = False
        relationship.save()

    user.save()
    partner.save()

    return jsonify({
        'success': True,
        'message': '已发起解绑，1个月冷静期后生效',
        'unbind_at': unbind_time.isoformat()
    })


@app.route('/api/binding/cancel_unbind', methods=['POST'])
def cancel_unbind():
    """撤销解绑"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    user = User.get(user_id)
    if not user.unbind_at:
        return jsonify({'success': False, 'message': '没有待撤销的解绑'}), 400

    # 检查是否在冷静期内
    if isinstance(user.unbind_at, str):
        user.unbind_at = datetime.fromisoformat(user.unbind_at)
    cool_down_end = user.unbind_at + timedelta(days=30)
    if datetime.now() > cool_down_end:
        return jsonify({'success': False, 'message': '冷静期已过，无法撤销'}), 400

    partner = User.get(user.partner_id)
    user.unbind_at = None
    partner.unbind_at = None

    # 恢复关系
    relationships = Relationship.filter(
        user1_id=min(user.id, partner.id),
        user2_id=max(user.id, partner.id)
    )
    if relationships:
        relationship = relationships[0]
        relationship.is_active = True
        relationship.save()

    user.save()
    partner.save()

    return jsonify({
        'success': True,
        'message': '已撤销解绑'
    })


# ==================== 个人教练聊天室 API ====================
@app.route('/api/coach/chat', methods=['POST'])
def coach_chat():
    """个人教练聊天"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    data = request.json
    message = data.get('message')

    if not message:
        return jsonify({'success': False, 'message': '消息不能为空'}), 400

    # 获取用户信息
    user = User.get(user_id)
    user_phone = user.phone

    # 保存用户消息
    user_msg = CoachChat(user_id=user_id, role='user', content=message)
    user_msg.save()

    # 获取历史对话（最近5条，避免消息过长）
    all_history = CoachChat.filter(user_id=user_id)
    all_history.sort(key=lambda x: x.created_at, reverse=True)
    history = all_history[:5]
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in reversed(history)]

    # 调用 Coze API
    ai_reply = call_coze_api(
        user_phone=user_phone,
        message=message,
        bot_id=COZE_BOT_ID_COACH,
        conversation_history=conversation_history[:-1] if conversation_history else None  # 排除当前消息
    )

    # 保存 AI 回复
    ai_msg = CoachChat(user_id=user_id, role='assistant', content=ai_reply)
    ai_msg.save()

    return jsonify({
        'success': True,
        'message': ai_reply
    })


@app.route('/api/coach/history', methods=['GET'])
def get_coach_history():
    """获取个人教练聊天记录"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    history = CoachChat.filter(user_id=user_id)
    history.sort(key=lambda x: x.created_at)

    return jsonify({
        'success': True,
        'messages': [msg.to_dict() for msg in history]
    })


@app.route('/api/coach/chat/stream', methods=['POST'])
def coach_chat_stream():
    """个人教练流式聊天 - 实时推送思考过程和正文"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    data = request.json
    message = data.get('message')

    if not message:
        return jsonify({'success': False, 'message': '消息不能为空'}), 400

    # 获取用户信息
    user = User.get(user_id)
    user_phone = user.phone

    # 保存用户消息
    user_msg = CoachChat(user_id=user_id, role='user', content=message)
    user_msg.save()

    # 获取历史对话（最近5条）
    all_history = CoachChat.filter(user_id=user_id)
    all_history.sort(key=lambda x: x.created_at, reverse=True)
    history = all_history[:5]
    conversation_history = [{"role": msg.role, "content": msg.content} for msg in reversed(history)]

    def generate():
        """流式生成器"""
        if not COZE_API_KEY or not COZE_BOT_ID_COACH:
            yield f"data: {json.dumps({'type': 'error', 'content': 'AI 服务未配置'}, ensure_ascii=False)}\n\n"
            return

        try:
            headers = {
                'Authorization': f'Bearer {COZE_API_KEY}',
                'Content-Type': 'application/json'
            }

            # 构建消息列表
            messages = []
            if conversation_history:
                for msg in conversation_history[:-1]:  # 排除当前消息
                    msg_type = "question" if msg["role"] == "user" else "answer"
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "content_type": "text",
                        "type": msg_type
                    })

            messages.append({
                "role": "user",
                "content": message,
                "content_type": "text",
                "type": "question"
            })

            payload = {
                "bot_id": COZE_BOT_ID_COACH,
                "user_id": user_phone,
                "stream": True,
                "auto_save_history": True,
                "additional_messages": messages
            }

            response = requests.post(COZE_API_URL, headers=headers, json=payload, timeout=60, stream=True)
            response.raise_for_status()

            current_event = None
            final_content = ""
            reasoning_content = ""

            for line in response.iter_lines():
                if line:
                    try:
                        line_text = line.decode('utf-8')

                        # 处理 event: 行
                        if line_text.startswith('event:'):
                            current_event = line_text[6:].strip()
                            continue

                        # 处理 data: 行
                        if line_text.startswith('data:'):
                            json_str = line_text[5:].strip()
                            if json_str == '[DONE]' or json_str == '"[DONE]"':
                                break

                            if not json_str:
                                continue

                            try:
                                data = json.loads(json_str)
                            except json.JSONDecodeError:
                                continue

                            if not isinstance(data, dict):
                                continue

                            # 跳过元数据消息
                            if data.get('msg_type'):
                                continue

                            role = data.get('role')
                            msg_type_field = data.get('type')

                            # 处理流式内容 (delta 事件)
                            if current_event == 'conversation.message.delta' and role == 'assistant' and msg_type_field == 'answer':
                                # 思考过程 (reasoning_content)
                                reasoning = data.get('reasoning_content', '')
                                if reasoning:
                                    reasoning_content += reasoning
                                    yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning}, ensure_ascii=False)}\n\n"

                                # 正文内容 (content)
                                content = data.get('content', '')
                                if content:
                                    final_content += content
                                    yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"

                            # 处理完成事件
                            elif current_event == 'conversation.message.completed' and role == 'assistant':
                                if msg_type_field == 'answer':
                                    # 思考完成信号
                                    yield f"data: {json.dumps({'type': 'reasoning_done'}, ensure_ascii=False)}\n\n"
                                elif msg_type_field == 'follow_up':
                                    # 跳过 follow_up
                                    pass

                    except Exception as e:
                        print(f"[Stream Error] {e}", flush=True)
                        continue

            # 保存 AI 回复到存储（包含思考过程）
            if final_content:
                ai_msg = CoachChat(
                    user_id=user_id, 
                    role='assistant', 
                    content=final_content,
                    reasoning_content=reasoning_content if reasoning_content else None
                )
                ai_msg.save()

            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done', 'final_content': final_content, 'reasoning_content': reasoning_content}, ensure_ascii=False)}\n\n"

        except Exception as e:
            print(f"[Stream API Error] {e}", flush=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


# ==================== 情感客厅聊天室 API ====================
@app.route('/api/lounge/room', methods=['GET'])
def get_lounge_room():
    """获取情感客厅房间信息"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    user = User.get(user_id)
    if not user.partner_id:
        return jsonify({'success': False, 'message': '您还没有绑定伴侣'}), 400

    # 查找用户相关的活跃关系（user1_id 或 user2_id 等于当前用户）
    all_relationships = Relationship.all()
    relationships = [
        r for r in all_relationships 
        if (r.user1_id == user.id or r.user2_id == user.id) and r.is_active
    ]
    relationship = relationships[0] if relationships else None

    if not relationship:
        return jsonify({'success': False, 'message': '未找到有效的关系'}), 404

    return jsonify({
        'success': True,
        'room_id': relationship.room_id
    })


@app.route('/api/lounge/history', methods=['GET'])
def get_lounge_history():
    """获取情感客厅聊天记录"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401

    user = User.get(user_id)
    # 查找用户相关的关系
    all_relationships = Relationship.all()
    relationships = [
        r for r in all_relationships 
        if r.user1_id == user.id or r.user2_id == user.id
    ]
    relationship = relationships[0] if relationships else None

    if not relationship:
        return jsonify({'success': False, 'message': '未找到房间'}), 404

    history = LoungeChat.filter(room_id=relationship.room_id)
    history.sort(key=lambda x: x.created_at)

    return jsonify({
        'success': True,
        'messages': [msg.to_dict() for msg in history]
    })


# ==================== HTTP API 通信 ====================
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
    
    # 保存消息
    msg = LoungeChat(room_id=room_id, user_id=user_id, role='user', content=content)
    msg.save()
    
    # 检查是否需要触发 AI
    is_calling_ai = '@AI' in content or '@ai' in content or '@教练' in content
    
    return jsonify({
        'success': True,
        'message': {**msg.to_dict(), 'trigger_ai': is_calling_ai}
    })

@app.route('/api/lounge/messages', methods=['GET'])
def get_lounge_messages():
    """获取情感客厅消息"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    room_id = request.args.get('room_id')
    since = request.args.get('since')  # 用于轮询的时间戳
    limit = request.args.get('limit', 50, type=int)
    
    if not room_id:
        return jsonify({'success': False, 'message': '参数错误'}), 400
    
    # 获取消息
    messages = LoungeChat.filter(room_id=room_id)
    messages.sort(key=lambda x: x.created_at)
    
    # 如果指定了时间，只返回该时间之后的消息
    if since:
        try:
            # 将字符串转换为datetime对象
            since_datetime = datetime.fromisoformat(since)
            messages = [msg for msg in messages if msg.created_at > since_datetime]
        except ValueError:
            # 如果时间格式不正确，忽略since参数
            pass
    
    # 限制返回数量
    messages = messages[-limit:]
    
    return jsonify({
        'success': True,
        'messages': [msg.to_dict() for msg in messages]
    })

@app.route('/api/lounge/call_ai', methods=['POST'])
def call_lounge_ai():
    """召唤 AI 助手"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': '未登录'}), 401
    
    data = request.get_json()
    room_id = data.get('room_id')
    
    if not room_id:
        return jsonify({'success': False, 'message': '参数错误'}), 400
    
    # 获取最近的对话记录（最近10条）
    all_history = LoungeChat.filter(room_id=room_id)
    all_history.sort(key=lambda x: x.created_at, reverse=True)
    history = all_history[:10]
    
    # 构建对话历史（保留所有对话，包括AI回复）
    latest_message = ""
    logging.info(f"当前对话历史数量: {len(history)}")
    for msg in reversed(history):
        role = "用户" if msg.role == "user" else "AI"
        latest_message += f"{role}: {msg.content}\n"
    
    logging.info(f"构建的对话历史: {latest_message.strip()}")
    logging.info(f"使用的 Bot ID: {COZE_BOT_ID_LOUNGE}")
    
    # 如果没有对话记录，返回提示
    if not latest_message.strip():
        ai_reply = "暂时没有对话内容可供分析哦～"
        ai_msg = LoungeChat(room_id=room_id, user_id=None, role='assistant', content=ai_reply)
        ai_msg.save()
        return jsonify({
            'success': True,
            'message': ai_msg.to_dict()
        })
    
    try:
        # 使用流式传输调用 Coze API 获取 AI 回复
        # 获取用户手机号作为 user_id
        user = User.get(user_id)
        user_phone = user.phone if user else "lounge_user"
        
        # 构建对话历史（转换为 call_coze_api 所需的格式）
        conversation_history = []
        for msg in history:
            conversation_history.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # 调用流式 API
        ai_response = call_coze_api(user_phone, latest_message, COZE_BOT_ID_LOUNGE, conversation_history)
        
        # 处理 AI 响应
        ai_msg = LoungeChat(
            room_id=room_id, 
            user_id=None, 
            role='assistant', 
            content=ai_response,
            reasoning_content=""  # 流式 API 不返回 reasoning_content
        )
        
        # 保存 AI 回复
        ai_msg.save()
        
        return jsonify({
            'success': True,
            'message': ai_msg.to_dict()
        })
    except Exception as e:
        logging.error(f"AI 调用失败: {e}")
        return jsonify({
            'success': False,
            'message': 'AI 调用失败，请稍后重试'
        })


# ==================== 前端路由 ====================
@app.route('/')
def index():
    """首页"""
    return render_template('login.html')


@app.route('/home')
def home():
    """主页"""
    return render_template('home.html')

@app.route('/profile')
def profile():
    """个人中心"""
    return render_template('profile.html')


@app.route('/coach')
def coach():
    """个人教练"""
    return render_template('coach.html')


@app.route('/lounge')
def lounge():
    """情感客厅"""
    return render_template('lounge.html')


if __name__ == '__main__':
    import os
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug_mode, host='0.0.0.0', port=7860, use_reloader=debug_mode)
