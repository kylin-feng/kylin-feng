import streamlit as st
import requests
import json

# 页面配置
st.set_page_config(
    page_title="职得 - AI简历优化助手",
    page_icon="📝",
    layout="wide"
)

# 标题区域
st.title("📝 职得 - AI简历优化助手")
st.markdown("*根据JD智能优化你的简历，提高投递成功率*")
st.divider()

# 格式选择
st.subheader("📋 简历格式选择")
col_format, col_model = st.columns(2)

with col_format:
    resume_format = st.selectbox(
        "选择你的简历格式",
        ["LaTeX", "Markdown", "Word/纯文本"],
        help="选择简历的原始格式，AI将保持相同格式进行优化"
    )

with col_model:
    ai_model = st.selectbox(
        "选择AI模型",
        [
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-V2.5", 
            "deepseek-ai/deepseek-chat",
            "Qwen/Qwen2.5-72B-Instruct"
        ],
        help="选择用于优化简历的AI模型"
    )

# 输入区域 - 两列布局
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 岗位描述 (JD)")
    jd_text = st.text_area(
        "请粘贴完整的岗位描述",
        placeholder="请粘贴目标岗位的完整JD，包括岗位职责、任职要求等...",
        height=500,  # 增加高度支持长内容
        help="包含岗位职责、技能要求、工作内容等信息，支持长文本输入"
    )
    if jd_text:
        st.caption(f"📊 JD字符数: {len(jd_text)} | 预估tokens: ~{len(jd_text)//4}")

with col2:
    st.subheader("📄 你的简历")
    if resume_format == "LaTeX":
        placeholder = """例如：
\\documentclass[11pt,a4paper]{article}
\\usepackage{latexsym}
\\name{张三}
\\begin{document}
\\section{教育背景}
\\textbf{清华大学} \\hfill 2020.09 - 2024.06 \\\\
计算机科学与技术学士 \\hfill GPA: 3.8/4.0
\\end{document}"""
    elif resume_format == "Markdown":
        placeholder = """例如：
# 张三
## 联系方式
- 邮箱：zhang@example.com
- 电话：138-0000-0000

## 教育背景
**清华大学** | 2020.09 - 2024.06
计算机科学与技术学士 | GPA: 3.8/4.0"""
    else:
        placeholder = """例如：
张三
联系方式：zhang@example.com | 138-0000-0000

教育背景
清华大学                           2020.09 - 2024.06
计算机科学与技术学士               GPA: 3.8/4.0"""
    
    resume_text = st.text_area(
        "请粘贴你的简历内容",
        placeholder=placeholder,
        height=500,  # 增加高度支持长LaTeX简历
        help=f"请粘贴{resume_format}格式的完整简历内容，支持长文本和复杂格式"
    )
    if resume_text:
        st.caption(f"📊 简历字符数: {len(resume_text)} | 预估tokens: ~{len(resume_text)//4}")
        total_chars = len(jd_text) + len(resume_text)
        if total_chars > 50000:
            st.warning(f"⚠️ 总内容较长({total_chars}字符)，处理时间可能较久")
        elif total_chars > 100000:
            st.error("❌ 内容过长，建议简化后重试")

# AI优化函数
def call_siliconflow_api(prompt, model="deepseek-ai/DeepSeek-V3", max_retries=3):
    """调用硅基流动API，带重试机制"""
    
    # API配置：真实密钥只从本地 .streamlit/secrets.toml 读取，不写入代码仓库。
    api_key = st.secrets.get("SILICONFLOW_API_KEY", "")
    if not api_key:
        st.error("请先在 .streamlit/secrets.toml 中配置 SILICONFLOW_API_KEY")
        return None
    base_url = "https://api.siliconflow.cn/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 使用用户选择的模型，拉满tokens以支持长输入输出
    data = {
        "model": model,
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 32768  # 拉满tokens，支持160K上下文和长LaTeX简历
    }
    
    import time
    
    for attempt in range(max_retries):
        try:
            # 显示当前尝试次数
            if attempt > 0:
                st.info(f"第 {attempt + 1} 次尝试连接API...")
            
            # 使用更长的超时时间，并启用连接池
            response = requests.post(
                base_url, 
                headers=headers, 
                json=data, 
                timeout=(10, 60),  # (连接超时, 读取超时)
                stream=False
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 递增等待时间
                st.warning(f"请求超时，{wait_time}秒后进行第 {attempt + 2} 次尝试...")
                time.sleep(wait_time)
            else:
                raise Exception("API请求超时，请检查网络连接或稍后重试")
                
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                st.warning(f"网络连接失败，{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                raise Exception("无法连接到API服务器，请检查网络连接")
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # 频率限制
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10
                    st.warning(f"API调用频率限制，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise Exception("API调用频率过高，请稍后重试")
            else:
                raise Exception(f"API调用失败: HTTP {e.response.status_code}")
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5
                st.warning(f"请求异常，{wait_time}秒后重试: {str(e)}")
                time.sleep(wait_time)
            else:
                raise Exception(f"API调用失败: {str(e)}")
                
        except KeyError as e:
            raise Exception(f"API响应格式错误: {str(e)}")
    
    raise Exception("所有重试均失败")

def optimize_resume_with_ai(jd_text, resume_text, resume_format, model="deepseek-ai/DeepSeek-V3"):
    """调用硅基流动AI优化简历"""
    
    # 根据格式定制提示词
    format_instructions = {
        "LaTeX": "保持LaTeX格式，包括所有命令、环境和特殊字符（如\\\\、\\textbf{}、\\section{}等）",
        "Markdown": "保持Markdown格式，包括标题层级（#、##）、列表（-、*）、粗体（**）等语法",
        "Word/纯文本": "保持纯文本格式，使用合适的缩进和空行来组织内容结构"
    }
    
    # 针对LaTeX格式，对特殊字符进行适当处理
    def escape_for_json(text):
        """对文本进行JSON转义，特别处理LaTeX特殊字符"""
        import json
        return json.dumps(text, ensure_ascii=False)[1:-1]  # 移除外层引号
    
    prompt = f"""你是一位专业的HR和简历优化专家。请根据以下岗位描述(JD)来优化用户的简历。

任务要求：
1. 仔细分析JD中的关键技能要求、工作职责和任职资格
2. 识别简历中与JD匹配和不匹配的地方
3. 优化简历内容，提高与目标岗位的匹配度
4. 保持简历的真实性，不要编造不存在的经历
5. **重要：严格保持原有的{resume_format}格式！{format_instructions[resume_format]}**
6. **LaTeX特殊处理：如果是LaTeX格式，请特别注意保持所有反斜杠、花括号、特殊命令完整**

岗位描述(JD)：
{jd_text}

用户简历（{resume_format}格式）：
{resume_text}

**重要说明：**
- 如果是LaTeX格式，请完整保持所有LaTeX命令和特殊字符，包括但不限于：反斜杠、花括号、百分号注释等
- 输出的JSON中，LaTeX特殊字符会自动转义，这是正常的
- 确保返回的简历内容可以直接在LaTeX编译器中使用

**请严格按照以下JSON格式返回结果，不要添加任何其他文字：**

{{"match_analysis": "分析简历与JD的匹配程度，指出优势和不足", "suggestions": ["建议1：具体描述需要改进的地方", "建议2：具体描述需要突出的技能", "建议3：具体描述需要调整的表达"], "optimized_resume": "在这里放入完整的优化后简历内容，严格保持{resume_format}格式", "key_improvements": ["关键改进1：说明具体改了什么", "关键改进2：说明为什么这样改"]}}"""

    try:
        # 硅基流动API调用，传入用户选择的模型
        response = call_siliconflow_api(prompt, model)
        
        # 调试信息（可选显示）
        with st.expander("🔍 查看API响应详情（调试用）"):
            st.text(response)
        
        # 尝试解析JSON，特别处理LaTeX特殊字符
        try:
            result = json.loads(response)
            # 如果是LaTeX格式，确保特殊字符正确显示
            if resume_format == "LaTeX" and "optimized_resume" in result:
                # JSON解析会自动处理转义字符，这里不需要额外处理
                pass
            return result
        except json.JSONDecodeError as e:
            # 如果不是纯JSON，尝试提取JSON部分
            import re
            # 更强大的JSON提取，处理可能包含LaTeX特殊字符的情况
            json_match = re.search(r'\{.*\}', response, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # 如果还是解析失败，尝试修复常见的JSON格式问题
                    # 处理可能的LaTeX转义问题
                    cleaned_json = json_str.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                    try:
                        return json.loads(cleaned_json)
                    except:
                        pass
            
            # 最后的备用方案
            return {
                "match_analysis": f"AI响应解析失败: {str(e)}",
                "suggestions": ["请重新尝试，可能是LaTeX特殊字符导致的JSON解析问题"],
                "optimized_resume": response,
                "key_improvements": ["需要修复JSON格式问题，可能与LaTeX特殊字符有关"]
            }
                
    except Exception as e:
        st.error(f"AI处理出错：{str(e)}")
        return {"error": "处理失败，请重试"}

# 连接状态检查函数
def check_api_connection():
    """检查API连接状态"""
    try:
        test_response = requests.get("https://api.siliconflow.cn", timeout=5)
        return True
    except:
        return False

# 优化按钮
st.divider()
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    if st.button("🚀 开始优化简历", use_container_width=True, type="primary"):
        if jd_text.strip() and resume_text.strip():
            # 先检查连接状态
            with st.spinner("检查网络连接..."):
                if not check_api_connection():
                    st.error("❌ 无法连接到AI服务，请检查网络连接后重试")
                else:
                    with st.spinner("AI正在分析JD和简历，生成优化建议...这可能需要1-2分钟"):
                        try:
                            # 显示处理进度
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("正在连接AI服务...")
                            progress_bar.progress(20)
                            
                            # 调用AI优化函数，传入用户选择的模型
                            result = optimize_resume_with_ai(jd_text, resume_text, resume_format, ai_model)
                            
                            progress_bar.progress(80)
                            status_text.text("正在处理结果...")
                            
                            # 检查结果是否有效
                            if result and 'error' not in result:
                                # 存储结果到session state
                                st.session_state.optimization_result = result
                                st.session_state.original_resume = resume_text
                                st.session_state.resume_format = resume_format
                                
                                progress_bar.progress(100)
                                status_text.text("优化完成！")
                                
                                # 清除进度显示
                                progress_bar.empty()
                                status_text.empty()
                                
                                st.success("✅ 简历优化完成！")
                                st.rerun()
                            else:
                                progress_bar.empty()
                                status_text.empty()
                                st.error("❌ 优化失败，请重试或检查输入内容")
                                
                        except Exception as e:
                            # 清除进度显示
                            try:
                                progress_bar.empty()
                                status_text.empty()
                            except:
                                pass
                            
                            # 更友好的错误信息
                            error_msg = str(e)
                            if "timeout" in error_msg.lower():
                                st.error("❌ 网络连接超时，请稍后重试。建议：\n- 检查网络连接\n- 尝试刷新页面\n- 简化简历内容后重试")
                            elif "connection" in error_msg.lower():
                                st.error("❌ 网络连接失败，请检查网络后重试")
                            elif "429" in error_msg or "频率" in error_msg:
                                st.error("❌ API调用过于频繁，请等待几分钟后重试")
                            else:
                                st.error(f"❌ 处理过程中出错：{error_msg}")
                                
                            # 显示重试建议
                            st.info("💡 如果问题持续出现，请尝试：\n- 等待几分钟后重试\n- 检查网络连接\n- 简化JD或简历内容")
        else:
            st.error("请先输入JD和简历内容！")

# 结果展示区域
if hasattr(st.session_state, 'optimization_result'):
    st.divider()
    st.subheader("📊 优化结果")
    
    # 创建选项卡
    tab1, tab2, tab3 = st.tabs(["💡 优化建议", "✨ 优化后简历", "🔄 对比查看"])
    
    with tab1:
        st.markdown("### 具体优化建议")
        suggestions = st.session_state.optimization_result.get('suggestions', [])
        for i, suggestion in enumerate(suggestions, 1):
            st.markdown(f"**{i}. {suggestion}**")
    
    with tab2:
        st.markdown("### 优化后的简历")
        optimized_resume = st.session_state.optimization_result.get('optimized_resume', '')
        format_used = st.session_state.get('resume_format', 'Word/纯文本')
        st.info(f"📄 格式：{format_used}")
        st.text_area("优化结果", optimized_resume, height=500, disabled=True)
        
        # 复制按钮
        if st.button("📋 复制优化后的简历"):
            st.write("请手动复制上方文本")
    
    with tab3:
        st.markdown("### 简历对比")
        col_orig, col_opt = st.columns(2)
        
        with col_orig:
            st.markdown("**原简历**")
            st.text_area("原简历内容", st.session_state.original_resume, height=400, disabled=True, key="orig", label_visibility="collapsed")
        
        with col_opt:
            st.markdown("**优化后简历**")
            st.text_area("优化后简历内容", optimized_resume, height=400, disabled=True, key="opt", label_visibility="collapsed")

# 侧边栏信息
with st.sidebar:
    st.markdown("### 📖 使用说明")
    st.markdown("""
    1. **选择格式**: 选择你的简历原始格式（LaTeX/Markdown/纯文本）
    2. **粘贴JD**: 复制完整的岗位描述到左侧框中
    3. **粘贴简历**: 复制你的简历内容到右侧框中  
    4. **点击优化**: AI会根据JD要求优化你的简历
    5. **查看结果**: 在不同标签页中查看建议和优化结果
    """)
    
    st.markdown("### ⚡ 优化亮点")
    st.markdown("""
    - 🎯 **精准匹配**: 根据JD关键词优化简历
    - 📝 **格式保持**: 支持LaTeX、Markdown、纯文本格式
    - 📈 **提升通过率**: 突出相关技能和经验
    - ✨ **专业润色**: 改善表达和格式
    - 🔍 **关键词优化**: 提高ATS系统识别度
    """)
    
    st.markdown("### 🤖 技术支持")
    st.markdown("*Powered by SiliconFlow AI*")

if __name__ == "__main__":
    # 如果没有API密钥，显示设置提示
    if not st.secrets.get("SILICONFLOW_API_KEY"):
        st.warning("请在.streamlit/secrets.toml中配置SILICONFLOW_API_KEY")
