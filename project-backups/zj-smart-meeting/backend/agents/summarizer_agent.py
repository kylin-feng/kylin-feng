"""
总结者Agent - 通义千问qwen-plus生成纪要
负责生成快速/标准/详细三版本会议纪要，3分钟内完成
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from loguru import logger
import dashscope
from datetime import datetime
import re

class SummarizerAgent:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化总结者Agent
        
        Args:
            config: 配置信息，包含通义千问配置
        """
        self.config = config
        self.api_key = config.get("dashscope_api_key", "")
        self.model_name = config.get("model_name", "qwen-plus")
        
        # 纪要模板配置
        self.summary_templates = {
            "quick": {
                "max_length": 500,
                "sections": ["核心决策", "关键任务"],
                "style": "简洁明了"
            },
            "standard": {
                "max_length": 1500,
                "sections": ["会议概述", "主要讨论", "决策事项", "行动计划"],
                "style": "结构完整"
            },
            "detailed": {
                "max_length": 3000,
                "sections": ["会议信息", "参会人员", "详细讨论", "决策过程", "任务分配", "后续跟进"],
                "style": "全面详细"
            }
        }
        
        # 生成历史
        self.summary_history = []
        
        self._setup_dashscope()
    
    def _setup_dashscope(self):
        """设置通义千问API"""
        if self.api_key:
            dashscope.api_key = self.api_key
            logger.info("总结者Agent - 通义千问API配置完成")
        else:
            logger.warning("未配置通义千问API密钥，使用模拟纪要生成")
    
    async def generate_summary(self, 
                             transcript: List[Dict[str, Any]], 
                             analysis: Dict[str, Any] = None,
                             meeting_info: Dict[str, Any] = None,
                             summary_types: List[str] = None) -> Dict[str, Any]:
        """
        生成会议纪要
        
        Args:
            transcript: 转录记录
            analysis: 分析结果
            meeting_info: 会议信息
            summary_types: 纪要类型列表，默认生成全部三种
            
        Returns:
            纪要生成结果
        """
        try:
            start_time = datetime.now()
            logger.info(f"开始生成会议纪要，转录记录: {len(transcript)} 条")
            
            # 默认生成所有类型的纪要
            if summary_types is None:
                summary_types = ["quick", "standard", "detailed"]
            
            # 预处理数据
            processed_data = await self._preprocess_data(transcript, analysis, meeting_info)
            
            # 并行生成不同类型的纪要
            tasks = []
            for summary_type in summary_types:
                tasks.append(self._generate_summary_by_type(processed_data, summary_type))
            
            results = await asyncio.gather(*tasks)
            
            # 整合结果
            summaries = {}
            for i, summary_type in enumerate(summary_types):
                summaries[summary_type] = results[i]
            
            # 计算生成时间
            generation_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "generation_time": round(generation_time, 2),
                "timestamp": datetime.now().isoformat(),
                "summaries": summaries,
                "metadata": {
                    "source_segments": len(transcript),
                    "meeting_duration": processed_data.get("meeting_duration", "未知"),
                    "participants_count": len(processed_data.get("speakers", [])),
                    "generation_model": self.model_name
                },
                "quality_score": await self._evaluate_summary_quality(summaries)
            }
            
            # 保存到历史记录
            self.summary_history.append(result)
            
            logger.info(f"会议纪要生成完成，耗时: {generation_time:.2f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"生成会议纪要失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _preprocess_data(self, 
                              transcript: List[Dict[str, Any]], 
                              analysis: Dict[str, Any],
                              meeting_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        预处理数据，提取关键信息
        
        Args:
            transcript: 转录记录
            analysis: 分析结果  
            meeting_info: 会议信息
            
        Returns:
            预处理后的数据
        """
        try:
            # 提取基本信息
            speakers = set()
            full_content = []
            key_moments = []
            
            for record in transcript:
                speaker = record.get("speaker", "未知")
                content = record.get("content", "")
                timestamp = record.get("timestamp")
                
                speakers.add(speaker)
                full_content.append(f"[{speaker}] {content}")
                
                # 识别关键时刻
                if self._is_key_moment(content):
                    key_moments.append({
                        "speaker": speaker,
                        "content": content,
                        "timestamp": timestamp,
                        "type": self._classify_key_moment(content)
                    })
            
            # 提取核心主题
            main_topics = self._extract_main_topics(transcript)
            
            # 提取决策和任务
            decisions = self._extract_decisions(transcript)
            tasks = self._extract_tasks_from_transcript(transcript)
            
            # 计算会议时长
            meeting_duration = self._calculate_meeting_duration(transcript)
            
            return {
                "speakers": list(speakers),
                "full_content": "\n".join(full_content),
                "key_moments": key_moments,
                "main_topics": main_topics,
                "decisions": decisions,
                "tasks": tasks,
                "meeting_duration": meeting_duration,
                "meeting_info": meeting_info or {},
                "analysis": analysis or {},
                "total_segments": len(transcript)
            }
            
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_summary_by_type(self, data: Dict[str, Any], summary_type: str) -> Dict[str, Any]:
        """
        根据类型生成特定的纪要
        
        Args:
            data: 预处理后的数据
            summary_type: 纪要类型 (quick/standard/detailed)
            
        Returns:
            特定类型的纪要
        """
        try:
            template = self.summary_templates[summary_type]
            
            if self.api_key:
                # 使用通义千问生成纪要
                summary_content = await self._generate_with_qwen(data, template)
            else:
                # 使用模板生成纪要
                summary_content = await self._generate_with_template(data, template)
            
            return {
                "type": summary_type,
                "content": summary_content,
                "length": len(summary_content),
                "sections": template["sections"],
                "style": template["style"],
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"生成{summary_type}纪要失败: {str(e)}")
            return {"type": summary_type, "error": str(e)}
    
    async def _generate_with_qwen(self, data: Dict[str, Any], template: Dict[str, Any]) -> str:
        """
        使用通义千问生成纪要
        
        Args:
            data: 会议数据
            template: 纪要模板
            
        Returns:
            生成的纪要内容
        """
        try:
            # 构建提示词
            prompt = self._build_summary_prompt(data, template)
            
            # 调用通义千问API（这里使用模拟，实际项目中调用真实API）
            response = await self._call_qwen_api(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"通义千问纪要生成失败: {str(e)}")
            # 降级到模板生成
            return await self._generate_with_template(data, template)
    
    async def _generate_with_template(self, data: Dict[str, Any], template: Dict[str, Any]) -> str:
        """
        使用模板生成纪要
        
        Args:
            data: 会议数据
            template: 纪要模板
            
        Returns:
            生成的纪要内容
        """
        try:
            summary_type = template.get("style", "")
            sections = template.get("sections", [])
            
            content = []
            
            # 添加标题
            meeting_title = data.get("meeting_info", {}).get("title", "会议纪要")
            content.append(f"# {meeting_title}")
            content.append("")
            
            # 根据纪要类型添加不同的内容
            if "quick" in summary_type:
                content.extend(self._generate_quick_summary(data))
            elif "standard" in summary_type:
                content.extend(self._generate_standard_summary(data))
            elif "detailed" in summary_type:
                content.extend(self._generate_detailed_summary(data))
            
            return "\n".join(content)
            
        except Exception as e:
            logger.error(f"模板纪要生成失败: {str(e)}")
            return f"纪要生成出错: {str(e)}"
    
    def _generate_quick_summary(self, data: Dict[str, Any]) -> List[str]:
        """生成快速纪要"""
        content = []
        
        content.append("## 📋 核心决策")
        decisions = data.get("decisions", [])
        if decisions:
            for i, decision in enumerate(decisions[:3], 1):
                content.append(f"{i}. {decision.get('content', '').strip()}")
        else:
            content.append("- 本次会议未产生明确决策")
        
        content.append("")
        content.append("## 🎯 关键任务")
        tasks = data.get("tasks", [])
        if tasks:
            for i, task in enumerate(tasks[:3], 1):
                content.append(f"{i}. {task.get('description', '').strip()}")
                if task.get("assignee"):
                    content.append(f"   负责人: {task.get('assignee')}")
        else:
            content.append("- 暂无明确任务分配")
        
        return content
    
    def _generate_standard_summary(self, data: Dict[str, Any]) -> List[str]:
        """生成标准纪要"""
        content = []
        
        # 会议概述
        content.append("## 📝 会议概述")
        content.append(f"- 会议时长: {data.get('meeting_duration', '未知')}")
        content.append(f"- 参会人员: {', '.join(data.get('speakers', []))}")
        content.append(f"- 主要议题: {', '.join(data.get('main_topics', [])[:3])}")
        content.append("")
        
        # 主要讨论
        content.append("## 💬 主要讨论")
        key_moments = data.get("key_moments", [])
        if key_moments:
            for moment in key_moments[:5]:
                content.append(f"- **{moment.get('speaker')}**: {moment.get('content', '').strip()}")
        else:
            content.append("- 详细讨论内容请参考会议录音")
        content.append("")
        
        # 决策事项
        content.append("## ✅ 决策事项")
        decisions = data.get("decisions", [])
        if decisions:
            for i, decision in enumerate(decisions, 1):
                content.append(f"{i}. {decision.get('content', '').strip()}")
        else:
            content.append("- 本次会议未产生明确决策")
        content.append("")
        
        # 行动计划
        content.append("## 📅 行动计划")
        tasks = data.get("tasks", [])
        if tasks:
            for i, task in enumerate(tasks, 1):
                content.append(f"{i}. {task.get('description', '').strip()}")
                if task.get("assignee"):
                    content.append(f"   - 负责人: {task.get('assignee')}")
                if task.get("deadline"):
                    content.append(f"   - 截止时间: {task.get('deadline')}")
        else:
            content.append("- 暂无明确行动计划")
        
        return content
    
    def _generate_detailed_summary(self, data: Dict[str, Any]) -> List[str]:
        """生成详细纪要"""
        content = []
        
        # 会议信息
        content.append("## ℹ️ 会议信息")
        meeting_info = data.get("meeting_info", {})
        content.append(f"- 会议主题: {meeting_info.get('title', '未命名会议')}")
        content.append(f"- 会议时间: {meeting_info.get('start_time', '未知')}")
        content.append(f"- 会议时长: {data.get('meeting_duration', '未知')}")
        content.append(f"- 会议地点: {meeting_info.get('location', '线上会议')}")
        content.append("")
        
        # 参会人员
        content.append("## 👥 参会人员")
        speakers = data.get("speakers", [])
        for speaker in speakers:
            content.append(f"- {speaker}")
        content.append("")
        
        # 详细讨论
        content.append("## 💭 详细讨论")
        
        # 按主题分组讨论内容
        main_topics = data.get("main_topics", [])
        for topic in main_topics[:3]:
            content.append(f"### {topic}")
            # 找到相关的讨论内容
            topic_discussions = self._get_topic_discussions(data, topic)
            for discussion in topic_discussions[:3]:
                content.append(f"- **{discussion.get('speaker')}**: {discussion.get('content', '').strip()}")
            content.append("")
        
        # 决策过程
        content.append("## 🤔 决策过程")
        analysis = data.get("analysis", {})
        if analysis:
            decision_analysis = analysis.get("decision_analysis", {})
            content.append(f"- 总决策数: {decision_analysis.get('total_decisions', 0)}")
            content.append(f"- 共识水平: {decision_analysis.get('consensus_level', 0)}%")
            if decision_analysis.get("insights"):
                content.append("- 决策质量分析:")
                for insight in decision_analysis.get("insights", [])[:3]:
                    content.append(f"  - {insight}")
        content.append("")
        
        # 任务分配
        content.append("## 📋 任务分配")
        tasks = data.get("tasks", [])
        if tasks:
            for i, task in enumerate(tasks, 1):
                content.append(f"### 任务 {i}: {task.get('description', '').strip()}")
                content.append(f"- 负责人: {task.get('assignee', '待分配')}")
                content.append(f"- 优先级: {task.get('priority', '中')}")
                content.append(f"- 截止时间: {task.get('deadline', '待确定')}")
                content.append(f"- 预估工作量: {task.get('estimated_hours', '未估算')}小时")
                content.append("")
        else:
            content.append("- 本次会议未分配具体任务")
            content.append("")
        
        # 后续跟进
        content.append("## 🔄 后续跟进")
        content.append("- 下次会议时间: 待确定")
        content.append("- 任务进度检查: 建议一周后")
        content.append("- 责任人汇报: 按任务截止时间")
        
        return content
    
    def _build_summary_prompt(self, data: Dict[str, Any], template: Dict[str, Any]) -> str:
        """构建通义千问的提示词"""
        summary_type = template.get("style", "")
        max_length = template.get("max_length", 1000)
        sections = template.get("sections", [])
        
        prompt = f"""
        请根据以下会议转录内容生成{summary_type}的会议纪要：
        
        会议信息：
        - 参会人员：{', '.join(data.get('speakers', []))}
        - 会议时长：{data.get('meeting_duration', '未知')}
        - 主要话题：{', '.join(data.get('main_topics', []))}
        
        转录内容：
        {data.get('full_content', '')[:2000]}  # 限制长度避免token过多
        
        要求：
        1. 纪要长度控制在{max_length}字以内
        2. 包含以下部分：{', '.join(sections)}
        3. 突出重点决策和任务
        4. 使用Markdown格式
        5. 语言简洁专业
        
        请生成纪要：
        """
        
        return prompt
    
    async def _call_qwen_api(self, prompt: str) -> str:
        """调用通义千问API"""
        try:
            # 这里是模拟实现，实际项目中需要调用真实的API
            # 根据提示词长度和复杂度模拟不同的响应
            
            if "快速" in prompt or "quick" in prompt.lower():
                return """## 📋 核心决策
1. 确定Q4产品规划重点关注用户体验提升
2. 登录流程优化目标：提升30%转化率
3. 首页改版项目启动，下周出设计稿

## 🎯 关键任务
1. 登录流程优化 - 负责人：李经理
2. 首页改版设计 - 负责人：王设计师
3. 任务进度跟进 - 负责人：张总"""
            
            elif "标准" in prompt or "standard" in prompt.lower():
                return """## 📝 会议概述
- 会议时长: 约25分钟
- 参会人员: 张总, 李经理, 王设计师
- 主要议题: Q4产品规划, 用户体验优化

## 💬 主要讨论
- **张总**: 讨论Q4产品规划的整体方向
- **李经理**: 提出重点关注用户体验提升的建议
- **王设计师**: 分析UI设计需要调整的具体方面
- **张总**: 确认功能优先级和时间安排

## ✅ 决策事项
1. Q4产品规划重点：用户体验提升
2. 启动登录流程优化项目
3. 启动首页改版项目

## 📅 行动计划
1. 登录流程优化
   - 负责人: 李经理
   - 目标: 提升30%转化率
2. 首页改版设计
   - 负责人: 王设计师
   - 截止时间: 下周出设计稿"""
            
            else:  # detailed
                return """## ℹ️ 会议信息
- 会议主题: Q4产品规划讨论会
- 会议时长: 约25分钟
- 会议地点: 线上会议
- 参会人数: 3人

## 👥 参会人员
- 张总 (主持人)
- 李经理 (产品经理)
- 王设计师 (UI设计师)

## 💭 详细讨论
### 用户体验优化
- **李经理**: 认为应该重点关注用户体验的提升，这是Q4的核心目标
- **王设计师**: UI设计方面需要做一些调整，特别是用户交互流程
- **张总**: 同意用户体验优化的重要性，需要制定具体的改进计划

### 功能规划
- **张总**: 需要确定几个关键的功能点作为重点
- **李经理**: 提出登录流程优化，目标是提升30%的转化率
- **王设计师**: 建议首页改版，提升整体视觉效果和用户体验

## 🤔 决策过程
- 总决策数: 3
- 共识水平: 90%
- 决策质量分析:
  - 团队共识度很高
  - 决策目标明确具体
  - 责任分工清晰

## 📋 任务分配
### 任务1: 登录流程优化
- 负责人: 李经理
- 优先级: 高
- 截止时间: 2周内
- 预估工作量: 40小时

### 任务2: 首页改版设计
- 负责人: 王设计师
- 优先级: 高
- 截止时间: 1周内(出设计稿)
- 预估工作量: 32小时

## 🔄 后续跟进
- 下次会议时间: 1周后
- 任务进度检查: 每周例会
- 责任人汇报: 按任务里程碑时间点"""
        
        except Exception as e:
            logger.error(f"调用通义千问API失败: {str(e)}")
            return "纪要生成失败，请稍后重试"
    
    def _is_key_moment(self, content: str) -> bool:
        """判断是否为关键时刻"""
        key_indicators = ["决定", "确定", "同意", "任务", "负责", "安排", "问题", "建议"]
        return any(indicator in content for indicator in key_indicators)
    
    def _classify_key_moment(self, content: str) -> str:
        """分类关键时刻类型"""
        if any(word in content for word in ["决定", "确定", "同意"]):
            return "决策"
        elif any(word in content for word in ["任务", "负责", "安排"]):
            return "任务"
        elif "问题" in content:
            return "问题"
        elif "建议" in content:
            return "建议"
        else:
            return "讨论"
    
    def _extract_main_topics(self, transcript: List[Dict[str, Any]]) -> List[str]:
        """提取主要话题"""
        topics = []
        topic_keywords = ["产品", "用户", "功能", "设计", "开发", "测试", "运营", "数据"]
        
        for record in transcript:
            content = record.get("content", "").lower()
            for keyword in topic_keywords:
                if keyword in content and keyword not in topics:
                    topics.append(keyword)
        
        return topics[:5]
    
    def _extract_decisions(self, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取决策信息"""
        decisions = []
        
        for record in transcript:
            content = record.get("content", "")
            if any(word in content for word in ["决定", "确定", "同意", "批准"]):
                decisions.append({
                    "content": content,
                    "speaker": record.get("speaker"),
                    "timestamp": record.get("timestamp")
                })
        
        return decisions
    
    def _extract_tasks_from_transcript(self, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从转录中提取任务"""
        tasks = []
        
        for record in transcript:
            content = record.get("content", "")
            if any(word in content for word in ["任务", "负责", "完成", "安排"]):
                # 尝试提取责任人
                assignee = self._extract_assignee(content, record.get("speaker"))
                
                tasks.append({
                    "description": content,
                    "assignee": assignee,
                    "priority": "中",
                    "deadline": "待确定",
                    "estimated_hours": "未估算",
                    "extracted_from": record.get("speaker"),
                    "timestamp": record.get("timestamp")
                })
        
        return tasks
    
    def _extract_assignee(self, content: str, speaker: str) -> str:
        """提取任务负责人"""
        # 简单的负责人提取逻辑
        if "我" in content or "自己" in content:
            return speaker
        
        # 检查是否提到了其他人
        names = ["张总", "李经理", "王设计师", "小李", "小王", "小张"]
        for name in names:
            if name in content:
                return name
        
        return "待分配"
    
    def _get_topic_discussions(self, data: Dict[str, Any], topic: str) -> List[Dict[str, Any]]:
        """获取特定话题的讨论内容"""
        discussions = []
        
        for record in data.get("key_moments", []):
            if topic in record.get("content", ""):
                discussions.append(record)
        
        return discussions
    
    def _calculate_meeting_duration(self, transcript: List[Dict[str, Any]]) -> str:
        """计算会议时长"""
        if len(transcript) < 2:
            return "未知"
        
        try:
            first_time = transcript[0].get("timestamp", 0)
            last_time = transcript[-1].get("timestamp", 0)
            
            if isinstance(first_time, (int, float)) and isinstance(last_time, (int, float)):
                duration_seconds = last_time - first_time
                duration_minutes = int(duration_seconds / 60)
                return f"约{duration_minutes}分钟"
            else:
                return "未知"
        except:
            return "未知"
    
    async def _evaluate_summary_quality(self, summaries: Dict[str, Any]) -> Dict[str, Any]:
        """评估纪要质量"""
        try:
            quality_scores = {}
            
            for summary_type, summary_data in summaries.items():
                if "error" in summary_data:
                    quality_scores[summary_type] = 0
                    continue
                
                content = summary_data.get("content", "")
                
                # 评估指标
                completeness = self._evaluate_completeness(content, summary_type)
                clarity = self._evaluate_clarity(content)
                structure = self._evaluate_structure(content)
                
                # 综合评分
                overall_score = (completeness + clarity + structure) / 3
                
                quality_scores[summary_type] = {
                    "overall_score": round(overall_score, 1),
                    "completeness": round(completeness, 1),
                    "clarity": round(clarity, 1),
                    "structure": round(structure, 1)
                }
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"评估纪要质量失败: {str(e)}")
            return {}
    
    def _evaluate_completeness(self, content: str, summary_type: str) -> float:
        """评估纪要完整性"""
        required_sections = self.summary_templates[summary_type]["sections"]
        
        found_sections = 0
        for section in required_sections:
            if any(keyword in content for keyword in section.split()):
                found_sections += 1
        
        return (found_sections / len(required_sections)) * 100
    
    def _evaluate_clarity(self, content: str) -> float:
        """评估纪要清晰度"""
        # 检查是否有清晰的结构标记
        structure_markers = content.count("#") + content.count("**") + content.count("-")
        
        # 检查句子长度合理性
        sentences = content.split("。")
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        # 清晰度评分
        structure_score = min(structure_markers / 10, 1) * 50
        length_score = 50 if 10 <= avg_sentence_length <= 50 else 30
        
        return structure_score + length_score
    
    def _evaluate_structure(self, content: str) -> float:
        """评估纪要结构性"""
        # 检查标题层级
        has_headers = "##" in content
        has_lists = "-" in content or "1." in content
        has_emphasis = "**" in content
        
        structure_score = 0
        if has_headers:
            structure_score += 40
        if has_lists:
            structure_score += 40  
        if has_emphasis:
            structure_score += 20
        
        return structure_score
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取纪要生成统计"""
        if not self.summary_history:
            return {"total_summaries": 0}
        
        total_summaries = len(self.summary_history)
        avg_generation_time = sum(s.get("generation_time", 0) for s in self.summary_history) / total_summaries
        
        return {
            "total_summaries": total_summaries,
            "avg_generation_time": round(avg_generation_time, 2),
            "latest_generation_time": self.summary_history[-1].get("generation_time", 0),
            "quality_trend": [s.get("quality_score", {}) for s in self.summary_history[-5:]]
        }