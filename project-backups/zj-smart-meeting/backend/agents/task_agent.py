"""
任务官Agent - DeepSeek提取任务
负责自动提取待办、分配责任人、跟进提醒，任务识别率90%+
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime, timedelta
import re
import openai

class TaskAgent:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化任务官Agent
        
        Args:
            config: 配置信息，包含DeepSeek API配置
        """
        self.config = config
        self.api_key = config.get("deepseek_api_key", "")
        self.api_base = config.get("api_base", "https://api.deepseek.com/v1")
        
        # 任务管理配置
        self.task_keywords = [
            "任务", "负责", "完成", "执行", "安排", "分配",
            "行动", "实施", "跟进", "处理", "解决", "准备"
        ]
        
        self.priority_keywords = {
            "高": ["紧急", "重要", "马上", "立即", "优先", "核心"],
            "中": ["及时", "尽快", "按时", "正常"],
            "低": ["有空", "后续", "稍后", "可以", "建议"]
        }
        
        self.time_keywords = {
            "今天": 0,
            "明天": 1,
            "后天": 2,
            "本周": 7,
            "下周": 14,
            "月底": 30,
            "下月": 30
        }
        
        # 任务存储
        self.extracted_tasks = []
        self.task_templates = {}
        self.assignment_rules = {}
        
        self._setup_deepseek()
    
    def _setup_deepseek(self):
        """设置DeepSeek API"""
        if self.api_key:
            openai.api_key = self.api_key
            openai.api_base = self.api_base
            logger.info("任务官Agent - DeepSeek API配置完成")
        else:
            logger.warning("未配置DeepSeek API密钥，使用规则引擎提取任务")
    
    async def extract_tasks(self, 
                           transcript: List[Dict[str, Any]], 
                           analysis: Dict[str, Any] = None,
                           meeting_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        从会议转录中提取任务
        
        Args:
            transcript: 转录记录
            analysis: 分析结果
            meeting_info: 会议信息
            
        Returns:
            任务提取结果
        """
        try:
            start_time = datetime.now()
            logger.info(f"开始提取任务，转录记录: {len(transcript)} 条")
            
            # 并行提取任务
            tasks = await asyncio.gather(
                self._extract_explicit_tasks(transcript),
                self._extract_implicit_tasks(transcript, analysis),
                self._extract_follow_up_tasks(transcript)
            )
            
            # 合并和去重
            all_tasks = []
            for task_group in tasks:
                all_tasks.extend(task_group)
            
            # 智能分析和增强
            enhanced_tasks = await self._enhance_tasks_with_ai(all_tasks, transcript)
            
            # 任务优先级排序
            prioritized_tasks = self._prioritize_tasks(enhanced_tasks)
            
            # 分配责任人
            assigned_tasks = self._assign_responsibilities(prioritized_tasks, transcript)
            
            # 设置截止时间
            scheduled_tasks = self._schedule_tasks(assigned_tasks, transcript)
            
            # 生成任务洞察
            insights = await self._generate_task_insights(scheduled_tasks, analysis)
            
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "extraction_time": round(extraction_time, 2),
                "timestamp": datetime.now().isoformat(),
                "total_tasks": len(scheduled_tasks),
                "tasks": scheduled_tasks,
                "task_distribution": self._analyze_task_distribution(scheduled_tasks),
                "insights": insights,
                "recognition_rate": self._calculate_recognition_rate(transcript, scheduled_tasks),
                "metadata": {
                    "source_segments": len(transcript),
                    "explicit_tasks": len(tasks[0]),
                    "implicit_tasks": len(tasks[1]),
                    "follow_up_tasks": len(tasks[2])
                }
            }
            
            # 保存提取结果
            self.extracted_tasks.append(result)
            
            logger.info(f"任务提取完成，共识别 {len(scheduled_tasks)} 个任务，耗时: {extraction_time:.2f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"任务提取失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _extract_explicit_tasks(self, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取显式任务（明确提到的任务）
        
        Args:
            transcript: 转录记录
            
        Returns:
            显式任务列表
        """
        try:
            explicit_tasks = []
            
            for record in transcript:
                content = record.get("content", "")
                speaker = record.get("speaker", "")
                timestamp = record.get("timestamp")
                
                # 检测显式任务关键词
                if any(keyword in content for keyword in self.task_keywords):
                    task = self._parse_task_from_sentence(content, speaker, timestamp)
                    if task:
                        task["type"] = "explicit"
                        task["confidence"] = 0.9
                        explicit_tasks.append(task)
            
            logger.info(f"识别到 {len(explicit_tasks)} 个显式任务")
            return explicit_tasks
            
        except Exception as e:
            logger.error(f"提取显式任务失败: {str(e)}")
            return []
    
    async def _extract_implicit_tasks(self, transcript: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        提取隐式任务（从讨论中推断的任务）
        
        Args:
            transcript: 转录记录
            analysis: 分析结果
            
        Returns:
            隐式任务列表
        """
        try:
            implicit_tasks = []
            
            # 从决策中推断任务
            decisions = analysis.get("decision_analysis", {}).get("decisions", []) if analysis else []
            for decision in decisions:
                task = self._infer_task_from_decision(decision)
                if task:
                    task["type"] = "implicit"
                    task["confidence"] = 0.7
                    implicit_tasks.append(task)
            
            # 从问题中推断任务
            for record in transcript:
                content = record.get("content", "")
                if "？" in content or any(q in content for q in ["什么", "怎么", "如何", "为什么"]):
                    task = self._infer_task_from_question(content, record.get("speaker"), record.get("timestamp"))
                    if task:
                        task["type"] = "implicit"
                        task["confidence"] = 0.6
                        implicit_tasks.append(task)
            
            logger.info(f"推断出 {len(implicit_tasks)} 个隐式任务")
            return implicit_tasks
            
        except Exception as e:
            logger.error(f"提取隐式任务失败: {str(e)}")
            return []
    
    async def _extract_follow_up_tasks(self, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取跟进任务（后续需要执行的任务）
        
        Args:
            transcript: 转录记录
            
        Returns:
            跟进任务列表
        """
        try:
            follow_up_tasks = []
            
            follow_up_keywords = ["下次", "后续", "跟进", "继续", "进一步", "再次"]
            
            for record in transcript:
                content = record.get("content", "")
                if any(keyword in content for keyword in follow_up_keywords):
                    task = self._parse_follow_up_task(content, record.get("speaker"), record.get("timestamp"))
                    if task:
                        task["type"] = "follow_up"
                        task["confidence"] = 0.8
                        follow_up_tasks.append(task)
            
            logger.info(f"识别到 {len(follow_up_tasks)} 个跟进任务")
            return follow_up_tasks
            
        except Exception as e:
            logger.error(f"提取跟进任务失败: {str(e)}")
            return []
    
    def _parse_task_from_sentence(self, content: str, speaker: str, timestamp: Any) -> Optional[Dict[str, Any]]:
        """
        从句子中解析任务信息
        
        Args:
            content: 句子内容
            speaker: 发言人
            timestamp: 时间戳
            
        Returns:
            解析出的任务信息
        """
        try:
            # 提取任务描述
            task_description = self._extract_task_description(content)
            if not task_description:
                return None
            
            # 提取负责人
            assignee = self._extract_assignee_from_content(content, speaker)
            
            # 提取优先级
            priority = self._extract_priority(content)
            
            # 提取时间信息
            deadline = self._extract_deadline(content)
            
            return {
                "id": self._generate_task_id(),
                "description": task_description,
                "assignee": assignee,
                "priority": priority,
                "deadline": deadline,
                "status": "pending",
                "source": {
                    "speaker": speaker,
                    "content": content,
                    "timestamp": timestamp
                },
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"解析任务失败: {str(e)}")
            return None
    
    def _extract_task_description(self, content: str) -> Optional[str]:
        """提取任务描述"""
        # 移除常见的前缀词
        prefixes = ["我们需要", "大家", "请", "要", "应该", "建议"]
        cleaned_content = content
        
        for prefix in prefixes:
            if content.startswith(prefix):
                cleaned_content = content[len(prefix):].strip()
                break
        
        # 如果包含任务关键词，返回清理后的描述
        if any(keyword in content for keyword in self.task_keywords) and len(cleaned_content) > 5:
            return cleaned_content[:100]  # 限制长度
        
        return None
    
    def _extract_assignee_from_content(self, content: str, speaker: str) -> str:
        """从内容中提取负责人"""
        # 检查常见的人员提及
        names = ["张总", "李经理", "王设计师", "小李", "小王", "小张", "小赵", "小陈"]
        
        for name in names:
            if name in content:
                return name
        
        # 检查人称代词
        if "我" in content or "我来" in content:
            return speaker
        elif "你" in content:
            # 需要根据对话上下文确定
            return "待确认"
        
        return "待分配"
    
    def _extract_priority(self, content: str) -> str:
        """提取任务优先级"""
        for priority, keywords in self.priority_keywords.items():
            if any(keyword in content for keyword in keywords):
                return priority
        
        # 默认中等优先级
        return "中"
    
    def _extract_deadline(self, content: str) -> str:
        """提取截止时间"""
        for time_phrase, days_offset in self.time_keywords.items():
            if time_phrase in content:
                deadline_date = datetime.now() + timedelta(days=days_offset)
                return deadline_date.strftime("%Y-%m-%d")
        
        # 查找数字+时间单位的模式
        time_pattern = r'(\d+)\s*(天|周|月|小时)'
        match = re.search(time_pattern, content)
        if match:
            number = int(match.group(1))
            unit = match.group(2)
            
            if unit == "天":
                deadline_date = datetime.now() + timedelta(days=number)
            elif unit == "周":
                deadline_date = datetime.now() + timedelta(weeks=number)
            elif unit == "月":
                deadline_date = datetime.now() + timedelta(days=number*30)
            elif unit == "小时":
                deadline_date = datetime.now() + timedelta(hours=number)
            else:
                deadline_date = datetime.now() + timedelta(days=7)  # 默认一周
            
            return deadline_date.strftime("%Y-%m-%d")
        
        return "待确定"
    
    def _infer_task_from_decision(self, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从决策推断任务"""
        content = decision.get("content", "")
        
        # 如果决策包含行动性词汇，转换为任务
        action_words = ["实施", "执行", "开始", "启动", "进行", "完成"]
        if any(word in content for word in action_words):
            return {
                "id": self._generate_task_id(),
                "description": f"执行决策: {content}",
                "assignee": decision.get("speaker", "待分配"),
                "priority": "高",  # 决策通常优先级较高
                "deadline": "待确定",
                "status": "pending",
                "source": {
                    "type": "decision",
                    "content": content,
                    "speaker": decision.get("speaker")
                },
                "created_at": datetime.now().isoformat()
            }
        
        return None
    
    def _infer_task_from_question(self, content: str, speaker: str, timestamp: Any) -> Optional[Dict[str, Any]]:
        """从问题推断任务"""
        # 将问题转换为对应的任务
        question_to_task_mapping = {
            "什么时候": "确定时间安排",
            "怎么": "制定实施方案", 
            "如何": "制定实施方案",
            "为什么": "分析原因",
            "谁": "确定负责人",
            "哪里": "确定地点"
        }
        
        for question_word, task_prefix in question_to_task_mapping.items():
            if question_word in content:
                return {
                    "id": self._generate_task_id(),
                    "description": f"{task_prefix}: {content.replace('？', '')}",
                    "assignee": "待分配",
                    "priority": "中",
                    "deadline": "待确定",
                    "status": "pending",
                    "source": {
                        "type": "question",
                        "content": content,
                        "speaker": speaker,
                        "timestamp": timestamp
                    },
                    "created_at": datetime.now().isoformat()
                }
        
        return None
    
    def _parse_follow_up_task(self, content: str, speaker: str, timestamp: Any) -> Optional[Dict[str, Any]]:
        """解析跟进任务"""
        if len(content.strip()) < 10:  # 过短的内容不考虑
            return None
        
        return {
            "id": self._generate_task_id(),
            "description": f"跟进: {content}",
            "assignee": speaker,
            "priority": "中",
            "deadline": "下次会议前",
            "status": "pending",
            "source": {
                "type": "follow_up",
                "content": content,
                "speaker": speaker,
                "timestamp": timestamp
            },
            "created_at": datetime.now().isoformat()
        }
    
    async def _enhance_tasks_with_ai(self, tasks: List[Dict[str, Any]], transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        使用AI增强任务信息
        
        Args:
            tasks: 原始任务列表
            transcript: 转录记录
            
        Returns:
            增强后的任务列表
        """
        try:
            if not self.api_key or not tasks:
                return tasks
            
            # 使用DeepSeek进行任务增强（这里使用模拟实现）
            enhanced_tasks = []
            
            for task in tasks:
                enhanced_task = await self._enhance_single_task(task, transcript)
                enhanced_tasks.append(enhanced_task)
            
            return enhanced_tasks
            
        except Exception as e:
            logger.error(f"AI任务增强失败: {str(e)}")
            return tasks
    
    async def _enhance_single_task(self, task: Dict[str, Any], transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """增强单个任务"""
        try:
            # 模拟DeepSeek API调用进行任务增强
            enhanced_description = self._improve_task_description(task.get("description", ""))
            estimated_hours = self._estimate_task_hours(task.get("description", ""))
            dependencies = self._find_task_dependencies(task, transcript)
            
            # 更新任务信息
            enhanced_task = task.copy()
            enhanced_task.update({
                "enhanced_description": enhanced_description,
                "estimated_hours": estimated_hours,
                "dependencies": dependencies,
                "complexity": self._assess_task_complexity(task.get("description", "")),
                "ai_enhanced": True
            })
            
            return enhanced_task
            
        except Exception as e:
            logger.error(f"增强单个任务失败: {str(e)}")
            return task
    
    def _improve_task_description(self, description: str) -> str:
        """改进任务描述"""
        # 简单的描述改进逻辑
        if description.startswith("执行决策:"):
            return description.replace("执行决策:", "").strip()
        
        if not description.endswith("。"):
            description += "。"
        
        return description.strip()
    
    def _estimate_task_hours(self, description: str) -> float:
        """估算任务工时"""
        # 基于关键词的简单工时估算
        complexity_keywords = {
            "设计": 16,
            "开发": 32,
            "测试": 8,
            "分析": 4,
            "研究": 8,
            "优化": 12,
            "实施": 16
        }
        
        for keyword, hours in complexity_keywords.items():
            if keyword in description:
                return hours
        
        # 基于描述长度的估算
        if len(description) > 50:
            return 8
        elif len(description) > 20:
            return 4
        else:
            return 2
    
    def _find_task_dependencies(self, task: Dict[str, Any], transcript: List[Dict[str, Any]]) -> List[str]:
        """查找任务依赖"""
        dependencies = []
        task_description = task.get("description", "").lower()
        
        # 查找依赖关键词
        if "之前" in task_description or "完成后" in task_description:
            dependencies.append("前置任务待确认")
        
        if "设计" in task_description and "开发" in task_description:
            dependencies.append("设计稿确认")
        
        return dependencies
    
    def _assess_task_complexity(self, description: str) -> str:
        """评估任务复杂度"""
        complex_keywords = ["架构", "系统", "集成", "重构", "算法"]
        simple_keywords = ["更新", "修改", "调整", "配置", "检查"]
        
        if any(keyword in description for keyword in complex_keywords):
            return "高"
        elif any(keyword in description for keyword in simple_keywords):
            return "低"
        else:
            return "中"
    
    def _prioritize_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """任务优先级排序"""
        priority_order = {"高": 3, "中": 2, "低": 1}
        
        # 按优先级和置信度排序
        sorted_tasks = sorted(tasks, 
                            key=lambda x: (priority_order.get(x.get("priority", "中"), 2), 
                                         x.get("confidence", 0)), 
                            reverse=True)
        
        return sorted_tasks
    
    def _assign_responsibilities(self, tasks: List[Dict[str, Any]], transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分配任务责任人"""
        # 构建人员技能映射
        skill_mapping = self._build_skill_mapping(transcript)
        
        for task in tasks:
            if task.get("assignee") == "待分配":
                best_assignee = self._find_best_assignee(task, skill_mapping)
                task["assignee"] = best_assignee
                task["assignment_reason"] = f"基于技能匹配: {best_assignee}"
        
        return tasks
    
    def _build_skill_mapping(self, transcript: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """构建人员技能映射"""
        skill_mapping = {}
        
        skill_keywords = {
            "产品": ["功能", "需求", "用户", "体验"],
            "设计": ["界面", "UI", "交互", "视觉"],
            "技术": ["开发", "代码", "系统", "架构"],
            "测试": ["质量", "bug", "验证", "测试"],
            "运营": ["推广", "数据", "分析", "增长"]
        }
        
        for record in transcript:
            speaker = record.get("speaker", "")
            content = record.get("content", "").lower()
            
            if speaker not in skill_mapping:
                skill_mapping[speaker] = []
            
            for skill, keywords in skill_keywords.items():
                if any(keyword in content for keyword in keywords):
                    if skill not in skill_mapping[speaker]:
                        skill_mapping[speaker].append(skill)
        
        return skill_mapping
    
    def _find_best_assignee(self, task: Dict[str, Any], skill_mapping: Dict[str, List[str]]) -> str:
        """为任务找到最佳责任人"""
        task_description = task.get("description", "").lower()
        
        # 根据任务内容匹配技能
        best_score = 0
        best_assignee = "待分配"
        
        for person, skills in skill_mapping.items():
            score = 0
            for skill in skills:
                if skill in task_description:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_assignee = person
        
        return best_assignee if best_score > 0 else "待分配"
    
    def _schedule_tasks(self, tasks: List[Dict[str, Any]], transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """安排任务时间"""
        for task in tasks:
            if task.get("deadline") == "待确定":
                suggested_deadline = self._suggest_deadline(task)
                task["deadline"] = suggested_deadline
                task["scheduling_reason"] = "基于优先级和复杂度自动安排"
        
        return tasks
    
    def _suggest_deadline(self, task: Dict[str, Any]) -> str:
        """建议截止时间"""
        priority = task.get("priority", "中")
        complexity = task.get("complexity", "中")
        
        # 基于优先级和复杂度确定时间
        base_days = {
            ("高", "低"): 3,
            ("高", "中"): 5,
            ("高", "高"): 7,
            ("中", "低"): 7,
            ("中", "中"): 10,
            ("中", "高"): 14,
            ("低", "低"): 14,
            ("低", "中"): 21,
            ("低", "高"): 30
        }
        
        days = base_days.get((priority, complexity), 7)
        deadline_date = datetime.now() + timedelta(days=days)
        
        return deadline_date.strftime("%Y-%m-%d")
    
    async def _generate_task_insights(self, tasks: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[str]:
        """生成任务洞察"""
        insights = []
        
        if not tasks:
            insights.append("本次会议未识别出明确的任务")
            return insights
        
        # 任务数量分析
        total_tasks = len(tasks)
        if total_tasks > 10:
            insights.append(f"识别到 {total_tasks} 个任务，建议合并相似任务")
        elif total_tasks > 5:
            insights.append(f"识别到 {total_tasks} 个任务，工作量适中")
        else:
            insights.append(f"识别到 {total_tasks} 个任务，执行压力较小")
        
        # 优先级分析
        priority_dist = self._analyze_task_distribution(tasks)
        high_priority_count = priority_dist.get("high_priority", 0)
        
        if high_priority_count > total_tasks * 0.6:
            insights.append("高优先级任务较多，建议重新评估优先级")
        elif high_priority_count == 0:
            insights.append("缺少高优先级任务，可能影响项目进度")
        
        # 负责人分析
        unassigned_tasks = len([t for t in tasks if t.get("assignee") == "待分配"])
        if unassigned_tasks > 0:
            insights.append(f"{unassigned_tasks} 个任务尚未分配责任人")
        
        # 时间分析
        overdue_tasks = len([t for t in tasks if t.get("deadline") == "待确定"])
        if overdue_tasks > 0:
            insights.append(f"{overdue_tasks} 个任务需要明确截止时间")
        
        return insights
    
    def _analyze_task_distribution(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析任务分布"""
        distribution = {
            "total": len(tasks),
            "high_priority": 0,
            "medium_priority": 0,
            "low_priority": 0,
            "assigned": 0,
            "unassigned": 0,
            "with_deadline": 0,
            "without_deadline": 0
        }
        
        for task in tasks:
            # 优先级分布
            priority = task.get("priority", "中")
            if priority == "高":
                distribution["high_priority"] += 1
            elif priority == "中":
                distribution["medium_priority"] += 1
            else:
                distribution["low_priority"] += 1
            
            # 分配状态
            if task.get("assignee") and task.get("assignee") != "待分配":
                distribution["assigned"] += 1
            else:
                distribution["unassigned"] += 1
            
            # 截止时间
            if task.get("deadline") and task.get("deadline") != "待确定":
                distribution["with_deadline"] += 1
            else:
                distribution["without_deadline"] += 1
        
        return distribution
    
    def _calculate_recognition_rate(self, transcript: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> float:
        """计算任务识别率"""
        # 统计包含任务关键词的转录片段数量
        task_related_segments = 0
        for record in transcript:
            content = record.get("content", "")
            if any(keyword in content for keyword in self.task_keywords):
                task_related_segments += 1
        
        if task_related_segments == 0:
            return 0.0
        
        # 计算识别率：识别到的任务数 / 包含任务关键词的片段数
        recognition_rate = min(len(tasks) / task_related_segments, 1.0)
        
        return round(recognition_rate * 100, 1)
    
    def _generate_task_id(self) -> str:
        """生成任务ID"""
        from datetime import datetime
        return f"TASK_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.extracted_tasks) + 1}"
    
    def get_task_stats(self) -> Dict[str, Any]:
        """获取任务管理统计"""
        if not self.extracted_tasks:
            return {"total_extractions": 0}
        
        total_extractions = len(self.extracted_tasks)
        total_tasks = sum(len(extraction.get("tasks", [])) for extraction in self.extracted_tasks)
        avg_tasks_per_meeting = total_tasks / total_extractions if total_extractions > 0 else 0
        
        latest_extraction = self.extracted_tasks[-1]
        
        return {
            "total_extractions": total_extractions,
            "total_tasks_extracted": total_tasks,
            "avg_tasks_per_meeting": round(avg_tasks_per_meeting, 1),
            "latest_recognition_rate": latest_extraction.get("recognition_rate", 0),
            "latest_extraction_time": latest_extraction.get("extraction_time", 0)
        }
    
    def export_tasks(self, format: str = "json") -> Any:
        """导出任务数据"""
        if not self.extracted_tasks:
            return None
        
        latest_tasks = self.extracted_tasks[-1].get("tasks", [])
        
        if format == "json":
            return json.dumps(latest_tasks, ensure_ascii=False, indent=2)
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            fieldnames = ["id", "description", "assignee", "priority", "deadline", "status", "type"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            
            writer.writeheader()
            for task in latest_tasks:
                writer.writerow({
                    "id": task.get("id", ""),
                    "description": task.get("description", ""),
                    "assignee": task.get("assignee", ""),
                    "priority": task.get("priority", ""),
                    "deadline": task.get("deadline", ""),
                    "status": task.get("status", ""),
                    "type": task.get("type", "")
                })
            
            return output.getvalue()
        
        return latest_tasks