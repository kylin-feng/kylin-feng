"""
分析师Agent - 通义千问qwen-max深度分析
负责实时分析会议内容，提供效率评分、参与度分析、改进建议
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from loguru import logger
import dashscope
from datetime import datetime, timedelta
import re

class AnalystAgent:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化分析师Agent
        
        Args:
            config: 配置信息，包含通义千问配置
        """
        self.config = config
        self.api_key = config.get("dashscope_api_key", "")
        self.model_name = config.get("model_name", "qwen-max")
        
        # 分析配置
        self.analysis_interval = config.get("analysis_interval", 60)  # 分析间隔（秒）
        self.min_segments_for_analysis = config.get("min_segments", 5)
        
        # 分析历史
        self.analysis_history = []
        self.current_analysis = {}
        
        # 评分权重
        self.scoring_weights = {
            "participation": 0.25,      # 参与度
            "focus": 0.20,             # 专注度
            "efficiency": 0.25,        # 效率
            "decision_making": 0.30     # 决策质量
        }
        
        self._setup_dashscope()
    
    def _setup_dashscope(self):
        """设置通义千问API"""
        if self.api_key:
            dashscope.api_key = self.api_key
            logger.info("分析师Agent - 通义千问API配置完成")
        else:
            logger.warning("未配置通义千问API密钥，使用模拟分析")
    
    async def analyze_transcript(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析会议转录内容
        
        Args:
            transcript: 转录记录列表
            
        Returns:
            分析结果
        """
        try:
            if len(transcript) < self.min_segments_for_analysis:
                return {"status": "insufficient_data", "message": "转录内容不足，无法分析"}
            
            logger.info(f"开始分析 {len(transcript)} 条转录记录")
            
            # 并行进行多维度分析
            tasks = [
                self._analyze_participation(transcript),
                self._analyze_focus_level(transcript),
                self._analyze_efficiency(transcript),
                self._analyze_decision_making(transcript),
                self._extract_key_topics(transcript),
                self._analyze_sentiment(transcript)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # 整合分析结果
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "transcript_segments": len(transcript),
                "participation_analysis": results[0],
                "focus_analysis": results[1], 
                "efficiency_analysis": results[2],
                "decision_analysis": results[3],
                "topics_analysis": results[4],
                "sentiment_analysis": results[5],
                "overall_score": self._calculate_overall_score(results[:4]),
                "insights": await self._generate_insights(transcript, results)
            }
            
            # 保存到分析历史
            self.analysis_history.append(analysis_result)
            self.current_analysis = analysis_result
            
            logger.info(f"会议分析完成，综合评分: {analysis_result['overall_score']}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"分析转录内容失败: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def _analyze_participation(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析参与度
        
        Args:
            transcript: 转录记录
            
        Returns:
            参与度分析结果
        """
        try:
            # 统计发言人数据
            speaker_stats = {}
            total_segments = len(transcript)
            
            for record in transcript:
                speaker = record.get("speaker", "未知")
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = {
                        "segments": 0,
                        "total_words": 0,
                        "topics_mentioned": set()
                    }
                
                content = record.get("content", "")
                speaker_stats[speaker]["segments"] += 1
                speaker_stats[speaker]["total_words"] += len(content)
                
                # 提取话题关键词
                topics = self._extract_topics_from_text(content)
                speaker_stats[speaker]["topics_mentioned"].update(topics)
            
            # 计算参与度指标
            participation_scores = {}
            for speaker, stats in speaker_stats.items():
                participation_rate = stats["segments"] / total_segments
                avg_words_per_segment = stats["total_words"] / stats["segments"]
                topic_diversity = len(stats["topics_mentioned"])
                
                # 综合评分
                score = min(100, (
                    participation_rate * 40 +
                    min(avg_words_per_segment / 20, 1) * 30 +
                    min(topic_diversity / 5, 1) * 30
                ))
                
                participation_scores[speaker] = {
                    "participation_rate": round(participation_rate * 100, 2),
                    "segments_count": stats["segments"],
                    "avg_words": round(avg_words_per_segment, 1),
                    "topic_diversity": topic_diversity,
                    "score": round(score, 1)
                }
            
            # 计算整体参与度评分
            if participation_scores:
                avg_participation = sum(s["score"] for s in participation_scores.values()) / len(participation_scores)
                balance_score = 100 - (max(participation_scores.values(), key=lambda x: x["score"])["score"] - 
                                     min(participation_scores.values(), key=lambda x: x["score"])["score"])
            else:
                avg_participation = 0
                balance_score = 0
            
            return {
                "overall_score": round((avg_participation + balance_score) / 2, 1),
                "speaker_scores": participation_scores,
                "total_speakers": len(speaker_stats),
                "balance_score": round(balance_score, 1),
                "insights": self._generate_participation_insights(participation_scores)
            }
            
        except Exception as e:
            logger.error(f"参与度分析失败: {str(e)}")
            return {"overall_score": 0, "error": str(e)}
    
    async def _analyze_focus_level(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析专注度
        
        Args:
            transcript: 转录记录
            
        Returns:
            专注度分析结果
        """
        try:
            # 检测话题转换频率
            topics = []
            interruptions = 0
            off_topic_count = 0
            
            for i, record in enumerate(transcript):
                content = record.get("content", "").lower()
                
                # 提取当前话题
                current_topics = self._extract_topics_from_text(content)
                topics.extend(current_topics)
                
                # 检测打断和跑题
                if self._is_interruption(record, transcript[max(0, i-1):i]):
                    interruptions += 1
                
                if self._is_off_topic(content, current_topics):
                    off_topic_count += 1
            
            # 计算专注度指标
            topic_consistency = self._calculate_topic_consistency(topics)
            interruption_rate = interruptions / len(transcript) if transcript else 0
            off_topic_rate = off_topic_count / len(transcript) if transcript else 0
            
            # 综合专注度评分
            focus_score = min(100, (
                topic_consistency * 40 +
                (1 - interruption_rate) * 30 +
                (1 - off_topic_rate) * 30
            ))
            
            return {
                "overall_score": round(focus_score, 1),
                "topic_consistency": round(topic_consistency * 100, 1),
                "interruption_rate": round(interruption_rate * 100, 2),
                "off_topic_rate": round(off_topic_rate * 100, 2),
                "main_topics": list(set(topics))[:5],
                "insights": self._generate_focus_insights(focus_score, interruption_rate, off_topic_rate)
            }
            
        except Exception as e:
            logger.error(f"专注度分析失败: {str(e)}")
            return {"overall_score": 0, "error": str(e)}
    
    async def _analyze_efficiency(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析会议效率
        
        Args:
            transcript: 转录记录
            
        Returns:
            效率分析结果
        """
        try:
            # 检测决策相关内容
            decisions_made = 0
            action_items = 0
            questions_raised = 0
            questions_answered = 0
            
            for record in transcript:
                content = record.get("content", "").lower()
                
                # 检测决策
                if any(keyword in content for keyword in ["决定", "确定", "同意", "通过", "批准"]):
                    decisions_made += 1
                
                # 检测行动项
                if any(keyword in content for keyword in ["任务", "负责", "完成", "执行", "安排"]):
                    action_items += 1
                
                # 检测问题
                if "？" in record.get("content", "") or any(keyword in content for keyword in ["什么", "怎么", "为什么", "如何"]):
                    questions_raised += 1
                
                # 检测回答
                if any(keyword in content for keyword in ["回答", "解释", "说明", "因为"]):
                    questions_answered += 1
            
            # 计算效率指标
            decision_rate = decisions_made / len(transcript) if transcript else 0
            action_rate = action_items / len(transcript) if transcript else 0
            question_resolution_rate = questions_answered / questions_raised if questions_raised else 1
            
            # 综合效率评分
            efficiency_score = min(100, (
                decision_rate * 100 * 35 +
                action_rate * 100 * 35 +
                question_resolution_rate * 30
            ))
            
            return {
                "overall_score": round(efficiency_score, 1),
                "decisions_made": decisions_made,
                "action_items": action_items,
                "questions_raised": questions_raised,
                "questions_answered": questions_answered,
                "question_resolution_rate": round(question_resolution_rate * 100, 1),
                "insights": self._generate_efficiency_insights(efficiency_score, decisions_made, action_items)
            }
            
        except Exception as e:
            logger.error(f"效率分析失败: {str(e)}")
            return {"overall_score": 0, "error": str(e)}
    
    async def _analyze_decision_making(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析决策质量
        
        Args:
            transcript: 转录记录
            
        Returns:
            决策分析结果
        """
        try:
            decisions = []
            consensus_indicators = 0
            conflict_indicators = 0
            
            for i, record in enumerate(transcript):
                content = record.get("content", "").lower()
                
                # 检测决策点
                if any(keyword in content for keyword in ["决定", "确定", "同意", "最终"]):
                    decisions.append({
                        "content": record.get("content", ""),
                        "speaker": record.get("speaker", ""),
                        "timestamp": record.get("timestamp")
                    })
                
                # 检测共识
                if any(keyword in content for keyword in ["同意", "赞成", "支持", "一致"]):
                    consensus_indicators += 1
                
                # 检测分歧
                if any(keyword in content for keyword in ["不同意", "反对", "分歧", "争议"]):
                    conflict_indicators += 1
            
            # 分析决策质量
            decision_clarity = self._analyze_decision_clarity(decisions)
            consensus_level = consensus_indicators / (consensus_indicators + conflict_indicators + 1)
            
            # 综合决策质量评分
            decision_score = min(100, (
                decision_clarity * 50 +
                consensus_level * 50
            ))
            
            return {
                "overall_score": round(decision_score, 1),
                "total_decisions": len(decisions),
                "consensus_indicators": consensus_indicators,
                "conflict_indicators": conflict_indicators,
                "consensus_level": round(consensus_level * 100, 1),
                "decisions": decisions[-3:],  # 最近3个决策
                "insights": self._generate_decision_insights(decision_score, len(decisions), consensus_level)
            }
            
        except Exception as e:
            logger.error(f"决策分析失败: {str(e)}")
            return {"overall_score": 0, "error": str(e)}
    
    async def _extract_key_topics(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        提取关键话题
        
        Args:
            transcript: 转录记录
            
        Returns:
            话题分析结果
        """
        try:
            # 提取所有话题
            all_topics = []
            for record in transcript:
                content = record.get("content", "")
                topics = self._extract_topics_from_text(content)
                all_topics.extend(topics)
            
            # 统计话题频率
            topic_frequency = {}
            for topic in all_topics:
                topic_frequency[topic] = topic_frequency.get(topic, 0) + 1
            
            # 排序获取主要话题
            main_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "main_topics": [{"topic": topic, "frequency": freq, "importance": freq/len(transcript)} 
                              for topic, freq in main_topics],
                "total_unique_topics": len(topic_frequency),
                "topic_distribution": topic_frequency
            }
            
        except Exception as e:
            logger.error(f"话题提取失败: {str(e)}")
            return {"main_topics": [], "error": str(e)}
    
    async def _analyze_sentiment(self, transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析情感趋势
        
        Args:
            transcript: 转录记录
            
        Returns:
            情感分析结果
        """
        try:
            sentiments = []
            
            for record in transcript:
                content = record.get("content", "").lower()
                sentiment_score = self._calculate_sentiment_score(content)
                sentiments.append(sentiment_score)
            
            # 计算整体情感指标
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            positive_rate = len([s for s in sentiments if s > 0.1]) / len(sentiments) if sentiments else 0
            negative_rate = len([s for s in sentiments if s < -0.1]) / len(sentiments) if sentiments else 0
            
            return {
                "overall_sentiment": round(avg_sentiment, 3),
                "positive_rate": round(positive_rate * 100, 1),
                "negative_rate": round(negative_rate * 100, 1),
                "neutral_rate": round((1 - positive_rate - negative_rate) * 100, 1),
                "sentiment_trend": sentiments[-10:] if len(sentiments) > 10 else sentiments
            }
            
        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            return {"overall_sentiment": 0, "error": str(e)}
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """从文本中提取话题关键词"""
        # 简化的关键词提取
        business_keywords = ["产品", "用户", "功能", "设计", "开发", "测试", "上线", "运营", "数据", "分析"]
        topics = []
        
        for keyword in business_keywords:
            if keyword in text:
                topics.append(keyword)
        
        return topics
    
    def _is_interruption(self, current_record: Dict[str, Any], previous_records: List[Dict[str, Any]]) -> bool:
        """检测是否为打断发言"""
        if not previous_records:
            return False
        
        current_speaker = current_record.get("speaker")
        previous_speaker = previous_records[-1].get("speaker") if previous_records else None
        
        # 简单的打断检测：频繁换人且内容简短
        return (current_speaker != previous_speaker and 
                len(current_record.get("content", "")) < 20)
    
    def _is_off_topic(self, content: str, current_topics: List[str]) -> bool:
        """检测是否跑题"""
        off_topic_indicators = ["顺便说", "插一句", "说起来", "想起来"]
        return any(indicator in content for indicator in off_topic_indicators)
    
    def _calculate_topic_consistency(self, topics: List[str]) -> float:
        """计算话题一致性"""
        if not topics:
            return 1.0
        
        unique_topics = set(topics)
        consistency = 1.0 / (1 + len(unique_topics) / len(topics))
        return min(consistency, 1.0)
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """计算情感分数"""
        positive_words = ["好", "赞", "优秀", "满意", "同意", "支持", "顺利"]
        negative_words = ["不好", "问题", "困难", "反对", "担心", "风险", "延期"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _analyze_decision_clarity(self, decisions: List[Dict[str, Any]]) -> float:
        """分析决策清晰度"""
        if not decisions:
            return 0.0
        
        clear_decisions = 0
        for decision in decisions:
            content = decision["content"].lower()
            if any(word in content for word in ["具体", "明确", "时间", "负责人", "目标"]):
                clear_decisions += 1
        
        return clear_decisions / len(decisions)
    
    def _calculate_overall_score(self, dimension_results: List[Dict[str, Any]]) -> float:
        """计算综合评分"""
        scores = []
        weights = list(self.scoring_weights.values())
        
        for i, result in enumerate(dimension_results):
            if result.get("overall_score") is not None:
                scores.append(result["overall_score"] * weights[i])
        
        return round(sum(scores), 1) if scores else 0
    
    def _generate_participation_insights(self, participation_scores: Dict[str, Any]) -> List[str]:
        """生成参与度洞察"""
        insights = []
        
        if not participation_scores:
            return ["暂无参与度数据"]
        
        scores = [s["score"] for s in participation_scores.values()]
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 80:
            insights.append("团队参与度很高，大家积极发言")
        elif avg_score > 60:
            insights.append("团队参与度良好，可适当鼓励更多发言")
        else:
            insights.append("团队参与度偏低，建议增加互动环节")
        
        # 找出最活跃和最不活跃的参与者
        most_active = max(participation_scores.items(), key=lambda x: x[1]["score"])
        least_active = min(participation_scores.items(), key=lambda x: x[1]["score"])
        
        insights.append(f"{most_active[0]} 参与度最高 ({most_active[1]['score']}分)")
        if most_active[1]["score"] - least_active[1]["score"] > 30:
            insights.append(f"建议鼓励 {least_active[0]} 更多参与讨论")
        
        return insights
    
    def _generate_focus_insights(self, focus_score: float, interruption_rate: float, off_topic_rate: float) -> List[str]:
        """生成专注度洞察"""
        insights = []
        
        if focus_score > 80:
            insights.append("会议专注度很高，讨论目标明确")
        elif focus_score > 60:
            insights.append("会议专注度良好，偶有分散注意")
        else:
            insights.append("会议专注度有待提升")
        
        if interruption_rate > 0.2:
            insights.append("打断频率较高，建议改善发言秩序")
        
        if off_topic_rate > 0.15:
            insights.append("跑题现象较多，建议主持人加强引导")
        
        return insights
    
    def _generate_efficiency_insights(self, efficiency_score: float, decisions_made: int, action_items: int) -> List[str]:
        """生成效率洞察"""
        insights = []
        
        if efficiency_score > 80:
            insights.append("会议效率很高，产出丰富")
        elif efficiency_score > 60:
            insights.append("会议效率良好，有一定产出")
        else:
            insights.append("会议效率偏低，建议明确议题")
        
        if decisions_made == 0:
            insights.append("未形成明确决策，建议加强决策环节")
        elif decisions_made > 5:
            insights.append("决策较多，注意后续执行跟进")
        
        if action_items == 0:
            insights.append("未分配具体任务，建议明确行动计划")
        
        return insights
    
    def _generate_decision_insights(self, decision_score: float, decisions_count: int, consensus_level: float) -> List[str]:
        """生成决策洞察"""
        insights = []
        
        if decision_score > 80:
            insights.append("决策质量很高，共识度强")
        elif decision_score > 60:
            insights.append("决策质量良好，基本达成共识")
        else:
            insights.append("决策质量有待提升")
        
        if consensus_level > 0.8:
            insights.append("团队共识度很高")
        elif consensus_level < 0.4:
            insights.append("存在较多分歧，建议深入沟通")
        
        if decisions_count == 0:
            insights.append("本次会议未产生明确决策")
        
        return insights
    
    async def _generate_insights(self, transcript: List[Dict[str, Any]], analysis_results: List[Dict[str, Any]]) -> List[str]:
        """生成综合洞察和改进建议"""
        insights = []
        
        # 从各维度分析中提取洞察
        for result in analysis_results:
            if "insights" in result:
                insights.extend(result["insights"])
        
        # 添加综合性建议
        overall_score = self._calculate_overall_score(analysis_results[:4])
        
        if overall_score > 85:
            insights.append("🎉 会议质量优秀，团队协作效果很好")
        elif overall_score > 70:
            insights.append("✅ 会议质量良好，有进一步优化空间")  
        elif overall_score > 55:
            insights.append("⚠️ 会议质量一般，建议重点改进参与度和效率")
        else:
            insights.append("❌ 会议质量较低，建议全面优化会议流程")
        
        return insights[:10]  # 返回前10条洞察