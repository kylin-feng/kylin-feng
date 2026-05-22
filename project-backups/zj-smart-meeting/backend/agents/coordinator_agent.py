"""
协调员Agent - LangGraph工作流编排
负责协调6个智能体的工作流程，实现多智能体协作
"""

from typing import Dict, List, Any
from langgraph import LangGraph, Node, Edge
from langchain.schema import BaseMessage
import asyncio
import json
from loguru import logger

class CoordinatorAgent:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化协调员Agent
        
        Args:
            config: 配置信息，包含各个Agent的配置
        """
        self.config = config
        self.workflow_graph = None
        self.agents = {}
        self.meeting_state = {
            "status": "待开始",
            "participants": [],
            "transcript": [],
            "analysis": {},
            "summary": {},
            "tasks": [],
            "knowledge": {}
        }
        self._build_workflow()
        
    def _build_workflow(self):
        """
        构建多智能体协作工作流
        使用LangGraph定义智能体间的协作关系
        """
        # 创建工作流图
        self.workflow_graph = LangGraph()
        
        # 定义节点（各个Agent）
        recorder_node = Node(
            name="recorder",
            function=self._recorder_process,
            description="实时转录会议内容"
        )
        
        analyst_node = Node(
            name="analyst", 
            function=self._analyst_process,
            description="分析会议内容和参与情况"
        )
        
        summarizer_node = Node(
            name="summarizer",
            function=self._summarizer_process, 
            description="生成会议纪要"
        )
        
        task_node = Node(
            name="task_agent",
            function=self._task_process,
            description="提取和管理任务"
        )
        
        knowledge_node = Node(
            name="knowledge",
            function=self._knowledge_process,
            description="知识沉淀和检索"
        )
        
        # 添加节点到图
        self.workflow_graph.add_nodes([
            recorder_node, analyst_node, summarizer_node, 
            task_node, knowledge_node
        ])
        
        # 定义边（工作流）
        # 记录员 -> 分析师
        self.workflow_graph.add_edge(Edge(
            from_node="recorder",
            to_node="analyst", 
            condition=lambda state: len(state.get("transcript", [])) > 0
        ))
        
        # 分析师 -> 总结者
        self.workflow_graph.add_edge(Edge(
            from_node="analyst",
            to_node="summarizer",
            condition=lambda state: state.get("analysis", {}) != {}
        ))
        
        # 分析师 -> 任务官  
        self.workflow_graph.add_edge(Edge(
            from_node="analyst", 
            to_node="task_agent",
            condition=lambda state: state.get("analysis", {}) != {}
        ))
        
        # 总结者/任务官 -> 知识管家
        self.workflow_graph.add_edge(Edge(
            from_node="summarizer",
            to_node="knowledge"
        ))
        
        self.workflow_graph.add_edge(Edge(
            from_node="task_agent",
            to_node="knowledge"
        ))
        
    async def start_meeting(self, meeting_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        启动会议流程
        
        Args:
            meeting_info: 会议信息（标题、参与者等）
            
        Returns:
            启动结果
        """
        try:
            logger.info(f"启动会议: {meeting_info.get('title', '未命名会议')}")
            
            # 初始化会议状态
            self.meeting_state.update({
                "status": "进行中",
                "meeting_info": meeting_info,
                "start_time": meeting_info.get("start_time"),
                "participants": meeting_info.get("participants", [])
            })
            
            # 启动记录员Agent开始转录
            from .recorder_agent import RecorderAgent
            self.agents["recorder"] = RecorderAgent(self.config.get("recorder", {}))
            
            # 启动实时转录
            await self.agents["recorder"].start_recording()
            
            return {
                "success": True,
                "message": "会议已启动",
                "meeting_id": meeting_info.get("meeting_id"),
                "status": self.meeting_state["status"]
            }
            
        except Exception as e:
            logger.error(f"启动会议失败: {str(e)}")
            return {
                "success": False,
                "message": f"启动失败: {str(e)}"
            }
    
    async def process_realtime_transcript(self, transcript_chunk: str, speaker: str = None) -> Dict[str, Any]:
        """
        处理实时转录内容
        
        Args:
            transcript_chunk: 转录文本片段
            speaker: 发言人
            
        Returns:
            处理结果
        """
        try:
            # 添加到转录记录
            self.meeting_state["transcript"].append({
                "speaker": speaker,
                "content": transcript_chunk,
                "timestamp": self._get_current_timestamp()
            })
            
            # 如果转录内容足够，触发分析流程
            if len(self.meeting_state["transcript"]) % 10 == 0:  # 每10条触发一次
                await self._trigger_analysis()
            
            return {
                "success": True,
                "transcript_length": len(self.meeting_state["transcript"])
            }
            
        except Exception as e:
            logger.error(f"处理实时转录失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def end_meeting(self) -> Dict[str, Any]:
        """
        结束会议并生成最终输出
        
        Returns:
            会议结束结果，包含完整的会议纪要、任务列表等
        """
        try:
            logger.info("开始结束会议流程")
            
            # 停止转录
            if "recorder" in self.agents:
                await self.agents["recorder"].stop_recording()
            
            # 最终分析和总结
            await self._trigger_final_analysis()
            
            # 更新状态
            self.meeting_state["status"] = "已结束"
            self.meeting_state["end_time"] = self._get_current_timestamp()
            
            # 生成最终报告
            final_report = await self._generate_final_report()
            
            return {
                "success": True,
                "meeting_report": final_report,
                "status": "已结束"
            }
            
        except Exception as e:
            logger.error(f"结束会议失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _recorder_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """记录员Agent处理流程"""
        # 实际的转录逻辑在recorder_agent中实现
        return state
    
    async def _analyst_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """分析师Agent处理流程"""
        if "analyst" not in self.agents:
            from .analyst_agent import AnalystAgent
            self.agents["analyst"] = AnalystAgent(self.config.get("analyst", {}))
        
        # 分析转录内容
        analysis_result = await self.agents["analyst"].analyze_transcript(
            state.get("transcript", [])
        )
        
        state["analysis"] = analysis_result
        return state
    
    async def _summarizer_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """总结者Agent处理流程"""
        if "summarizer" not in self.agents:
            from .summarizer_agent import SummarizerAgent
            self.agents["summarizer"] = SummarizerAgent(self.config.get("summarizer", {}))
        
        # 生成会议纪要
        summary_result = await self.agents["summarizer"].generate_summary(
            transcript=state.get("transcript", []),
            analysis=state.get("analysis", {})
        )
        
        state["summary"] = summary_result
        return state
    
    async def _task_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """任务官Agent处理流程"""
        if "task_agent" not in self.agents:
            from .task_agent import TaskAgent
            self.agents["task_agent"] = TaskAgent(self.config.get("task_agent", {}))
        
        # 提取任务
        tasks_result = await self.agents["task_agent"].extract_tasks(
            transcript=state.get("transcript", []),
            analysis=state.get("analysis", {})
        )
        
        state["tasks"] = tasks_result
        return state
    
    async def _knowledge_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """知识管家Agent处理流程"""
        if "knowledge" not in self.agents:
            from .knowledge_agent import KnowledgeAgent
            self.agents["knowledge"] = KnowledgeAgent(self.config.get("knowledge", {}))
        
        # 知识沉淀
        knowledge_result = await self.agents["knowledge"].store_knowledge(
            summary=state.get("summary", {}),
            tasks=state.get("tasks", []),
            meeting_info=state.get("meeting_info", {})
        )
        
        state["knowledge"] = knowledge_result
        return state
    
    async def _trigger_analysis(self):
        """触发分析流程"""
        try:
            # 执行分析师节点
            updated_state = await self._analyst_process(self.meeting_state)
            self.meeting_state.update(updated_state)
            
        except Exception as e:
            logger.error(f"触发分析失败: {str(e)}")
    
    async def _trigger_final_analysis(self):
        """触发最终分析"""
        try:
            # 按顺序执行所有节点
            state = self.meeting_state
            
            # 分析师
            state = await self._analyst_process(state)
            
            # 并行执行总结者和任务官
            summary_task = asyncio.create_task(self._summarizer_process(state.copy()))
            task_task = asyncio.create_task(self._task_process(state.copy()))
            
            summary_result, task_result = await asyncio.gather(summary_task, task_task)
            
            # 合并结果
            state.update(summary_result)
            state.update(task_result)
            
            # 知识管家
            state = await self._knowledge_process(state)
            
            self.meeting_state.update(state)
            
        except Exception as e:
            logger.error(f"最终分析失败: {str(e)}")
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终会议报告"""
        return {
            "meeting_info": self.meeting_state.get("meeting_info", {}),
            "duration": self._calculate_duration(),
            "participants": self.meeting_state.get("participants", []),
            "transcript_summary": len(self.meeting_state.get("transcript", [])),
            "analysis": self.meeting_state.get("analysis", {}),
            "summary": self.meeting_state.get("summary", {}),
            "tasks": self.meeting_state.get("tasks", []),
            "knowledge_entries": self.meeting_state.get("knowledge", {}),
            "generated_at": self._get_current_timestamp()
        }
    
    def _get_current_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _calculate_duration(self) -> str:
        """计算会议时长"""
        try:
            from datetime import datetime
            start = datetime.fromisoformat(self.meeting_state.get("start_time", ""))
            end = datetime.fromisoformat(self.meeting_state.get("end_time", ""))
            duration = end - start
            return str(duration)
        except:
            return "未知"
    
    def get_meeting_status(self) -> Dict[str, Any]:
        """获取会议状态"""
        return {
            "status": self.meeting_state["status"],
            "transcript_count": len(self.meeting_state.get("transcript", [])),
            "participants": len(self.meeting_state.get("participants", [])),
            "analysis_ready": bool(self.meeting_state.get("analysis")),
            "summary_ready": bool(self.meeting_state.get("summary")),
            "tasks_ready": bool(self.meeting_state.get("tasks"))
        }