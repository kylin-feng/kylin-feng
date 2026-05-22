"""
之江智会 - 多智能体协作系统

6个专业智能体协作架构：
1. 记录员Agent - 实时转录
2. 分析师Agent - 会议分析  
3. 总结者Agent - 纪要生成
4. 任务官Agent - 任务提取
5. 知识管家Agent - 知识管理
6. 协调员Agent - 工作流编排
"""

from .recorder_agent import RecorderAgent
from .analyst_agent import AnalystAgent
from .summarizer_agent import SummarizerAgent
from .task_agent import TaskAgent
from .knowledge_agent import KnowledgeAgent
from .coordinator_agent import CoordinatorAgent

__all__ = [
    'RecorderAgent',
    'AnalystAgent', 
    'SummarizerAgent',
    'TaskAgent',
    'KnowledgeAgent',
    'CoordinatorAgent'
]