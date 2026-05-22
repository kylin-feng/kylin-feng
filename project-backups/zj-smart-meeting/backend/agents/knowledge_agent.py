"""
知识管家Agent - Qdrant向量检索
负责知识沉淀、向量检索、智能问答、跨会议关联
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from loguru import logger
import dashscope
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import hashlib

class KnowledgeAgent:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化知识管家Agent
        
        Args:
            config: 配置信息，包含Qdrant和通义千问配置
        """
        self.config = config
        self.api_key = config.get("dashscope_api_key", "")
        self.qdrant_url = config.get("qdrant_url", "localhost")
        self.qdrant_port = config.get("qdrant_port", 6333)
        self.collection_name = config.get("collection_name", "meeting_knowledge")
        
        # 向量维度（通义千问embedding维度）
        self.vector_size = config.get("vector_size", 1536)
        
        # 初始化Qdrant客户端
        self.qdrant_client = None
        self._setup_qdrant()
        
        # 知识存储
        self.knowledge_base = {}
        self.meeting_relations = {}
        
        self._setup_dashscope()
    
    def _setup_dashscope(self):
        """设置通义千问API"""
        if self.api_key:
            dashscope.api_key = self.api_key
            logger.info("知识管家Agent - 通义千问API配置完成")
        else:
            logger.warning("未配置通义千问API密钥，使用模拟向量化")
    
    def _setup_qdrant(self):
        """初始化Qdrant向量数据库"""
        try:
            self.qdrant_client = QdrantClient(
                host=self.qdrant_url,
                port=self.qdrant_port
            )
            
            # 检查集合是否存在，不存在则创建
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建Qdrant集合: {self.collection_name}")
            else:
                logger.info(f"Qdrant集合已存在: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"初始化Qdrant失败: {str(e)}")
            logger.warning("将使用内存存储作为降级方案")
            self.qdrant_client = None
    
    async def store_knowledge(self,
                             summary: Dict[str, Any] = None,
                             tasks: Dict[str, Any] = None,
                             meeting_info: Dict[str, Any] = None,
                             transcript: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        存储会议知识到向量数据库
        
        Args:
            summary: 会议纪要
            tasks: 任务列表
            meeting_info: 会议信息
            transcript: 转录记录
            
        Returns:
            存储结果
        """
        try:
            start_time = datetime.now()
            logger.info("开始知识沉淀")
            
            meeting_id = meeting_info.get("meeting_id") if meeting_info else self._generate_meeting_id()
            
            # 提取知识片段
            knowledge_chunks = await self._extract_knowledge_chunks(
                summary, tasks, meeting_info, transcript
            )
            
            # 向量化并存储
            stored_points = []
            for chunk in knowledge_chunks:
                point = await self._store_chunk(chunk, meeting_id)
                if point:
                    stored_points.append(point)
            
            # 建立会议关联
            relations = await self._build_meeting_relations(meeting_id, knowledge_chunks)
            
            # 更新知识图谱
            await self._update_knowledge_graph(meeting_id, knowledge_chunks, relations)
            
            storage_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "meeting_id": meeting_id,
                "timestamp": datetime.now().isoformat(),
                "storage_time": round(storage_time, 2),
                "chunks_stored": len(stored_points),
                "knowledge_chunks": len(knowledge_chunks),
                "relations_created": len(relations),
                "metadata": {
                    "collection": self.collection_name,
                    "vector_size": self.vector_size
                }
            }
            
            logger.info(f"知识沉淀完成，存储 {len(stored_points)} 个知识片段")
            
            return result
            
        except Exception as e:
            logger.error(f"知识沉淀失败: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _extract_knowledge_chunks(self,
                                       summary: Dict[str, Any],
                                       tasks: Dict[str, Any],
                                       meeting_info: Dict[str, Any],
                                       transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从会议内容中提取知识片段
        
        Args:
            summary: 会议纪要
            tasks: 任务列表
            meeting_info: 会议信息
            transcript: 转录记录
            
        Returns:
            知识片段列表
        """
        chunks = []
        
        # 从纪要中提取知识
        if summary:
            summary_data = summary.get("summaries", {})
            for summary_type, summary_content in summary_data.items():
                if isinstance(summary_content, dict):
                    content = summary_content.get("content", "")
                else:
                    content = str(summary_content)
                
                if content:
                    chunks.append({
                        "type": "summary",
                        "subtype": summary_type,
                        "content": content,
                        "source": "meeting_summary",
                        "metadata": {
                            "meeting_title": meeting_info.get("title", "") if meeting_info else "",
                            "generation_time": summary.get("generation_time", 0)
                        }
                    })
        
        # 从任务中提取知识
        if tasks:
            task_list = tasks.get("tasks", [])
            for task in task_list:
                task_description = task.get("description", "") or task.get("enhanced_description", "")
                if task_description:
                    chunks.append({
                        "type": "task",
                        "content": task_description,
                        "source": "task_extraction",
                        "metadata": {
                            "task_id": task.get("id", ""),
                            "assignee": task.get("assignee", ""),
                            "priority": task.get("priority", ""),
                            "deadline": task.get("deadline", "")
                        }
                    })
        
        # 从转录中提取关键片段
        if transcript:
            key_segments = self._extract_key_segments(transcript)
            for segment in key_segments:
                chunks.append({
                    "type": "transcript",
                    "content": segment.get("content", ""),
                    "source": "transcript",
                    "metadata": {
                        "speaker": segment.get("speaker", ""),
                        "timestamp": segment.get("timestamp")
                    }
                })
        
        # 从会议信息中提取元数据
        if meeting_info:
            chunks.append({
                "type": "metadata",
                "content": json.dumps(meeting_info, ensure_ascii=False),
                "source": "meeting_info",
                "metadata": meeting_info
            })
        
        return chunks
    
    def _extract_key_segments(self, transcript: List[Dict[str, Any]], max_segments: int = 10) -> List[Dict[str, Any]]:
        """从转录中提取关键片段"""
        key_segments = []
        
        # 识别关键时刻
        key_indicators = ["决定", "确定", "同意", "任务", "负责", "问题", "建议", "重要"]
        
        for record in transcript:
            content = record.get("content", "")
            if any(indicator in content for indicator in key_indicators):
                if len(content) > 20:  # 过滤太短的内容
                    key_segments.append(record)
        
        # 如果关键片段不够，选择最长的片段
        if len(key_segments) < max_segments:
            sorted_transcript = sorted(transcript, key=lambda x: len(x.get("content", "")), reverse=True)
            for record in sorted_transcript:
                if record not in key_segments and len(record.get("content", "")) > 30:
                    key_segments.append(record)
                if len(key_segments) >= max_segments:
                    break
        
        return key_segments[:max_segments]
    
    async def _store_chunk(self, chunk: Dict[str, Any], meeting_id: str) -> Optional[PointStruct]:
        """
        存储单个知识片段到向量数据库
        
        Args:
            chunk: 知识片段
            meeting_id: 会议ID
            
        Returns:
            存储的点结构
        """
        try:
            if not self.qdrant_client:
                # 降级到内存存储
                chunk_id = self._generate_chunk_id(chunk)
                self.knowledge_base[chunk_id] = chunk
                return None
            
            # 生成向量
            vector = await self._generate_embedding(chunk.get("content", ""))
            if vector is None:
                return None
            
            # 生成唯一ID
            chunk_id = self._generate_chunk_id(chunk)
            
            # 构建点结构
            point = PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "content": chunk.get("content", ""),
                    "type": chunk.get("type", ""),
                    "source": chunk.get("source", ""),
                    "meeting_id": meeting_id,
                    "metadata": chunk.get("metadata", {}),
                    "created_at": datetime.now().isoformat()
                }
            )
            
            # 存储到Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return point
            
        except Exception as e:
            logger.error(f"存储知识片段失败: {str(e)}")
            return None
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        使用通义千问生成文本向量
        
        Args:
            text: 文本内容
            
        Returns:
            向量列表
        """
        try:
            if not self.api_key or not text:
                # 返回模拟向量
                return [0.0] * self.vector_size
            
            # 调用通义千问embedding API
            # 这里使用模拟实现，实际项目中需要调用真实API
            # response = dashscope.TextEmbedding.call(
            #     model=dashscope.TextEmbedding.Models.text_embedding_v1,
            #     input=text
            # )
            
            # 模拟向量生成（实际应使用真实API）
            text_hash = hash(text)
            np.random.seed(text_hash % (2**32))
            vector = np.random.normal(0, 1, self.vector_size).tolist()
            # 归一化
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = (np.array(vector) / norm).tolist()
            
            return vector
            
        except Exception as e:
            logger.error(f"生成向量失败: {str(e)}")
            return None
    
    async def search_knowledge(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        搜索相关知识
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filters: 过滤条件
            
        Returns:
            搜索结果
        """
        try:
            logger.info(f"搜索知识: {query}")
            
            # 生成查询向量
            query_vector = await self._generate_embedding(query)
            if query_vector is None:
                return {"results": [], "error": "向量生成失败"}
            
            if not self.qdrant_client:
                # 降级到内存搜索
                return self._search_in_memory(query, top_k, filters)
            
            # 构建查询
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=None  # 可以添加过滤条件
            )
            
            # 格式化结果
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "content": point.payload.get("content", ""),
                    "type": point.payload.get("type", ""),
                    "source": point.payload.get("source", ""),
                    "meeting_id": point.payload.get("meeting_id", ""),
                    "metadata": point.payload.get("metadata", {})
                })
            
            return {
                "query": query,
                "total_results": len(results),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"知识搜索失败: {str(e)}")
            return {"results": [], "error": str(e)}
    
    def _search_in_memory(self, query: str, top_k: int, filters: Dict[str, Any]) -> Dict[str, Any]:
        """内存搜索（降级方案）"""
        results = []
        
        # 简单的关键词匹配
        query_lower = query.lower()
        for chunk_id, chunk in self.knowledge_base.items():
            content = chunk.get("content", "").lower()
            if query_lower in content:
                results.append({
                    "id": chunk_id,
                    "score": 0.8,  # 模拟相似度
                    "content": chunk.get("content", ""),
                    "type": chunk.get("type", ""),
                    "source": chunk.get("source", ""),
                    "metadata": chunk.get("metadata", {})
                })
        
        # 按相关性排序（简化）
        results.sort(key=lambda x: len(x["content"]), reverse=True)
        
        return {
            "query": query,
            "total_results": len(results),
            "results": results[:top_k],
            "timestamp": datetime.now().isoformat()
        }
    
    async def intelligent_qa(self, question: str, context_meeting_ids: List[str] = None) -> Dict[str, Any]:
        """
        智能问答
        
        Args:
            question: 问题
            context_meeting_ids: 上下文会议ID列表
            
        Returns:
            问答结果
        """
        try:
            logger.info(f"智能问答: {question}")
            
            # 搜索相关知识
            search_results = await self.search_knowledge(question, top_k=5)
            
            # 构建上下文
            context = self._build_qa_context(search_results.get("results", []))
            
            # 使用通义千问生成答案
            answer = await self._generate_answer_with_qwen(question, context)
            
            # 查找相关会议
            related_meetings = await self._find_related_meetings(question, context_meeting_ids)
            
            return {
                "question": question,
                "answer": answer,
                "context": context,
                "related_meetings": related_meetings,
                "sources": search_results.get("results", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"智能问答失败: {str(e)}")
            return {"question": question, "answer": "抱歉，无法回答此问题", "error": str(e)}
    
    def _build_qa_context(self, search_results: List[Dict[str, Any]]) -> str:
        """构建问答上下文"""
        context_parts = []
        
        for i, result in enumerate(search_results[:3], 1):
            content = result.get("content", "")
            source = result.get("source", "")
            context_parts.append(f"[来源{i}: {source}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_answer_with_qwen(self, question: str, context: str) -> str:
        """使用通义千问生成答案"""
        try:
            if not self.api_key:
                # 模拟答案生成
                return f"基于上下文信息，关于'{question}'的回答：请参考相关会议记录。"
            
            # 构建提示词
            prompt = f"""
            基于以下会议知识库内容，回答用户问题：
            
            上下文：
            {context}
            
            问题：{question}
            
            请提供准确、简洁的回答：
            """
            
            # 调用通义千问API（这里使用模拟）
            # response = dashscope.Generation.call(
            #     model="qwen-plus",
            #     prompt=prompt
            # )
            
            # 模拟答案
            answer = f"根据会议记录，关于'{question}'：\n"
            if "任务" in question:
                answer += "相关任务信息已在会议中讨论，请查看任务列表获取详细信息。"
            elif "决策" in question:
                answer += "相关决策已在会议中确定，请查看会议纪要获取详细信息。"
            else:
                answer += "相关信息请参考上述上下文内容。"
            
            return answer
            
        except Exception as e:
            logger.error(f"生成答案失败: {str(e)}")
            return "无法生成答案，请稍后重试"
    
    async def _find_related_meetings(self, question: str, context_meeting_ids: List[str] = None) -> List[Dict[str, Any]]:
        """查找相关会议"""
        related_meetings = []
        
        # 搜索相关会议
        search_results = await self.search_knowledge(question, top_k=10)
        
        meeting_ids = set()
        for result in search_results.get("results", []):
            meeting_id = result.get("meeting_id")
            if meeting_id and meeting_id not in meeting_ids:
                meeting_ids.add(meeting_id)
                related_meetings.append({
                    "meeting_id": meeting_id,
                    "relevance_score": result.get("score", 0),
                    "excerpt": result.get("content", "")[:100]
                })
        
        # 按相关性排序
        related_meetings.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return related_meetings[:5]
    
    async def _build_meeting_relations(self, meeting_id: str, knowledge_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """建立会议关联关系"""
        relations = []
        
        # 提取关键词和主题
        topics = set()
        for chunk in knowledge_chunks:
            content = chunk.get("content", "")
            # 简单的主题提取
            topic_keywords = ["产品", "用户", "功能", "设计", "开发", "测试", "运营"]
            for keyword in topic_keywords:
                if keyword in content:
                    topics.add(keyword)
        
        # 查找相关会议
        for topic in topics:
            related_meetings = await self.search_knowledge(topic, top_k=3)
            for result in related_meetings.get("results", []):
                related_meeting_id = result.get("meeting_id")
                if related_meeting_id and related_meeting_id != meeting_id:
                    relations.append({
                        "from_meeting": meeting_id,
                        "to_meeting": related_meeting_id,
                        "relation_type": "topic_similarity",
                        "topic": topic,
                        "strength": result.get("score", 0)
                    })
        
        return relations
    
    async def _update_knowledge_graph(self, meeting_id: str, knowledge_chunks: List[Dict[str, Any]], relations: List[Dict[str, Any]]):
        """更新知识图谱"""
        if meeting_id not in self.meeting_relations:
            self.meeting_relations[meeting_id] = {
                "chunks": len(knowledge_chunks),
                "relations": relations,
                "topics": set(),
                "created_at": datetime.now().isoformat()
            }
        
        # 更新主题
        for chunk in knowledge_chunks:
            content = chunk.get("content", "")
            topic_keywords = ["产品", "用户", "功能", "设计", "开发", "测试", "运营", "数据"]
            for keyword in topic_keywords:
                if keyword in content:
                    self.meeting_relations[meeting_id]["topics"].add(keyword)
    
    def _generate_meeting_id(self) -> str:
        """生成会议ID"""
        return f"MEETING_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def _generate_chunk_id(self, chunk: Dict[str, Any]) -> int:
        """生成知识片段ID"""
        content = chunk.get("content", "")
        chunk_type = chunk.get("type", "")
        content_hash = hashlib.md5(f"{content}_{chunk_type}".encode()).hexdigest()
        return int(content_hash[:8], 16) % (2**63 - 1)
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识库统计"""
        try:
            if self.qdrant_client:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                total_points = collection_info.points_count
            else:
                total_points = len(self.knowledge_base)
            
            return {
                "total_knowledge_chunks": total_points,
                "total_meetings": len(self.meeting_relations),
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "storage_type": "qdrant" if self.qdrant_client else "memory"
            }
            
        except Exception as e:
            logger.error(f"获取知识库统计失败: {str(e)}")
            return {"error": str(e)}
    
    async def get_meeting_knowledge_graph(self, meeting_id: str) -> Dict[str, Any]:
        """获取会议知识图谱"""
        if meeting_id not in self.meeting_relations:
            return {"meeting_id": meeting_id, "error": "会议不存在"}
        
        relations = self.meeting_relations[meeting_id]
        
        return {
            "meeting_id": meeting_id,
            "knowledge_chunks": relations.get("chunks", 0),
            "topics": list(relations.get("topics", set())),
            "related_meetings": relations.get("relations", []),
            "created_at": relations.get("created_at", "")
        }


