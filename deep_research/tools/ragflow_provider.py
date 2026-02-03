"""
RAGFlow API Provider for Deep Research System.

This module provides integration with RAGFlow's retrieval API
for accessing private knowledge bases and document collections.

API文档参考: https://ragflow.io/docs/dev/http_api_reference

主要API端点:
- POST /api/v1/retrieval - 检索chunks
- POST /api/v1/chats/{chat_id}/completions - 对话补全
- GET /api/v1/datasets - 列出数据集
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from core.utils import get_logger, generate_id

logger = get_logger(__name__)


class RAGFlowResult(BaseModel):
    """A single RAGFlow retrieval result."""
    
    id: str = Field(default_factory=lambda: generate_id("rag"))
    
    # Content
    content: str = Field(..., description="Retrieved content/chunk")
    title: Optional[str] = Field(None, description="Document title")
    
    # Source information
    document_id: Optional[str] = Field(None, description="Source document ID")
    chunk_id: Optional[str] = Field(None, description="Chunk ID within document")
    dataset_id: Optional[str] = Field(None, description="Dataset/knowledge base ID")
    
    # Scores (RAGFlow returns similarity, vector_similarity, term_similarity)
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Combined similarity score")
    vector_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Vector cosine similarity")
    term_similarity: float = Field(default=0.0, ge=0.0, le=1.0, description="Term/keyword similarity")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    
    # Metadata
    url: Optional[str] = Field(None, description="Source URL if available")
    source: str = Field(default="ragflow", description="Source identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=datetime.now)


class RAGFlowProvider:
    """
    RAGFlow API client for knowledge base retrieval.
    
    Supports both the retrieval API and optional conversation/chat API.
    
    API参考: https://ragflow.io/docs/dev/http_api_reference
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RAGFlow provider.
        
        Args:
            config: Optional configuration override
        """
        self._config = config or get_config().get_ragflow_config()
        self.enabled = self._config.get("enabled", True)
        self.base_url = self._config.get("base_url", "http://localhost:9380").rstrip("/")
        self.api_key = self._config.get("api_key", "")
        
        # Dataset configuration
        self.default_dataset_id = self._config.get("dataset_id", "")
        self.datasets = self._config.get("datasets", [])
        
        # Retrieval parameters (参考RAGFlow API文档)
        retrieval_config = self._config.get("retrieval", {})
        self.top_k = retrieval_config.get("top_k", 1024)  # RAGFlow默认1024
        self.similarity_threshold = retrieval_config.get("similarity_threshold", 0.2)  # RAGFlow默认0.2
        self.vector_similarity_weight = retrieval_config.get("vector_similarity_weight", 0.3)  # RAGFlow默认0.3
        self.rerank_id = retrieval_config.get("rerank_id", "")  # Rerank模型ID，为空则不使用
        self.keyword = retrieval_config.get("keyword", True)  # 是否启用关键词匹配
        self.highlight = retrieval_config.get("highlight", False)  # 是否高亮匹配词
        
        # Chat configuration (optional)
        chat_config = self._config.get("chat", {})
        self.chat_enabled = chat_config.get("enabled", False)
        self.chat_id = chat_config.get("chat_id", "")  # Chat assistant ID
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def retrieve(
        self,
        query: str,
        dataset_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[RAGFlowResult]:
        """
        Retrieve relevant chunks from RAGFlow.
        
        API文档: POST /api/v1/retrieval
        
        Args:
            query: Search query (required)
            dataset_ids: List of dataset IDs to search
            document_ids: List of document IDs to search (可选，与dataset_ids二选一)
            top_k: Number of chunks for vector cosine computation (default 1024)
            similarity_threshold: Minimum similarity score (default 0.2)
            **kwargs: Additional parameters (keyword, highlight, rerank_id, etc.)
            
        Returns:
            List of RAGFlowResult objects
        """
        if not self.enabled:
            logger.warning("RAGFlow is disabled, returning empty results")
            return []
        
        # Determine datasets to search
        search_datasets = dataset_ids or self.datasets or [self.default_dataset_id]
        search_datasets = [d for d in search_datasets if d]  # Filter empty
        
        # 必须指定 dataset_ids 或 document_ids
        if not search_datasets and not document_ids:
            logger.warning("No datasets or documents configured for RAGFlow search")
            return []
        
        # 构建请求参数 (严格按照RAGFlow API文档)
        request_body = {
            "question": query,  # Required
            "top_k": top_k or self.top_k,
            "similarity_threshold": similarity_threshold or self.similarity_threshold,
            "vector_similarity_weight": kwargs.pop("vector_similarity_weight", self.vector_similarity_weight),
            "keyword": kwargs.pop("keyword", self.keyword),
            "highlight": kwargs.pop("highlight", self.highlight),
        }
        
        # 添加 dataset_ids 或 document_ids
        if search_datasets:
            request_body["dataset_ids"] = search_datasets
        if document_ids:
            request_body["document_ids"] = document_ids
        
        # 添加 rerank_id (如果配置了)
        rerank_id = kwargs.pop("rerank_id", self.rerank_id)
        if rerank_id:
            request_body["rerank_id"] = rerank_id
        
        # 合并其他参数
        request_body.update(kwargs)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/retrieval",
                    headers=self._get_headers(),
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "RAGFlow API error", 
                    status=e.response.status_code, 
                    detail=str(e),
                    response_text=e.response.text[:500] if e.response.text else ""
                )
                return []
            except httpx.RequestError as e:
                logger.error("RAGFlow connection error", error=str(e))
                return []
        
        # 检查响应状态
        if data.get("code", 0) != 0:
            logger.error("RAGFlow API returned error", code=data.get("code"), message=data.get("message"))
            return []
        
        results = []
        response_data = data.get("data", {})
        chunks = response_data.get("chunks", [])
        
        for chunk in chunks:
            # 解析chunk字段 (参考RAGFlow API文档)
            similarity = chunk.get("similarity", 0.0)
            vector_sim = chunk.get("vector_similarity", 0.0)
            term_sim = chunk.get("term_similarity", 0.0)
            
            results.append(RAGFlowResult(
                id=chunk.get("id", generate_id("rag")),
                content=chunk.get("content", ""),
                # document_keyword 是文档名 (RAGFlow API返回字段)
                title=chunk.get("document_keyword", chunk.get("document_name", "")),
                document_id=chunk.get("document_id"),
                chunk_id=chunk.get("id"),  # chunk的id字段
                dataset_id=chunk.get("kb_id", chunk.get("dataset_id")),  # kb_id 是dataset_id
                similarity_score=similarity,
                vector_similarity=vector_sim,
                term_similarity=term_sim,
                confidence_score=similarity,  # 使用similarity作为confidence
                url=chunk.get("url"),
                metadata={
                    "positions": chunk.get("positions", []),
                    "important_keywords": chunk.get("important_keywords", []),
                    "image_id": chunk.get("image_id", ""),
                    "highlight": chunk.get("highlight", ""),
                },
            ))
        
        # 记录文档聚合信息
        doc_aggs = response_data.get("doc_aggs", [])
        
        logger.info("RAGFlow retrieval completed", results_count=len(results))
        
        return results
    
    def retrieve_sync(
        self,
        query: str,
        dataset_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RAGFlowResult]:
        """Synchronous version of retrieve."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.retrieve(query, dataset_ids, top_k, **kwargs))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Use RAGFlow's chat completion API.
        
        API文档: POST /api/v1/chats/{chat_id}/completions
        
        Args:
            query: User message (question)
            session_id: Optional session ID for multi-turn conversation
            stream: Whether to stream the response (default False)
            **kwargs: Additional parameters
            
        Returns:
            Chat response dictionary with 'answer' and 'reference' fields
        """
        if not self.chat_enabled or not self.chat_id:
            logger.warning("RAGFlow chat is not configured (chat_id required)")
            return {"answer": "", "reference": {}}
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                # 正确的端点: /api/v1/chats/{chat_id}/completions
                request_body = {
                    "question": query,
                    "stream": stream,
                }
                
                # 添加session_id (如果提供)
                if session_id:
                    request_body["session_id"] = session_id
                
                # 合并其他参数
                request_body.update(kwargs)
                
                response = await client.post(
                    f"{self.base_url}/api/v1/chats/{self.chat_id}/completions",
                    headers=self._get_headers(),
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()
                
                # 检查响应状态
                if data.get("code", 0) != 0:
                    logger.error("RAGFlow chat API error", code=data.get("code"), message=data.get("message"))
                    return {"answer": "", "reference": {}, "error": data.get("message")}
                
                # 解析响应
                result_data = data.get("data", {})
                return {
                    "answer": result_data.get("answer", ""),
                    "reference": result_data.get("reference", {}),
                    "session_id": result_data.get("session_id"),
                    "audio_binary": result_data.get("audio_binary"),
                }
                
            except httpx.HTTPStatusError as e:
                logger.error("RAGFlow chat API error", status=e.response.status_code)
                return {"answer": "", "reference": {}, "error": str(e)}
            except httpx.RequestError as e:
                logger.error("RAGFlow chat connection error", error=str(e))
                return {"answer": "", "reference": {}, "error": str(e)}
    
    async def create_chat_session(self, name: str = "new session") -> Optional[str]:
        """
        Create a new chat session.
        
        API文档: POST /api/v1/chats/{chat_id}/sessions
        
        Args:
            name: Session name
            
        Returns:
            Session ID if successful, None otherwise
        """
        if not self.chat_id:
            logger.warning("chat_id not configured")
            return None
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/v1/chats/{self.chat_id}/sessions",
                    headers=self._get_headers(),
                    json={"name": name},
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("code", 0) == 0:
                    return data.get("data", {}).get("id")
                return None
            except Exception as e:
                logger.error("Failed to create chat session", error=str(e))
                return None
    
    def chat_sync(
        self,
        query: str,
        session_id: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Synchronous version of chat."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.chat(query, session_id, stream, **kwargs))
    
    async def list_datasets(self) -> List[Dict[str, Any]]:
        """List available datasets/knowledge bases."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/api/v1/datasets",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
            except Exception as e:
                logger.error("Failed to list RAGFlow datasets", error=str(e))
                return []
    
    async def health_check(self) -> bool:
        """Check if RAGFlow API is accessible."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/health",
                    headers=self._get_headers(),
                )
                return response.status_code == 200
        except Exception:
            return False


class MockRAGFlowProvider(RAGFlowProvider):
    """Mock RAGFlow provider for testing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.enabled = True  # Always enabled for mock
    
    async def retrieve(
        self,
        query: str,
        dataset_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs
    ) -> List[RAGFlowResult]:
        """Return mock RAGFlow results."""
        logger.debug("Mock RAGFlow retrieval executed")
        
        return [
            RAGFlowResult(
                id="mock_chunk_001",
                content=f"Mock RAG content for query: {query}. "
                        "This simulates retrieved content from a private knowledge base.",
                title="Mock Document 1",
                document_id="mock_doc_1",
                chunk_id="mock_chunk_001",
                dataset_id="mock_dataset",
                similarity_score=0.92,
                vector_similarity=0.88,
                term_similarity=0.95,
                confidence_score=0.92,
            ),
            RAGFlowResult(
                id="mock_chunk_002",
                content=f"Additional RAG content related to: {query}. "
                        "Contains supplementary information from internal documents.",
                title="Mock Document 2",
                document_id="mock_doc_2",
                chunk_id="mock_chunk_002",
                dataset_id="mock_dataset",
                similarity_score=0.85,
                vector_similarity=0.82,
                term_similarity=0.88,
                confidence_score=0.85,
            ),
            RAGFlowResult(
                id="mock_chunk_003",
                content=f"Third RAG result for: {query}. "
                        "Provides expert-level details from the knowledge base.",
                title="Mock Document 3",
                document_id="mock_doc_3",
                chunk_id="mock_chunk_003",
                dataset_id="mock_dataset",
                similarity_score=0.78,
                vector_similarity=0.75,
                term_similarity=0.80,
                confidence_score=0.78,
            ),
        ]
    
    async def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Return mock chat response."""
        return {
            "answer": f"Mock RAGFlow answer for: {query}",
            "reference": {
                "chunks": {
                    "1": {"content": "Reference content 1", "document_name": "Mock Doc 1"},
                    "2": {"content": "Reference content 2", "document_name": "Mock Doc 2"},
                },
                "doc_aggs": {
                    "Mock Doc 1": {"count": 1},
                    "Mock Doc 2": {"count": 1},
                }
            },
            "session_id": session_id or "mock_session_001",
        }


# Global provider instance
_ragflow_provider: Optional[RAGFlowProvider] = None


def get_ragflow_provider(use_mock: bool = False) -> RAGFlowProvider:
    """
    Get the global RAGFlow provider instance.
    
    Args:
        use_mock: Whether to use mock provider
        
    Returns:
        RAGFlowProvider instance
    """
    global _ragflow_provider
    if _ragflow_provider is None or (use_mock and not isinstance(_ragflow_provider, MockRAGFlowProvider)):
        if use_mock:
            _ragflow_provider = MockRAGFlowProvider()
        else:
            _ragflow_provider = RAGFlowProvider()
    return _ragflow_provider
