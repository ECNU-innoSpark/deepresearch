"""
Mock workflow tests for Deep Research System.

Tests the complete workflow using mock LLM and tools.
Run with: pytest tests/test_workflow_mock.py -v
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockLLMResponse:
    """Mock LLM response."""
    
    def __init__(self, content: str):
        self.content = content


def create_mock_decompose_response():
    """Create mock decomposition response."""
    return json.dumps({
        "sub_tasks": [
            {
                "id": "task_1",
                "question": "什么是人工智能的基本概念?",
                "description": "了解AI的定义和基础",
                "keywords": ["AI", "人工智能", "定义"],
                "preferred_sources": ["search", "rag"]
            },
            {
                "id": "task_2",
                "question": "人工智能有哪些主要应用领域?",
                "description": "探索AI的实际应用",
                "keywords": ["AI应用", "机器学习", "深度学习"],
                "preferred_sources": ["search"]
            },
            {
                "id": "task_3",
                "question": "人工智能的未来发展趋势是什么?",
                "description": "分析AI的发展方向",
                "keywords": ["AI趋势", "未来", "发展"],
                "preferred_sources": ["search", "rag"]
            }
        ]
    })


def create_mock_plan_response():
    """Create mock planning response."""
    return json.dumps({
        "task_plan": ["task_1", "task_2", "task_3"],
        "dependencies": {
            "task_2": ["task_1"],
            "task_3": ["task_1", "task_2"]
        },
        "parallel_groups": [],
        "reasoning": "首先了解基本概念，然后探索应用，最后分析趋势"
    })


def create_mock_execution_response():
    """Create mock execution response."""
    return json.dumps({
        "tool_calls": [
            {
                "tool": "web_search",
                "query": "人工智能基本概念定义",
                "reason": "搜索AI的基础知识"
            },
            {
                "tool": "ragflow_search",
                "query": "人工智能概述",
                "reason": "从知识库获取专业资料"
            }
        ],
        "search_strategy": "同时使用网络搜索和知识库检索"
    })


def create_mock_selection_response():
    """Create mock selection response."""
    return json.dumps({
        "selections": [
            {
                "data_id": "raw_1",
                "decision": "keep",
                "relevance_score": 0.9,
                "confidence_score": 0.85,
                "reason": "高度相关的权威来源"
            },
            {
                "data_id": "raw_2",
                "decision": "keep",
                "relevance_score": 0.85,
                "confidence_score": 0.9,
                "reason": "知识库的专业内容"
            }
        ],
        "summary": {
            "total_reviewed": 2,
            "kept": 2,
            "discarded": 0
        }
    })


def create_mock_writing_response():
    """Create mock writing response."""
    return """# 人工智能研究报告

## 摘要

本报告探讨了人工智能的基本概念、应用领域和未来发展趋势。

## 1. 人工智能的基本概念

人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统[^ref_1]。

## 2. 主要应用领域

AI已广泛应用于医疗、金融、教育等领域[^ref_2]。

## 3. 未来发展趋势

专家预测AI将继续快速发展，特别是在自然语言处理和计算机视觉领域[^ref_1][^ref_2]。

## 结论

人工智能正在深刻改变我们的生活和工作方式。
"""


def create_mock_review_response_approved():
    """Create mock approved review response."""
    return json.dumps({
        "is_approved": True,
        "checks": {
            "format_check": True,
            "citation_check": True,
            "content_accuracy": True,
            "citation_count_check": True
        },
        "metrics": {
            "total_citations": 2,
            "search_citations": 1,
            "rag_citations": 1,
            "word_count": 200
        },
        "issues": [],
        "suggestions": [],
        "route_to": "end",
        "detailed_feedback": "报告质量良好，已通过审核。"
    })


class TestMockWorkflow:
    """Test workflow with mocked components."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset citation manager before each test."""
        from core.citation_manager import CitationManager
        CitationManager.reset()
    
    def test_decompose_node_mock(self):
        """Test decompose node with mocked LLM."""
        from core.state import create_initial_state
        from agents.decompose import decompose_node
        
        # Create initial state
        state = create_initial_state("人工智能的发展")
        
        # Mock the LLM client
        with patch('agents.decompose.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.invoke_with_prompt.return_value = create_mock_decompose_response()
            mock_get_client.return_value = mock_client
            
            # Run the node
            result = decompose_node(state)
            
            # Verify results
            assert "sub_tasks" in result
            assert len(result["sub_tasks"]) == 3
            assert result["sub_tasks"][0].id == "task_1"
            assert result["workflow_status"] == "decomposed"
    
    def test_plan_node_mock(self):
        """Test plan node with mocked LLM."""
        from core.state import create_initial_state, SubTask, DataSourceType
        from agents.plan import plan_node
        
        # Create state with sub-tasks
        state = create_initial_state("人工智能的发展")
        state.sub_tasks = [
            SubTask(id="task_1", question="Q1", preferred_sources=[DataSourceType.SEARCH]),
            SubTask(id="task_2", question="Q2", preferred_sources=[DataSourceType.SEARCH]),
            SubTask(id="task_3", question="Q3", preferred_sources=[DataSourceType.SEARCH]),
        ]
        
        with patch('agents.plan.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.invoke_with_prompt.return_value = create_mock_plan_response()
            mock_get_client.return_value = mock_client
            
            result = plan_node(state)
            
            assert "task_plan" in result
            assert len(result["task_plan"]) == 3
            assert result["task_plan"][0] == "task_1"
            assert result["workflow_status"] == "planned"
    
    def test_selection_node_mock(self):
        """Test selection node with mocked LLM."""
        from core.state import create_initial_state, RawDataItem, DataSourceType
        from agents.selection import selection_node
        
        state = create_initial_state("人工智能的发展")
        state.raw_data = [
            RawDataItem(
                id="raw_1",
                task_id="task_1",
                source_type=DataSourceType.SEARCH,
                content="AI是人工智能的缩写...",
                title="AI基础",
                url="https://example.com/ai",
                relevance_score=0.8,
                confidence_score=0.7,
            ),
            RawDataItem(
                id="raw_2",
                task_id="task_1",
                source_type=DataSourceType.RAG,
                content="人工智能的定义是...",
                title="AI定义",
                url="https://rag.example.com/ai",
                relevance_score=0.9,
                confidence_score=0.85,
            ),
        ]
        
        with patch('agents.selection.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.invoke_with_prompt.return_value = create_mock_selection_response()
            mock_get_client.return_value = mock_client
            
            result = selection_node(state)
            
            assert "selected_data" in result
            assert len(result["selected_data"]) > 0
            assert result["workflow_status"] == "selected"
    
    def test_writing_node_mock(self):
        """Test writing node with mocked LLM."""
        from core.state import create_initial_state, SelectedDataItem, DataSourceType
        from agents.writing import writing_node
        from core.citation_manager import CitationManager
        
        CitationManager.reset()
        manager = CitationManager.get_instance()
        
        # Add citations first
        c1 = manager.add_citation(title="Source 1", url="https://example.com/1")
        c2 = manager.add_citation(title="Source 2", url="https://example.com/2")
        
        state = create_initial_state("人工智能的发展")
        state.selected_data = [
            SelectedDataItem(
                raw_data_id="raw_1",
                task_id="task_1",
                source_type=DataSourceType.SEARCH,
                content="AI内容1",
                title="Source 1",
                url="https://example.com/1",
                citation_id=c1.id,
                citation_key=c1.get_citation_key(),
                relevance_score=0.9,
                confidence_score=0.85,
            ),
            SelectedDataItem(
                raw_data_id="raw_2",
                task_id="task_1",
                source_type=DataSourceType.RAG,
                content="AI内容2",
                title="Source 2",
                url="https://example.com/2",
                citation_id=c2.id,
                citation_key=c2.get_citation_key(),
                relevance_score=0.85,
                confidence_score=0.9,
            ),
        ]
        
        with patch('agents.writing.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.invoke_with_prompt.return_value = create_mock_writing_response()
            mock_get_client.return_value = mock_client
            
            result = writing_node(state)
            
            assert "draft_report" in result
            assert "人工智能" in result["draft_report"]
            assert result["workflow_status"] == "written"
    
    def test_review_node_approved_mock(self):
        """Test review node when report is approved."""
        from core.state import create_initial_state
        from agents.review import review_node
        
        state = create_initial_state("人工智能的发展")
        state.draft_report = create_mock_writing_response()
        state.selected_data = []
        
        with patch('agents.review.get_llm_client') as mock_get_client:
            mock_client = MagicMock()
            mock_client.invoke_with_prompt.return_value = create_mock_review_response_approved()
            mock_get_client.return_value = mock_client
            
            result = review_node(state)
            
            assert "review_feedback" in result
            assert result["review_feedback"].is_approved == True
            assert result["workflow_status"] == "completed"
            assert "final_report" in result


class TestReviewRouter:
    """Test review routing logic."""
    
    def test_router_approved(self):
        """Test routing when report is approved."""
        from core.state import create_initial_state, ReviewFeedback
        from agents.review import review_router
        
        state = create_initial_state("test")
        state.review_feedback = ReviewFeedback(
            is_approved=True,
            route_to="end"
        )
        
        result = review_router(state)
        assert result == "end"
    
    def test_router_need_more_data(self):
        """Test routing when more data is needed."""
        from core.state import create_initial_state, ReviewFeedback
        from agents.review import review_router
        
        state = create_initial_state("test")
        state.review_feedback = ReviewFeedback(
            is_approved=False,
            route_to="plan"
        )
        
        result = review_router(state)
        assert result == "plan"
    
    def test_router_rewrite(self):
        """Test routing when rewrite is needed."""
        from core.state import create_initial_state, ReviewFeedback
        from agents.review import review_router
        
        state = create_initial_state("test")
        state.review_feedback = ReviewFeedback(
            is_approved=False,
            route_to="writing"
        )
        
        result = review_router(state)
        assert result == "writing"


class TestExecutionContinuation:
    """Test execution continuation logic."""
    
    def test_should_continue_execution_yes(self):
        """Test that execution continues when tasks remain."""
        from core.state import create_initial_state
        from agents.execution import should_continue_execution
        
        state = create_initial_state("test")
        state.task_plan = ["task_1", "task_2", "task_3"]
        state.current_task_index = 1
        
        result = should_continue_execution(state)
        assert result == "continue"
    
    def test_should_continue_execution_no(self):
        """Test that execution stops when all tasks done."""
        from core.state import create_initial_state
        from agents.execution import should_continue_execution
        
        state = create_initial_state("test")
        state.task_plan = ["task_1", "task_2"]
        state.current_task_index = 2
        
        result = should_continue_execution(state)
        assert result == "done"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
