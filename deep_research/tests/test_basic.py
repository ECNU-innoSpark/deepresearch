"""
Basic tests for Deep Research System components.

Run with: pytest tests/test_basic.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Test configuration loading."""
    
    def test_config_loader_singleton(self):
        """Test that ConfigLoader is a singleton."""
        from config import ConfigLoader, get_config
        
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_settings_loaded(self):
        """Test that settings.yaml is loaded."""
        from config import get_config
        
        config = get_config()
        assert config.settings is not None
        assert "llm" in config.settings
        assert "workflow" in config.settings
    
    def test_tools_config_loaded(self):
        """Test that tools_config.yaml is loaded."""
        from config import get_config
        
        config = get_config()
        assert config.tools is not None
        assert "search" in config.tools
        assert "ragflow" in config.tools
    
    def test_get_llm_config(self):
        """Test LLM config retrieval."""
        from config import get_config
        
        config = get_config()
        
        # Default config
        default_config = config.get_llm_config()
        assert "model" in default_config
        
        # Agent-specific config
        decompose_config = config.get_llm_config("decompose")
        assert "model" in decompose_config


class TestState:
    """Test state definitions."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        from core.state import create_initial_state, GraphState
        
        query = "测试研究问题"
        state = create_initial_state(query)
        
        assert isinstance(state, GraphState)
        assert state.original_query == query
        assert state.workflow_status == "initialized"
        assert state.sub_tasks == []
        assert state.raw_data == []
        assert state.selected_data == []
    
    def test_sub_task_creation(self):
        """Test SubTask model."""
        from core.state import SubTask, TaskStatus, DataSourceType
        
        task = SubTask(
            id="task_1",
            question="什么是人工智能?",
            description="了解AI的基本概念",
            keywords=["AI", "人工智能", "机器学习"],
            preferred_sources=[DataSourceType.SEARCH],
        )
        
        assert task.id == "task_1"
        assert task.status == TaskStatus.PENDING
        assert len(task.keywords) == 3
    
    def test_raw_data_item(self):
        """Test RawDataItem model."""
        from core.state import RawDataItem, DataSourceType
        
        item = RawDataItem(
            id="raw_1",
            task_id="task_1",
            source_type=DataSourceType.SEARCH,
            content="测试内容",
            title="测试标题",
            url="https://example.com",
            relevance_score=0.8,
            confidence_score=0.7,
        )
        
        assert item.source_type == DataSourceType.SEARCH
        assert 0 <= item.relevance_score <= 1


class TestCitationManager:
    """Test citation management."""
    
    def test_citation_manager_singleton(self):
        """Test CitationManager singleton pattern."""
        from core.citation_manager import CitationManager
        
        CitationManager.reset()
        
        manager1 = CitationManager.get_instance()
        manager2 = CitationManager.get_instance()
        assert manager1 is manager2
    
    def test_add_citation(self):
        """Test adding citations."""
        from core.citation_manager import CitationManager
        
        CitationManager.reset()
        manager = CitationManager.get_instance()
        
        citation = manager.add_citation(
            title="Test Article",
            url="https://example.com/article",
            content_snippet="This is a test snippet.",
            source_type="search",
        )
        
        assert citation.id == "ref_1"
        assert citation.numeric_id == 1
        assert citation.title == "Test Article"
    
    def test_citation_deduplication(self):
        """Test that duplicate URLs are not added."""
        from core.citation_manager import CitationManager
        
        CitationManager.reset()
        manager = CitationManager.get_instance()
        
        url = "https://example.com/same-article"
        
        citation1 = manager.add_citation(title="First", url=url)
        citation2 = manager.add_citation(title="Second", url=url)
        
        assert citation1.id == citation2.id
        assert manager.get_citation_count() == 1
    
    def test_generate_references(self):
        """Test reference section generation."""
        from core.citation_manager import CitationManager
        
        CitationManager.reset()
        manager = CitationManager.get_instance()
        
        manager.add_citation(title="Article 1", url="https://example.com/1")
        manager.add_citation(title="Article 2", url="https://example.com/2")
        
        references = manager.generate_references_section()
        
        assert "参考文献" in references
        assert "Article 1" in references
        assert "Article 2" in references


class TestTools:
    """Test tool providers."""
    
    def test_mock_search_provider(self):
        """Test mock search provider."""
        import asyncio
        from tools import get_search_provider
        
        provider = get_search_provider(use_mock=True)
        
        async def run_search():
            results = await provider.search("test query")
            return results
        
        results = asyncio.run(run_search())
        
        assert len(results) > 0
        assert results[0].title is not None
        assert results[0].source == "mock"
    
    def test_mock_ragflow_provider(self):
        """Test mock RAGFlow provider."""
        import asyncio
        from tools import get_ragflow_provider
        
        provider = get_ragflow_provider(use_mock=True)
        
        async def run_retrieve():
            results = await provider.retrieve("test query")
            return results
        
        results = asyncio.run(run_retrieve())
        
        assert len(results) > 0
        assert results[0].content is not None


class TestWorkflow:
    """Test workflow creation."""
    
    def test_create_workflow(self):
        """Test workflow graph creation."""
        from workflow import create_workflow
        
        workflow = create_workflow()
        assert workflow is not None
    
    def test_compile_workflow(self):
        """Test workflow compilation."""
        from workflow import compile_workflow
        
        compiled = compile_workflow(use_checkpointer=False)
        assert compiled is not None
    
    def test_workflow_has_required_nodes(self):
        """Test that workflow has all required nodes."""
        from workflow import compile_workflow
        
        compiled = compile_workflow(use_checkpointer=False)
        graph = compiled.get_graph()
        
        required_nodes = ["decompose", "plan", "execute", "select", "write", "review"]
        node_names = list(graph.nodes.keys())
        
        for node in required_nodes:
            assert node in node_names, f"Missing node: {node}"


class TestPrompts:
    """Test prompt templates."""
    
    def test_prompts_not_empty(self):
        """Test that all prompts are defined and not empty."""
        from agents.prompts import (
            DECOMPOSE_SYSTEM_PROMPT,
            PLAN_SYSTEM_PROMPT,
            EXECUTION_SYSTEM_PROMPT,
            SELECTION_SYSTEM_PROMPT,
            WRITING_SYSTEM_PROMPT,
            REVIEW_SYSTEM_PROMPT,
        )
        
        prompts = [
            DECOMPOSE_SYSTEM_PROMPT,
            PLAN_SYSTEM_PROMPT,
            EXECUTION_SYSTEM_PROMPT,
            SELECTION_SYSTEM_PROMPT,
            WRITING_SYSTEM_PROMPT,
            REVIEW_SYSTEM_PROMPT,
        ]
        
        for prompt in prompts:
            assert prompt is not None
            assert len(prompt) > 100  # Should be substantial


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
