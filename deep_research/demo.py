"""
Deep Research Demo Script

Demonstrates the complete workflow using mock components.
This is useful for testing the system without actual API calls.

Usage:
    python demo.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.state import create_initial_state, GraphState
from core.citation_manager import CitationManager
from core.utils import setup_logging, print_banner, print_step
from workflow import compile_workflow


# Mock responses for demo
MOCK_RESPONSES = {
    "decompose": json.dumps({
        "sub_tasks": [
            {
                "id": "task_1",
                "question": "大型语言模型的核心技术原理是什么?",
                "description": "了解Transformer架构和注意力机制",
                "keywords": ["LLM", "Transformer", "注意力机制", "GPT"],
                "preferred_sources": ["search", "rag"]
            },
            {
                "id": "task_2",
                "question": "主流大型语言模型有哪些及其特点?",
                "description": "比较GPT、Claude、Llama等模型",
                "keywords": ["GPT-4", "Claude", "Llama", "模型对比"],
                "preferred_sources": ["search"]
            },
            {
                "id": "task_3",
                "question": "大型语言模型的应用场景有哪些?",
                "description": "探索LLM的实际应用",
                "keywords": ["LLM应用", "代码生成", "对话系统", "内容创作"],
                "preferred_sources": ["search", "rag"]
            },
            {
                "id": "task_4",
                "question": "大型语言模型面临的挑战和局限性?",
                "description": "分析LLM的问题和改进方向",
                "keywords": ["幻觉", "偏见", "安全性", "可解释性"],
                "preferred_sources": ["search"]
            }
        ]
    }),
    
    "plan": json.dumps({
        "task_plan": ["task_1", "task_2", "task_3", "task_4"],
        "dependencies": {
            "task_2": ["task_1"],
            "task_3": ["task_1"],
            "task_4": ["task_1", "task_2", "task_3"]
        },
        "parallel_groups": [["task_2", "task_3"]],
        "reasoning": "首先理解技术原理，然后并行了解模型和应用，最后分析挑战"
    }),
    
    "execution": json.dumps({
        "tool_calls": [
            {"tool": "web_search", "query": "大型语言模型技术原理 Transformer"},
            {"tool": "ragflow_search", "query": "LLM架构原理"}
        ]
    }),
    
    "selection": json.dumps({
        "selections": [
            {"data_id": "raw_1", "decision": "keep", "relevance_score": 0.92, "confidence_score": 0.88},
            {"data_id": "raw_2", "decision": "keep", "relevance_score": 0.88, "confidence_score": 0.91},
            {"data_id": "raw_3", "decision": "keep", "relevance_score": 0.85, "confidence_score": 0.80},
        ]
    }),
    
    "writing": """# 大型语言模型（LLM）深度研究报告

## 摘要

本报告全面分析了大型语言模型的技术原理、主流模型对比、应用场景及面临的挑战。通过深入研究，我们发现LLM正在深刻改变人工智能的应用格局[^ref_1]。

## 1. 技术原理

### 1.1 Transformer架构

大型语言模型的核心是Transformer架构，该架构于2017年由Google提出[^ref_1]。其核心创新在于自注意力机制（Self-Attention），能够有效捕捉序列中的长距离依赖关系[^ref_2]。

### 1.2 预训练与微调

现代LLM采用"预训练+微调"范式：
- **预训练阶段**：在大规模文本语料上学习语言的统计规律[^ref_1]
- **微调阶段**：针对特定任务进行优化[^ref_3]

## 2. 主流模型对比

| 模型 | 开发者 | 参数量 | 特点 |
|------|--------|--------|------|
| GPT-4 | OpenAI | 未公开 | 多模态能力强[^ref_2] |
| Claude | Anthropic | 未公开 | 安全性突出[^ref_3] |
| Llama 2 | Meta | 7B-70B | 开源可商用[^ref_1] |

## 3. 应用场景

### 3.1 代码生成
LLM在代码生成领域表现出色，GitHub Copilot已被广泛使用[^ref_2][^ref_3]。

### 3.2 对话系统
ChatGPT等对话系统改变了人机交互方式[^ref_1]。

### 3.3 内容创作
在文案写作、翻译等领域，LLM显著提升了效率[^ref_3]。

## 4. 挑战与局限

### 4.1 幻觉问题
LLM可能生成看似合理但实际错误的内容[^ref_2]。

### 4.2 偏见与公平性
训练数据中的偏见可能被模型继承[^ref_1]。

### 4.3 计算成本
训练和推理需要大量计算资源[^ref_3]。

## 5. 结论

大型语言模型代表了人工智能的重要突破，但仍需解决幻觉、偏见等问题。随着技术进步，LLM将在更多领域发挥重要作用[^ref_1][^ref_2][^ref_3]。
""",
    
    "review": json.dumps({
        "is_approved": True,
        "checks": {
            "format_check": True,
            "citation_check": True,
            "content_accuracy": True,
            "citation_count_check": True
        },
        "metrics": {
            "total_citations": 3,
            "search_citations": 2,
            "rag_citations": 1,
            "word_count": 800
        },
        "issues": [],
        "suggestions": ["可以增加更多技术细节"],
        "route_to": "end",
        "detailed_feedback": "报告结构清晰，引用规范，内容准确。建议后续可补充更多技术实现细节。"
    })
}


class MockLLMClient:
    """Mock LLM client for demo."""
    
    def __init__(self, agent_name=None):
        self.agent_name = agent_name
        self._call_count = 0
    
    def invoke_with_prompt(self, system_prompt, user_message):
        """Return mock response based on agent type."""
        self._call_count += 1
        
        # Determine which response to return based on context
        if "分解" in system_prompt or "拆解" in system_prompt or "decompos" in system_prompt.lower():
            return MOCK_RESPONSES["decompose"]
        elif "规划" in system_prompt or "排序" in system_prompt or "plan" in system_prompt.lower():
            return MOCK_RESPONSES["plan"]
        elif "检索" in system_prompt or "工具" in system_prompt or "execution" in system_prompt.lower():
            return MOCK_RESPONSES["execution"]
        elif "筛选" in system_prompt or "评估" in system_prompt or "selection" in system_prompt.lower():
            return MOCK_RESPONSES["selection"]
        elif "写作" in system_prompt or "撰写" in system_prompt or "writing" in system_prompt.lower():
            return MOCK_RESPONSES["writing"]
        elif "审查" in system_prompt or "检查" in system_prompt or "review" in system_prompt.lower():
            return MOCK_RESPONSES["review"]
        else:
            return "{}"


def run_demo():
    """Run the demo workflow."""
    print_banner("Deep Research Demo")
    print("\n这是一个使用模拟数据的演示，展示完整的研究流程。\n")
    
    # Setup
    setup_logging(level="INFO")
    CitationManager.reset()
    
    # Add mock citations
    manager = CitationManager.get_instance()
    manager.add_citation(
        title="Attention Is All You Need",
        url="https://arxiv.org/abs/1706.03762",
        source_type="search",
    )
    manager.add_citation(
        title="GPT-4 Technical Report",
        url="https://arxiv.org/abs/2303.08774",
        source_type="search",
    )
    manager.add_citation(
        title="LLM内部技术文档",
        url="https://internal.rag/llm-docs",
        source_type="rag",
    )
    
    # Research query
    query = "大型语言模型的技术原理、应用场景和发展趋势"
    print(f"研究问题: {query}\n")
    print("-" * 60)
    
    # Create initial state
    initial_state = create_initial_state(query)
    
    # Compile workflow
    print_step("编译工作流", "running")
    workflow = compile_workflow(use_checkpointer=False)
    print_step("编译工作流", "done")
    
    # Mock the LLM client
    def mock_get_llm_client(agent_name=None):
        return MockLLMClient(agent_name)
    
    # Run with mocks
    print_step("开始模拟研究流程", "running")
    print()
    
    with patch('agents.decompose.get_llm_client', mock_get_llm_client), \
         patch('agents.plan.get_llm_client', mock_get_llm_client), \
         patch('agents.execution.get_llm_client', mock_get_llm_client), \
         patch('agents.selection.get_llm_client', mock_get_llm_client), \
         patch('agents.writing.get_llm_client', mock_get_llm_client), \
         patch('agents.review.get_llm_client', mock_get_llm_client), \
         patch('agents.execution.execute_web_search') as mock_search, \
         patch('agents.execution.execute_ragflow_search') as mock_rag:
        
        # Mock tool responses
        from core.state import RawDataItem, DataSourceType
        
        async def mock_search_fn(query, use_mock=False):
            return [
                RawDataItem(
                    id="raw_1",
                    task_id="task_1",
                    source_type=DataSourceType.SEARCH,
                    content="Transformer架构是现代大型语言模型的基础...",
                    title="Transformer Architecture",
                    url="https://example.com/transformer",
                    relevance_score=0.9,
                    confidence_score=0.85,
                ),
                RawDataItem(
                    id="raw_2",
                    task_id="task_1",
                    source_type=DataSourceType.SEARCH,
                    content="GPT-4展示了强大的多模态能力...",
                    title="GPT-4 Overview",
                    url="https://example.com/gpt4",
                    relevance_score=0.85,
                    confidence_score=0.80,
                ),
            ]
        
        async def mock_rag_fn(query, use_mock=False):
            return [
                RawDataItem(
                    id="raw_3",
                    task_id="task_1",
                    source_type=DataSourceType.RAG,
                    content="内部文档：LLM技术实现详解...",
                    title="LLM技术文档",
                    url="https://rag.internal/llm",
                    relevance_score=0.88,
                    confidence_score=0.92,
                ),
            ]
        
        mock_search.side_effect = mock_search_fn
        mock_rag.side_effect = mock_rag_fn
        
        # Run workflow
        try:
            for event in workflow.stream(initial_state.model_dump()):
                for node_name, state_update in event.items():
                    print_step(f"执行节点: {node_name}", "done")
                    
                    # Show some details
                    if "sub_tasks" in state_update and state_update["sub_tasks"]:
                        print(f"    → 生成 {len(state_update['sub_tasks'])} 个子任务")
                    if "task_plan" in state_update and state_update["task_plan"]:
                        print(f"    → 制定执行计划: {state_update['task_plan']}")
                    if "raw_data" in state_update:
                        print(f"    → 收集 {len(state_update['raw_data'])} 条原始数据")
                    if "selected_data" in state_update:
                        print(f"    → 筛选 {len(state_update['selected_data'])} 条数据")
                    if "draft_report" in state_update and state_update["draft_report"]:
                        word_count = len(state_update["draft_report"])
                        print(f"    → 生成报告 ({word_count} 字符)")
                    if "review_feedback" in state_update and state_update["review_feedback"]:
                        feedback = state_update["review_feedback"]
                        status = "✓ 通过" if feedback.is_approved else "✗ 需修改"
                        print(f"    → 审核结果: {status}")
            
            print()
            print_step("研究流程完成", "done")
            
            # Show final report
            print("\n" + "=" * 60)
            print("最终报告预览")
            print("=" * 60)
            print(MOCK_RESPONSES["writing"][:1000] + "...")
            
            print("\n" + "=" * 60)
            print("参考文献")
            print("=" * 60)
            print(manager.generate_references_section())
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_demo()
