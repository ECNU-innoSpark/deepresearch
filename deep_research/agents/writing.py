"""
Writing Agent.

Responsible for generating the research report
based on selected data with proper citations.
"""

from typing import Any, Dict, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from core.state import GraphState, SelectedDataItem, SubTask
from core.llm_client import get_llm_client
from core.citation_manager import get_citation_manager
from core.utils import get_logger, truncate_text
from .prompts import WRITING_SYSTEM_PROMPT, WRITING_USER_TEMPLATE

logger = get_logger(__name__)


def format_sub_tasks_summary(sub_tasks: List[SubTask]) -> str:
    """Format sub-tasks as a summary for writing context."""
    if not sub_tasks:
        return "无子任务信息"
    
    lines = []
    for i, task in enumerate(sub_tasks, 1):
        lines.append(f"""
### 子问题 {i}: {task.question}
- 描述: {task.description or "无"}
- 关键词: {', '.join(task.keywords) if task.keywords else "无"}
""")
    return "\n".join(lines)


def format_selected_data_for_writing(selected_data: List[SelectedDataItem]) -> str:
    """Format selected data with citations for the writing prompt."""
    if not selected_data:
        return "无可用数据"
    
    lines = []
    for item in selected_data:
        lines.append(f"""
---
**引用ID**: {item.citation_id}
**引用标记**: {item.citation_key}
**来源类型**: {item.source_type.value}
**标题**: {item.title or "无标题"}
**URL**: {item.url or "N/A"}
**内容**:
{truncate_text(item.content, 800)}

**使用示例**: 根据研究显示...{item.citation_key}
""")
    return "\n".join(lines)


def format_review_feedback(feedback: Any) -> str:
    """Format review feedback for revision guidance."""
    if not feedback:
        return "这是首次撰写，无审稿意见。"
    
    lines = ["## 上一轮审稿意见\n"]
    
    if hasattr(feedback, 'issues') and feedback.issues:
        lines.append("### 需要修正的问题:")
        for issue in feedback.issues:
            lines.append(f"- {issue}")
    
    if hasattr(feedback, 'suggestions') and feedback.suggestions:
        lines.append("\n### 改进建议:")
        for suggestion in feedback.suggestions:
            lines.append(f"- {suggestion}")
    
    if hasattr(feedback, 'detailed_feedback'):
        lines.append(f"\n### 详细反馈:\n{feedback.detailed_feedback}")
    
    return "\n".join(lines)


def writing_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node for report writing.
    
    Generates a research report based on selected data.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with draft_report populated
    """
    logger.info(
        "Starting report writing",
        num_sources=len(state.selected_data),
        revision=state.revision_count
    )
    
    if not state.selected_data:
        logger.warning("No selected data available for writing")
        return {
            "draft_report": "# 研究报告\n\n无法生成报告：没有可用的数据源。",
            "workflow_status": "written",
        }
    
    # Get configuration
    config_loader = get_config()
    workflow_config = config_loader.get_workflow_config()
    min_citations = workflow_config.get("min_citations", 30)
    
    # Get LLM client for writing agent
    llm_client = get_llm_client("writing")
    
    # Format inputs
    sub_tasks_summary = format_sub_tasks_summary(state.sub_tasks)
    selected_data_str = format_selected_data_for_writing(state.selected_data)
    review_feedback_str = format_review_feedback(state.review_feedback)
    
    # Determine word count requirements
    min_words = 1500
    max_words = 5000
    
    user_message = WRITING_USER_TEMPLATE.format(
        original_query=state.original_query,
        sub_tasks_summary=sub_tasks_summary,
        selected_data=selected_data_str,
        min_words=min_words,
        max_words=max_words,
        min_citations=min(min_citations, len(state.selected_data)),
        review_feedback=review_feedback_str,
    )
    
    try:
        # Call LLM for writing
        response = llm_client.invoke_with_prompt(
            system_prompt=WRITING_SYSTEM_PROMPT,
            user_message=user_message,
        )
        
        # Get citation manager for references
        citation_manager = get_citation_manager()
        references_section = citation_manager.generate_references_markdown()
        
        # Combine report with references
        full_report = response
        if references_section:
            full_report = f"{response}\n\n{references_section}"
        
        logger.info(
            "Report writing completed",
            report_length=len(full_report),
            has_references=bool(references_section)
        )
        
        return {
            "draft_report": full_report,
            "references_section": references_section,
            "workflow_status": "written",
        }
        
    except Exception as e:
        logger.error("Report writing failed", error=str(e))
        return {
            "draft_report": f"# 报告生成失败\n\n错误: {str(e)}",
            "errors": state.errors + [f"Writing failed: {str(e)}"],
            "workflow_status": "error",
        }


async def writing_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Async version of the writing node.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with draft_report populated
    """
    logger.info(
        "Starting async report writing",
        num_sources=len(state.selected_data),
        revision=state.revision_count
    )
    
    if not state.selected_data:
        logger.warning("No selected data available for writing")
        return {
            "draft_report": "# 研究报告\n\n无法生成报告：没有可用的数据源。",
            "workflow_status": "written",
        }
    
    config_loader = get_config()
    workflow_config = config_loader.get_workflow_config()
    min_citations = workflow_config.get("min_citations", 30)
    
    llm_client = get_llm_client("writing")
    
    sub_tasks_summary = format_sub_tasks_summary(state.sub_tasks)
    selected_data_str = format_selected_data_for_writing(state.selected_data)
    review_feedback_str = format_review_feedback(state.review_feedback)
    
    min_words = 1500
    max_words = 5000
    
    user_message = WRITING_USER_TEMPLATE.format(
        original_query=state.original_query,
        sub_tasks_summary=sub_tasks_summary,
        selected_data=selected_data_str,
        min_words=min_words,
        max_words=max_words,
        min_citations=min(min_citations, len(state.selected_data)),
        review_feedback=review_feedback_str,
    )
    
    try:
        response = await llm_client.ainvoke_with_prompt(
            system_prompt=WRITING_SYSTEM_PROMPT,
            user_message=user_message,
        )
        
        citation_manager = get_citation_manager()
        references_section = citation_manager.generate_references_markdown()
        
        full_report = response
        if references_section:
            full_report = f"{response}\n\n{references_section}"
        
        logger.info(
            "Async report writing completed",
            report_length=len(full_report)
        )
        
        return {
            "draft_report": full_report,
            "references_section": references_section,
            "workflow_status": "written",
        }
        
    except Exception as e:
        logger.error("Async report writing failed", error=str(e))
        return {
            "draft_report": f"# 报告生成失败\n\n错误: {str(e)}",
            "errors": state.errors + [f"Writing failed: {str(e)}"],
            "workflow_status": "error",
        }
