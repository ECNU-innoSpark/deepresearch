"""
Review Agent.

Responsible for quality checking the generated report,
verifying citations, and deciding whether to approve or request revisions.
"""

import json
import re
from typing import Any, Dict, List, Literal

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from core.state import GraphState, ReviewFeedback, SelectedDataItem, DataSourceType
from core.llm_client import get_llm_client
from core.citation_manager import get_citation_manager
from core.utils import get_logger, truncate_text
from .prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_TEMPLATE

logger = get_logger(__name__)


def count_citations_in_report(report: str) -> Dict[str, int]:
    """Count citation references in the report."""
    # Match patterns like [^ref_1], [^ref_2], [1], [2]
    footnote_pattern = r'\[\^ref_\d+\]'
    numeric_pattern = r'\[(\d+)\]'
    
    footnote_matches = re.findall(footnote_pattern, report)
    numeric_matches = re.findall(numeric_pattern, report)
    
    # Get unique citations
    unique_footnotes = set(footnote_matches)
    unique_numeric = set(numeric_matches)
    
    return {
        "footnote_citations": len(unique_footnotes),
        "numeric_citations": len(unique_numeric),
        "total_unique": len(unique_footnotes) + len(unique_numeric),
        "total_occurrences": len(footnote_matches) + len(numeric_matches),
    }


def count_words(text: str) -> int:
    """Count words in text (handles Chinese and English)."""
    # For Chinese: count characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # For English: count words
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    # Approximate total "words"
    return chinese_chars + english_words


def format_selected_data_for_review(selected_data: List[SelectedDataItem]) -> str:
    """Format selected data for review verification."""
    if not selected_data:
        return "无可用数据"
    
    lines = ["可用引用ID列表:\n"]
    for item in selected_data:
        lines.append(f"- {item.citation_id} ({item.citation_key}): {item.title or '无标题'} [{item.source_type.value}]")
    return "\n".join(lines)


def parse_review_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response for review decisions."""
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning("Failed to parse review response")
        return {
            "is_approved": False,
            "checks": {},
            "issues": ["无法解析审稿结果"],
            "route_to": "writing",
        }


def create_review_feedback(
    review_data: Dict[str, Any],
    selected_data: List[SelectedDataItem],
    report: str,
    max_revisions: int,
    revision_count: int
) -> ReviewFeedback:
    """Create a ReviewFeedback object from review data."""
    checks = review_data.get("checks", {})
    metrics = review_data.get("metrics", {})
    
    # Count citations by source type
    search_citations = sum(
        1 for d in selected_data
        if d.source_type == DataSourceType.SEARCH
    )
    rag_citations = sum(
        1 for d in selected_data
        if d.source_type == DataSourceType.RAG
    )
    
    # Get actual counts from report
    citation_counts = count_citations_in_report(report)
    word_count = count_words(report)
    
    # Determine routing
    route_to = review_data.get("route_to", "writing")
    is_approved = review_data.get("is_approved", False)
    
    # Force approval if max revisions reached
    if revision_count >= max_revisions:
        logger.warning(
            "Max revisions reached, forcing approval",
            revision_count=revision_count,
            max_revisions=max_revisions
        )
        is_approved = True
        route_to = "end"
    
    return ReviewFeedback(
        is_approved=is_approved,
        format_check=checks.get("format_check", True),
        citation_check=checks.get("citation_check", True),
        content_accuracy=checks.get("content_accuracy", True),
        citation_count_check=checks.get("citation_count_check", True),
        total_citations=citation_counts["total_unique"],
        search_citations=search_citations,
        rag_citations=rag_citations,
        issues=review_data.get("issues", []),
        suggestions=review_data.get("suggestions", []),
        route_to=route_to if not is_approved else "end",
    )


def review_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node for report review.
    
    Reviews the draft report and decides whether to approve or request revisions.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with review_feedback populated
    """
    logger.info(
        "Starting report review",
        revision_count=state.revision_count,
        report_length=len(state.draft_report)
    )
    
    if not state.draft_report:
        logger.warning("No draft report to review")
        return {
            "review_feedback": ReviewFeedback(
                is_approved=False,
                issues=["没有可审查的报告"],
                route_to="writing",
            ),
            "workflow_status": "reviewed",
        }
    
    # Get configuration
    config_loader = get_config()
    workflow_config = config_loader.get_workflow_config()
    min_citations = workflow_config.get("min_citations", 30)
    max_revisions = workflow_config.get("max_revisions", 3)
    
    # Get LLM client for review agent
    llm_client = get_llm_client("review")
    
    # Format inputs
    selected_data_str = format_selected_data_for_review(state.selected_data)
    word_count = count_words(state.draft_report)
    
    # Update system prompt with min_citations
    system_prompt = REVIEW_SYSTEM_PROMPT.replace(
        "{min_citations}",
        str(min_citations)
    )
    
    user_message = REVIEW_USER_TEMPLATE.format(
        original_query=state.original_query,
        draft_report=state.draft_report,
        selected_data=selected_data_str,
        min_citations=min_citations,
        min_total_sources=min_citations,
        min_words=1500,
        max_words=5000,
        revision_count=state.revision_count,
        max_revisions=max_revisions,
    )
    
    try:
        # Call LLM for review
        response = llm_client.invoke_with_prompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )
        
        # Parse response
        review_data = parse_review_response(response)
        
        # Create feedback object
        feedback = create_review_feedback(
            review_data,
            state.selected_data,
            state.draft_report,
            max_revisions,
            state.revision_count
        )
        
        # Update revision count if not approved
        new_revision_count = state.revision_count
        if not feedback.is_approved:
            new_revision_count += 1
        
        logger.info(
            "Report review completed",
            is_approved=feedback.is_approved,
            route_to=feedback.route_to,
            issues=len(feedback.issues),
            revision_count=new_revision_count
        )
        
        # Determine final report if approved
        updates: Dict[str, Any] = {
            "review_feedback": feedback,
            "revision_count": new_revision_count,
            "workflow_status": "reviewed",
        }
        
        if feedback.is_approved:
            updates["final_report"] = state.draft_report
            updates["workflow_status"] = "completed"
        
        return updates
        
    except Exception as e:
        logger.error("Report review failed", error=str(e))
        # On error, approve to avoid infinite loop
        return {
            "review_feedback": ReviewFeedback(
                is_approved=True,
                issues=[f"审查过程出错: {str(e)}"],
                route_to="end",
            ),
            "final_report": state.draft_report,
            "errors": state.errors + [f"Review failed: {str(e)}"],
            "workflow_status": "completed",
        }


async def review_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Async version of the review node.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with review_feedback populated
    """
    logger.info(
        "Starting async report review",
        revision_count=state.revision_count
    )
    
    if not state.draft_report:
        return {
            "review_feedback": ReviewFeedback(
                is_approved=False,
                issues=["没有可审查的报告"],
                route_to="writing",
            ),
            "workflow_status": "reviewed",
        }
    
    config_loader = get_config()
    workflow_config = config_loader.get_workflow_config()
    min_citations = workflow_config.get("min_citations", 30)
    max_revisions = workflow_config.get("max_revisions", 3)
    
    llm_client = get_llm_client("review")
    selected_data_str = format_selected_data_for_review(state.selected_data)
    
    system_prompt = REVIEW_SYSTEM_PROMPT.replace(
        "{min_citations}",
        str(min_citations)
    )
    
    user_message = REVIEW_USER_TEMPLATE.format(
        original_query=state.original_query,
        draft_report=state.draft_report,
        selected_data=selected_data_str,
        min_citations=min_citations,
        min_total_sources=min_citations,
        min_words=1500,
        max_words=5000,
        revision_count=state.revision_count,
        max_revisions=max_revisions,
    )
    
    try:
        response = await llm_client.ainvoke_with_prompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )
        
        review_data = parse_review_response(response)
        feedback = create_review_feedback(
            review_data,
            state.selected_data,
            state.draft_report,
            max_revisions,
            state.revision_count
        )
        
        new_revision_count = state.revision_count
        if not feedback.is_approved:
            new_revision_count += 1
        
        logger.info(
            "Async report review completed",
            is_approved=feedback.is_approved,
            route_to=feedback.route_to
        )
        
        updates: Dict[str, Any] = {
            "review_feedback": feedback,
            "revision_count": new_revision_count,
            "workflow_status": "reviewed",
        }
        
        if feedback.is_approved:
            updates["final_report"] = state.draft_report
            updates["workflow_status"] = "completed"
        
        return updates
        
    except Exception as e:
        logger.error("Async report review failed", error=str(e))
        return {
            "review_feedback": ReviewFeedback(
                is_approved=True,
                issues=[f"审查过程出错: {str(e)}"],
                route_to="end",
            ),
            "final_report": state.draft_report,
            "errors": state.errors + [f"Review failed: {str(e)}"],
            "workflow_status": "completed",
        }


def review_router(state: GraphState) -> Literal["plan", "writing", "end"]:
    """
    Conditional edge function to route based on review feedback.
    
    Returns:
        "plan" to get more data
        "writing" to rewrite the report
        "end" if approved
    """
    if not state.review_feedback:
        return "end"
    
    if state.review_feedback.is_approved:
        return "end"
    
    route = state.review_feedback.route_to
    if route == "plan":
        return "plan"
    elif route == "writing":
        return "writing"
    else:
        return "end"
