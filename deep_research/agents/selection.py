"""
Selection Agent.

Responsible for filtering and scoring collected data,
assigning citation IDs, and prioritizing high-quality sources.
"""

import json
from typing import Any, Dict, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from core.state import GraphState, RawDataItem, SelectedDataItem, DataSourceType
from core.llm_client import get_llm_client
from core.citation_manager import get_citation_manager, Citation
from core.utils import get_logger, safe_json_loads, truncate_text
from .prompts import SELECTION_SYSTEM_PROMPT, SELECTION_USER_TEMPLATE

logger = get_logger(__name__)


def format_raw_data_for_selection(raw_data: List[RawDataItem]) -> str:
    """Format raw data items as a string for the selection prompt."""
    lines = []
    for item in raw_data:
        lines.append(f"""
---
**ID**: {item.id}
**来源类型**: {item.source_type.value}
**标题**: {item.title or "无标题"}
**URL**: {item.url or "N/A"}
**内容摘要**: {truncate_text(item.content, 500)}
**初始相关性**: {item.relevance_score:.2f}
**初始置信度**: {item.confidence_score:.2f}
""")
    return "\n".join(lines)


def parse_selection_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response for selection decisions."""
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
        logger.warning("Failed to parse selection response")
        return {"selections": [], "summary": {}}


def apply_selection_rules(
    raw_data: List[RawDataItem],
    selections: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[SelectedDataItem]:
    """
    Apply selection rules and create SelectedDataItem objects.
    
    Args:
        raw_data: List of raw data items
        selections: LLM selection decisions
        config: Selection configuration
        
    Returns:
        List of selected data items with citations
    """
    citation_manager = get_citation_manager()
    
    min_relevance = config.get("min_relevance", 0.5)
    min_confidence = config.get("min_confidence", 0.6)
    rag_priority_threshold = config.get("rag_priority_threshold", 0.8)
    max_items = config.get("max_selected_items", 50)
    
    # Create lookup for raw data
    raw_data_map = {item.id: item for item in raw_data}
    
    # Create lookup for selection decisions
    selection_map = {s.get("data_id", ""): s for s in selections}
    
    selected_items: List[SelectedDataItem] = []
    
    for raw_item in raw_data:
        selection = selection_map.get(raw_item.id, {})
        decision = selection.get("decision", "")
        
        # Get scores from selection or use original
        relevance = selection.get("relevance_score", raw_item.relevance_score)
        confidence = selection.get("confidence_score", raw_item.confidence_score)
        
        # Apply decision rules
        if decision == "discard":
            continue
        
        # Apply threshold rules if no explicit decision
        if not decision:
            if relevance < min_relevance or confidence < min_confidence:
                continue
        
        # RAG priority: boost RAG sources with high confidence
        if raw_item.source_type == DataSourceType.RAG:
            if confidence >= rag_priority_threshold:
                relevance = min(1.0, relevance + 0.1)  # Slight boost
        
        # Skip if still below thresholds after rules
        if decision != "keep" and (relevance < min_relevance or confidence < min_confidence):
            continue
        
        # Create citation
        citation = citation_manager.add_citation(
            title=raw_item.title,
            url=raw_item.url,
            content_snippet=truncate_text(raw_item.content, 200),
            source_type=raw_item.source_type.value,
            relevance_score=relevance,
            confidence_score=confidence,
            metadata=raw_item.metadata,
        )
        
        # Create selected data item
        selected_item = SelectedDataItem(
            raw_data_id=raw_item.id,
            task_id=raw_item.task_id,
            source_type=raw_item.source_type,
            content=raw_item.content,
            title=raw_item.title,
            url=raw_item.url,
            citation_id=citation.id,
            citation_key=citation.get_citation_key(),
            relevance_score=relevance,
            confidence_score=confidence,
            selection_reason=selection.get("reason", ""),
        )
        selected_items.append(selected_item)
    
    # Sort by combined score and limit
    selected_items.sort(
        key=lambda x: (x.relevance_score + x.confidence_score) / 2,
        reverse=True
    )
    
    return selected_items[:max_items]


def prefilter_raw_data(raw_data: List[RawDataItem], min_score: float = 0.4) -> List[RawDataItem]:
    """
    预筛选：根据初始分数过滤明显低质量的数据。
    
    Args:
        raw_data: 原始数据列表
        min_score: 最低分数阈值
        
    Returns:
        预筛选后的数据列表
    """
    # 按综合分数排序
    scored_data = [
        (item, (item.relevance_score + item.confidence_score) / 2)
        for item in raw_data
    ]
    scored_data.sort(key=lambda x: x[1], reverse=True)
    
    # 过滤低分数据，并限制数量（最多保留前50条）
    filtered = [item for item, score in scored_data if score >= min_score]
    return filtered[:50]


def process_selection_batch(
    llm_client,
    batch_data: List[RawDataItem],
    original_query: str,
    current_task_info: str,
    selection_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    处理一批数据的选择。
    
    Args:
        llm_client: LLM 客户端
        batch_data: 批次数据
        original_query: 原始查询
        current_task_info: 当前任务信息
        selection_config: 筛选配置
        
    Returns:
        选择结果列表
    """
    raw_data_str = format_raw_data_for_selection(batch_data)
    
    user_message = SELECTION_USER_TEMPLATE.format(
        original_query=original_query,
        current_task=current_task_info,
        raw_data=raw_data_str,
        min_relevance=selection_config.get("min_relevance", 0.5),
        min_confidence=selection_config.get("min_confidence", 0.6),
        rag_priority_threshold=selection_config.get("rag_priority_threshold", 0.8),
    )
    
    response = llm_client.invoke_with_prompt(
        system_prompt=SELECTION_SYSTEM_PROMPT,
        user_message=user_message,
    )
    
    selection_result = parse_selection_response(response)
    return selection_result.get("selections", [])


def selection_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node for data selection.
    
    Filters and scores collected data, assigns citation IDs.
    使用预筛选和分批处理来避免上下文过长。
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with selected_data populated
    """
    logger.info("Starting data selection", num_raw_items=len(state.raw_data))
    
    if not state.raw_data:
        logger.warning("No raw data to select from")
        return {
            "selected_data": [],
            "workflow_status": "selected",
        }
    
    # Get configuration
    config_loader = get_config()
    workflow_config = config_loader.get_workflow_config()
    selection_config = workflow_config.get("selection", {})
    
    # 预筛选：减少数据量
    prefiltered_data = prefilter_raw_data(state.raw_data, min_score=0.4)
    logger.info("Prefiltered data", original=len(state.raw_data), filtered=len(prefiltered_data))
    
    # 如果预筛选后数据量仍然很大，使用基于规则的快速筛选而不调用 LLM
    BATCH_SIZE = 20  # 每批最多处理20条
    MAX_BATCHES = 3  # 最多处理3批
    
    if len(prefiltered_data) > BATCH_SIZE * MAX_BATCHES:
        logger.info("Too much data, using rule-based selection without LLM")
        # 直接使用规则筛选
        citation_manager = get_citation_manager()
        selected_data = []
        
        for raw_item in prefiltered_data[:50]:  # 最多选50条
            if raw_item.relevance_score >= selection_config.get("min_relevance", 0.5):
                citation = citation_manager.add_citation(
                    title=raw_item.title,
                    url=raw_item.url,
                    content_snippet=truncate_text(raw_item.content, 200),
                    source_type=raw_item.source_type.value,
                    relevance_score=raw_item.relevance_score,
                    confidence_score=raw_item.confidence_score,
                )
                selected_data.append(SelectedDataItem(
                    raw_data_id=raw_item.id,
                    task_id=raw_item.task_id,
                    source_type=raw_item.source_type,
                    content=raw_item.content,
                    title=raw_item.title,
                    url=raw_item.url,
                    citation_id=citation.id,
                    citation_key=citation.get_citation_key(),
                    relevance_score=raw_item.relevance_score,
                    confidence_score=raw_item.confidence_score,
                    selection_reason="Rule-based selection (data volume too large for LLM)",
                ))
        
        search_count = sum(1 for d in selected_data if d.source_type == DataSourceType.SEARCH)
        rag_count = sum(1 for d in selected_data if d.source_type == DataSourceType.RAG)
        
        logger.info("Rule-based selection completed", total=len(selected_data), search=search_count, rag=rag_count)
        
        return {
            "selected_data": selected_data,
            "workflow_status": "selected",
        }
    
    # Get LLM client for selection agent
    llm_client = get_llm_client("selection")
    
    # Get current task info
    current_task_info = "综合所有子任务"
    if state.sub_tasks:
        task_questions = [t.question for t in state.sub_tasks]
        current_task_info = "\n".join([f"- {q}" for q in task_questions])
    
    try:
        all_selections = []
        
        # 分批处理
        for i in range(0, len(prefiltered_data), BATCH_SIZE):
            batch = prefiltered_data[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            logger.info(f"Processing batch {batch_num}", batch_size=len(batch))
            
            batch_selections = process_selection_batch(
                llm_client,
                batch,
                state.original_query,
                current_task_info,
                selection_config
            )
            all_selections.extend(batch_selections)
            
            # 最多处理 MAX_BATCHES 批
            if batch_num >= MAX_BATCHES:
                break
        
        # Apply selection rules and create citations
        selected_data = apply_selection_rules(
            prefiltered_data,
            all_selections,
            selection_config
        )
        
        # Count by source type
        search_count = sum(1 for d in selected_data if d.source_type == DataSourceType.SEARCH)
        rag_count = sum(1 for d in selected_data if d.source_type == DataSourceType.RAG)
        
        logger.info(
            "Data selection completed",
            total_selected=len(selected_data),
            search_sources=search_count,
            rag_sources=rag_count
        )
        
        return {
            "selected_data": selected_data,
            "workflow_status": "selected",
        }
        
    except Exception as e:
        logger.error("Data selection failed", error=str(e))
        # Fallback: select all with basic filtering
        citation_manager = get_citation_manager()
        fallback_selected = []
        
        for raw_item in prefiltered_data:
            if raw_item.relevance_score >= 0.4:
                citation = citation_manager.add_citation(
                    title=raw_item.title,
                    url=raw_item.url,
                    content_snippet=truncate_text(raw_item.content, 200),
                    source_type=raw_item.source_type.value,
                )
                fallback_selected.append(SelectedDataItem(
                    raw_data_id=raw_item.id,
                    task_id=raw_item.task_id,
                    source_type=raw_item.source_type,
                    content=raw_item.content,
                    title=raw_item.title,
                    url=raw_item.url,
                    citation_id=citation.id,
                    citation_key=citation.get_citation_key(),
                    relevance_score=raw_item.relevance_score,
                    confidence_score=raw_item.confidence_score,
                ))
        
        return {
            "selected_data": fallback_selected[:50],
            "errors": state.errors + [f"Selection failed, using fallback: {str(e)}"],
            "workflow_status": "selected",
        }


async def selection_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Async version of the selection node.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with selected_data populated
    """
    logger.info("Starting async data selection", num_raw_items=len(state.raw_data))
    
    if not state.raw_data:
        return {
            "selected_data": [],
            "workflow_status": "selected",
        }
    
    config_loader = get_config()
    workflow_config = config_loader.get_workflow_config()
    selection_config = workflow_config.get("selection", {})
    
    llm_client = get_llm_client("selection")
    raw_data_str = format_raw_data_for_selection(state.raw_data)
    
    current_task_info = "综合所有子任务"
    if state.sub_tasks:
        task_questions = [t.question for t in state.sub_tasks]
        current_task_info = "\n".join([f"- {q}" for q in task_questions])
    
    user_message = SELECTION_USER_TEMPLATE.format(
        original_query=state.original_query,
        current_task=current_task_info,
        raw_data=raw_data_str,
        min_relevance=selection_config.get("min_relevance", 0.5),
        min_confidence=selection_config.get("min_confidence", 0.6),
        rag_priority_threshold=selection_config.get("rag_priority_threshold", 0.8),
    )
    
    try:
        response = await llm_client.ainvoke_with_prompt(
            system_prompt=SELECTION_SYSTEM_PROMPT,
            user_message=user_message,
        )
        
        selection_result = parse_selection_response(response)
        selections = selection_result.get("selections", [])
        
        selected_data = apply_selection_rules(
            state.raw_data,
            selections,
            selection_config
        )
        
        logger.info(
            "Async data selection completed",
            total_selected=len(selected_data)
        )
        
        return {
            "selected_data": selected_data,
            "workflow_status": "selected",
        }
        
    except Exception as e:
        logger.error("Async data selection failed", error=str(e))
        return {
            "selected_data": [],
            "errors": state.errors + [f"Selection failed: {str(e)}"],
            "workflow_status": "selected",
        }
