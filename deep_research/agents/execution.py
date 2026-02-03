"""
Execution Agent.

Responsible for executing data gathering tasks using
various tools (Web Search, RAGFlow, MCP).
"""

import json
import asyncio
from typing import Any, Dict, List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.state import GraphState, SubTask, RawDataItem, TaskStatus, DataSourceType
from core.llm_client import get_llm_client
from config import get_config
from core.utils import get_logger, safe_json_loads, generate_id
from tools import get_search_provider, get_ragflow_provider, get_mcp_client
from .prompts import EXECUTION_SYSTEM_PROMPT, EXECUTION_USER_TEMPLATE

logger = get_logger(__name__)


def get_current_task(state: GraphState) -> Optional[SubTask]:
    """Get the current task to execute based on task_plan and index."""
    if not state.task_plan or state.current_task_index >= len(state.task_plan):
        return None
    
    current_task_id = state.task_plan[state.current_task_index]
    for task in state.sub_tasks:
        if task.id == current_task_id:
            return task
    return None


def parse_execution_response(response: str) -> Dict[str, Any]:
    """Parse the LLM response for tool calls."""
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
        logger.warning("Failed to parse execution response, using defaults")
        return {"tool_calls": [], "search_strategy": ""}


async def execute_web_search(query: str, use_mock: bool = False) -> List[RawDataItem]:
    """Execute web search and convert results to RawDataItem."""
    search_provider = get_search_provider(use_mock=use_mock)
    results = await search_provider.search(query)
    
    raw_items = []
    for result in results:
        raw_items.append(RawDataItem(
            id=generate_id("raw"),
            task_id="",  # Will be set later
            source_type=DataSourceType.SEARCH,
            content=result.snippet or result.content or "",
            title=result.title,
            url=result.url,
            relevance_score=result.relevance_score,
            confidence_score=0.5,  # Default confidence for web search
            metadata={
                "source": result.source,
                "published_date": result.published_date,
            },
        ))
    
    return raw_items


async def execute_ragflow_search(query: str, use_mock: bool = False) -> List[RawDataItem]:
    """Execute RAGFlow retrieval and convert results to RawDataItem."""
    ragflow_provider = get_ragflow_provider(use_mock=use_mock)
    results = await ragflow_provider.retrieve(query)
    
    raw_items = []
    for result in results:
        raw_items.append(RawDataItem(
            id=generate_id("raw"),
            task_id="",  # Will be set later
            source_type=DataSourceType.RAG,
            content=result.content,
            title=result.title,
            url=result.url,
            relevance_score=result.similarity_score,
            confidence_score=result.confidence_score,
            metadata={
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "dataset_id": result.dataset_id,
            },
        ))
    
    return raw_items


async def execute_mcp_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    use_mock: bool = False
) -> List[RawDataItem]:
    """Execute MCP tool and convert results to RawDataItem."""
    mcp_client = get_mcp_client(use_mock=use_mock)
    
    if not mcp_client.enabled:
        return []
    
    result = await mcp_client.call_tool(tool_name, arguments)
    
    if not result.success:
        logger.warning(f"MCP tool {tool_name} failed", error=result.error_message)
        return []
    
    return [RawDataItem(
        id=generate_id("raw"),
        task_id="",
        source_type=DataSourceType.MCP,
        content=result.content,
        title=f"MCP: {tool_name}",
        url=None,
        relevance_score=0.7,
        confidence_score=0.6,
        metadata={
            "server": result.server_name,
            "tool": result.tool_name,
        },
    )]


async def execute_task_tools(
    task: SubTask,
    original_query: str,
    use_mock: bool = False
) -> List[RawDataItem]:
    """
    Execute appropriate tools for a task.
    
    Args:
        task: The sub-task to execute
        original_query: The original research question
        use_mock: Whether to use mock tools
        
    Returns:
        List of raw data items collected
    """
    tool_calls = []
    logger.debug("Selecting tools for task", task_id=task.id, use_mock=use_mock)
    if not use_mock:
        # Get LLM to decide tool usage
        llm_client = get_llm_client("execution")
        
        user_message = EXECUTION_USER_TEMPLATE.format(
            question=task.question,
            description=task.description or "",
            keywords=", ".join(task.keywords),
            preferred_sources=", ".join([s.value for s in task.preferred_sources]),
            original_query=original_query,
        )
        
        response = await llm_client.ainvoke_with_prompt(
            system_prompt=EXECUTION_SYSTEM_PROMPT,
            user_message=user_message,
        )
        
        tool_plan = parse_execution_response(response)
        tool_calls = tool_plan.get("tool_calls", [])
    
    # If no tool calls specified, use defaults based on preferred sources
    if not tool_calls:
        search_query = task.question
        if DataSourceType.SEARCH in task.preferred_sources:
            tool_calls.append({"tool": "web_search", "query": search_query})
        if DataSourceType.RAG in task.preferred_sources:
            tool_calls.append({"tool": "ragflow_search", "query": search_query})
        if not tool_calls:
            tool_calls.append({"tool": "web_search", "query": search_query})
    
    logger.debug("Tool calls decided", task_id=task.id, num_calls=len(tool_calls))
    
    # Execute tools in parallel with timeout
    all_results: List[RawDataItem] = []
    tasks = []
    workflow_config = get_config().get_workflow_config()
    tool_timeout = workflow_config.get("tool_timeout_seconds", 60)
    
    for call in tool_calls:
        tool_type = call.get("tool", "")
        query = call.get("query", task.question)
        
        if tool_type == "web_search":
            tasks.append(asyncio.wait_for(execute_web_search(query, use_mock), timeout=tool_timeout))
        elif tool_type == "ragflow_search":
            tasks.append(asyncio.wait_for(execute_ragflow_search(query, use_mock), timeout=tool_timeout))
        elif tool_type == "mcp_tool":
            tool_name = call.get("tool_name", "")
            args = call.get("arguments", {})
            tasks.append(asyncio.wait_for(execute_mcp_tool(tool_name, args, use_mock), timeout=tool_timeout))
    
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                # Set task_id for all items
                for item in result:
                    item.task_id = task.id
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.error("Tool execution failed", error=str(result))
    
    logger.info(
        "Task tools executed",
        task_id=task.id,
        num_results=len(all_results),
        tool_calls=len(tool_calls)
    )
    
    return all_results


def execution_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node for task execution.
    
    Executes the current task using appropriate tools.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with raw_data populated
    """
    logger.info(
        "Starting task execution",
        task_index=state.current_task_index,
        total_tasks=len(state.task_plan)
    )
    
    current_task = get_current_task(state)
    if not current_task:
        logger.warning("No current task to execute")
        return {
            "workflow_status": "execution_complete",
        }
    
    # Update task status
    current_task.status = TaskStatus.IN_PROGRESS
    
    try:
        # Run async execution
        # 从环境变量读取是否使用 Mock 模式
        import os
        use_mock = os.getenv("DEEP_RESEARCH_USE_MOCK", "false").lower() in ("true", "1", "yes")
        logger.debug("Execution mode resolved", use_mock=use_mock)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            new_data = loop.run_until_complete(
                execute_task_tools(
                    current_task,
                    state.original_query,
                    use_mock=use_mock
                )
            )
        finally:
            loop.close()
        
        # Update task status
        current_task.status = TaskStatus.COMPLETED
        
        # Merge with existing raw_data
        all_raw_data = list(state.raw_data) + new_data
        
        # Move to next task
        new_index = state.current_task_index + 1
        is_complete = new_index >= len(state.task_plan)
        
        logger.info(
            "Task execution completed",
            task_id=current_task.id,
            new_items=len(new_data),
            total_items=len(all_raw_data),
            is_complete=is_complete
        )
        
        return {
            "raw_data": all_raw_data,
            "current_task_index": new_index,
            "workflow_status": "execution_complete" if is_complete else "executing",
        }
        
    except Exception as e:
        logger.error("Task execution failed", error=str(e), task_id=current_task.id)
        current_task.status = TaskStatus.FAILED
        return {
            "current_task_index": state.current_task_index + 1,
            "errors": state.errors + [f"Execution failed for {current_task.id}: {str(e)}"],
            "workflow_status": "executing",
        }


async def execution_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Async version of the execution node.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with raw_data populated
    """
    logger.info(
        "Starting async task execution",
        task_index=state.current_task_index,
        total_tasks=len(state.task_plan)
    )
    
    current_task = get_current_task(state)
    if not current_task:
        logger.warning("No current task to execute")
        return {
            "workflow_status": "execution_complete",
        }
    
    current_task.status = TaskStatus.IN_PROGRESS
    
    try:
        new_data = await execute_task_tools(
            current_task,
            state.original_query,
            use_mock=False
        )
        
        current_task.status = TaskStatus.COMPLETED
        all_raw_data = list(state.raw_data) + new_data
        
        new_index = state.current_task_index + 1
        is_complete = new_index >= len(state.task_plan)
        
        logger.info(
            "Async task execution completed",
            task_id=current_task.id,
            new_items=len(new_data),
            is_complete=is_complete
        )
        
        return {
            "raw_data": all_raw_data,
            "current_task_index": new_index,
            "workflow_status": "execution_complete" if is_complete else "executing",
        }
        
    except Exception as e:
        logger.error("Async task execution failed", error=str(e))
        current_task.status = TaskStatus.FAILED
        return {
            "current_task_index": state.current_task_index + 1,
            "errors": state.errors + [f"Execution failed for {current_task.id}: {str(e)}"],
            "workflow_status": "executing",
        }


def should_continue_execution(state: GraphState) -> str:
    """
    Conditional edge function to determine if execution should continue.
    
    Returns:
        "continue" if more tasks to execute, "done" otherwise
    """
    if not state.task_plan:
        return "done"
    if state.current_task_index < len(state.task_plan):
        return "continue"
    return "done"
