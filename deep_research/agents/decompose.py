"""
Task Decomposition Agent.

Responsible for breaking down the user's research question
into manageable sub-tasks (maximum 5).
"""

import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.state import GraphState, SubTask, TaskStatus, DataSourceType
from core.llm_client import get_llm_client
from core.utils import get_logger, safe_json_loads
from .prompts import DECOMPOSE_SYSTEM_PROMPT, DECOMPOSE_USER_TEMPLATE

logger = get_logger(__name__)


class DecomposeOutput(BaseModel):
    """Structured output for task decomposition."""
    
    sub_tasks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of decomposed sub-tasks"
    )


def parse_decompose_response(response: str) -> List[SubTask]:
    """
    Parse the LLM response into SubTask objects.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        List of SubTask objects
    """
    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        data = json.loads(response)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse decompose response as JSON", error=str(e))
        # Attempt to extract with safe_json_loads
        data = safe_json_loads(response, {"sub_tasks": []})
    
    sub_tasks = []
    raw_tasks = data.get("sub_tasks", [])
    
    for i, task_data in enumerate(raw_tasks[:5]):  # Limit to 5 tasks
        # Parse preferred sources
        sources_raw = task_data.get("preferred_sources", ["search", "rag"])
        sources = []
        for s in sources_raw:
            if s == "search":
                sources.append(DataSourceType.SEARCH)
            elif s == "rag":
                sources.append(DataSourceType.RAG)
            elif s == "mcp":
                sources.append(DataSourceType.MCP)
        
        if not sources:
            sources = [DataSourceType.SEARCH, DataSourceType.RAG]
        
        sub_task = SubTask(
            id=task_data.get("id", f"task_{i+1}"),
            question=task_data.get("question", ""),
            description=task_data.get("description", ""),
            keywords=task_data.get("keywords", []),
            preferred_sources=sources,
            priority=i,
            status=TaskStatus.PENDING,
        )
        sub_tasks.append(sub_task)
    
    return sub_tasks


def decompose_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node for task decomposition.
    
    Takes the original query and decomposes it into sub-tasks.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with sub_tasks populated
    """
    logger.info("Starting task decomposition")
    
    # Get LLM client for decompose agent
    llm_client = get_llm_client("decompose")
    
    # Format user message
    user_message = DECOMPOSE_USER_TEMPLATE.format(
        query=state.original_query
    )
    
    try:
        # Call LLM
        response = llm_client.invoke_with_prompt(
            system_prompt=DECOMPOSE_SYSTEM_PROMPT,
            user_message=user_message,
        )
        
        # Parse response
        sub_tasks = parse_decompose_response(response)
        
        if not sub_tasks:
            logger.warning("No sub-tasks generated, creating default task")
            sub_tasks = [
                SubTask(
                    id="task_1",
                    question=state.original_query,
                    description="直接研究原始问题",
                    keywords=[],
                    preferred_sources=[DataSourceType.SEARCH, DataSourceType.RAG],
                    priority=0,
                    status=TaskStatus.PENDING,
                )
            ]
        
        logger.info(
            "Task decomposition completed",
            num_tasks=len(sub_tasks),
            tasks=[t.id for t in sub_tasks]
        )
        
        return {
            "sub_tasks": sub_tasks,
            "workflow_status": "decomposed",
        }
        
    except Exception as e:
        logger.error("Task decomposition failed", error=str(e))
        return {
            "errors": state.errors + [f"Decomposition failed: {str(e)}"],
            "workflow_status": "error",
        }


async def decompose_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Async version of the decompose node.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with sub_tasks populated
    """
    logger.info("Starting async task decomposition")
    
    llm_client = get_llm_client("decompose")
    
    user_message = DECOMPOSE_USER_TEMPLATE.format(
        query=state.original_query
    )
    
    try:
        response = await llm_client.ainvoke_with_prompt(
            system_prompt=DECOMPOSE_SYSTEM_PROMPT,
            user_message=user_message,
        )
        
        sub_tasks = parse_decompose_response(response)
        
        if not sub_tasks:
            logger.warning("No sub-tasks generated, creating default task")
            sub_tasks = [
                SubTask(
                    id="task_1",
                    question=state.original_query,
                    description="直接研究原始问题",
                    keywords=[],
                    preferred_sources=[DataSourceType.SEARCH, DataSourceType.RAG],
                    priority=0,
                    status=TaskStatus.PENDING,
                )
            ]
        
        logger.info(
            "Async task decomposition completed",
            num_tasks=len(sub_tasks),
            tasks=[t.id for t in sub_tasks]
        )
        
        return {
            "sub_tasks": sub_tasks,
            "workflow_status": "decomposed",
        }
        
    except Exception as e:
        logger.error("Async task decomposition failed", error=str(e))
        return {
            "errors": state.errors + [f"Decomposition failed: {str(e)}"],
            "workflow_status": "error",
        }
