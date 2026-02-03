"""
Planning Agent.

Responsible for organizing and prioritizing sub-tasks,
identifying dependencies, and creating an optimal execution plan.
"""

import json
from typing import Any, Dict, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.state import GraphState, SubTask
from core.llm_client import get_llm_client
from core.utils import get_logger, safe_json_loads
from .prompts import PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE

logger = get_logger(__name__)


def format_sub_tasks_for_planning(sub_tasks: List[SubTask]) -> str:
    """Format sub-tasks as a string for the planning prompt."""
    lines = []
    for task in sub_tasks:
        sources = [s.value for s in task.preferred_sources]
        lines.append(f"""
- **ID**: {task.id}
  - 问题: {task.question}
  - 描述: {task.description}
  - 关键词: {', '.join(task.keywords)}
  - 建议来源: {', '.join(sources)}
""")
    return "\n".join(lines)


def parse_plan_response(response: str, sub_tasks: List[SubTask]) -> Dict[str, Any]:
    """
    Parse the LLM response into a task plan.
    
    Args:
        response: Raw LLM response string
        sub_tasks: List of available sub-tasks
        
    Returns:
        Dictionary with task_plan and dependencies
    """
    # Try to extract JSON from response
    try:
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
        logger.warning("Failed to parse plan response, using default order", error=str(e))
        # Default to the order they came in
        return {
            "task_plan": [t.id for t in sub_tasks],
            "dependencies": {},
            "parallel_groups": [],
            "reasoning": "使用默认顺序（解析失败）",
        }
    
    # Validate task IDs
    valid_ids = {t.id for t in sub_tasks}
    task_plan = data.get("task_plan", [])
    validated_plan = [tid for tid in task_plan if tid in valid_ids]
    
    # Add any missing tasks at the end
    for task in sub_tasks:
        if task.id not in validated_plan:
            validated_plan.append(task.id)
    
    return {
        "task_plan": validated_plan,
        "dependencies": data.get("dependencies") or {},
        "parallel_groups": data.get("parallel_groups") or [],
        "reasoning": data.get("reasoning") or "",  # 确保不是 None
    }


def update_task_dependencies(
    sub_tasks: List[SubTask],
    dependencies: Dict[str, List[str]]
) -> List[SubTask]:
    """
    Update sub-tasks with dependency information.
    
    Args:
        sub_tasks: List of sub-tasks
        dependencies: Dependency mapping
        
    Returns:
        Updated sub-tasks
    """
    task_map = {t.id: t for t in sub_tasks}
    
    for task_id, deps in dependencies.items():
        if task_id in task_map:
            task_map[task_id].dependencies = deps
    
    return list(task_map.values())


def plan_node(state: GraphState) -> Dict[str, Any]:
    """
    LangGraph node for task planning.
    
    Organizes sub-tasks into an optimal execution order.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with task_plan populated
    """
    logger.info("Starting task planning", num_tasks=len(state.sub_tasks))
    
    if not state.sub_tasks:
        logger.warning("No sub-tasks to plan")
        return {
            "task_plan": [],
            "workflow_status": "planned",
        }
    
    # Get LLM client for plan agent
    llm_client = get_llm_client("plan")
    
    # Format sub-tasks for prompt
    sub_tasks_str = format_sub_tasks_for_planning(state.sub_tasks)
    
    user_message = PLAN_USER_TEMPLATE.format(
        original_query=state.original_query,
        sub_tasks=sub_tasks_str,
    )
    
    try:
        # Call LLM
        logger.debug("Calling LLM for planning...")
        response = llm_client.invoke_with_prompt(
            system_prompt=PLAN_SYSTEM_PROMPT,
            user_message=user_message,
        )
        logger.debug(f"LLM response received, length={len(response)}")
        
        # Parse response
        logger.debug("Parsing plan response...")
        plan_result = parse_plan_response(response, state.sub_tasks)
        logger.debug(f"Plan parsed: {plan_result.get('task_plan', [])}")
        
        # Update tasks with dependencies
        logger.debug("Updating task dependencies...")
        updated_tasks = update_task_dependencies(
            state.sub_tasks,
            plan_result.get("dependencies", {})
        )
        logger.debug(f"Dependencies updated, {len(updated_tasks)} tasks")
        
        result = {
            "sub_tasks": updated_tasks,
            "task_plan": plan_result["task_plan"],
            "current_task_index": 0,
            "workflow_status": "planned",
        }
        # 避免结构化日志在某些终端环境中卡住
        logger.info("Task planning completed")
        logger.debug(f"plan_node returning: task_plan={result['task_plan']}")
        return result
        
    except Exception as e:
        logger.error("Task planning failed", error=str(e), exc_info=True)
        # Fallback to default order
        default_plan = [t.id for t in state.sub_tasks]
        return {
            "task_plan": default_plan,
            "current_task_index": 0,
            "errors": state.errors + [f"Planning failed, using default order: {str(e)}"],
            "workflow_status": "planned",
        }


async def plan_node_async(state: GraphState) -> Dict[str, Any]:
    """
    Async version of the plan node.
    
    Args:
        state: Current graph state
        
    Returns:
        State updates with task_plan populated
    """
    logger.info("Starting async task planning", num_tasks=len(state.sub_tasks))
    
    if not state.sub_tasks:
        logger.warning("No sub-tasks to plan")
        return {
            "task_plan": [],
            "workflow_status": "planned",
        }
    
    llm_client = get_llm_client("plan")
    sub_tasks_str = format_sub_tasks_for_planning(state.sub_tasks)
    
    user_message = PLAN_USER_TEMPLATE.format(
        original_query=state.original_query,
        sub_tasks=sub_tasks_str,
    )
    
    try:
        response = await llm_client.ainvoke_with_prompt(
            system_prompt=PLAN_SYSTEM_PROMPT,
            user_message=user_message,
        )
        
        plan_result = parse_plan_response(response, state.sub_tasks)
        updated_tasks = update_task_dependencies(
            state.sub_tasks,
            plan_result.get("dependencies", {})
        )
        
        logger.info(
            "Async task planning completed",
            plan=plan_result["task_plan"]
        )
        
        return {
            "sub_tasks": updated_tasks,
            "task_plan": plan_result["task_plan"],
            "current_task_index": 0,
            "workflow_status": "planned",
        }
        
    except Exception as e:
        logger.error("Async task planning failed", error=str(e))
        default_plan = [t.id for t in state.sub_tasks]
        return {
            "task_plan": default_plan,
            "current_task_index": 0,
            "errors": state.errors + [f"Planning failed, using default order: {str(e)}"],
            "workflow_status": "planned",
        }
