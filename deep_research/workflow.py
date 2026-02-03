"""
LangGraph Workflow Definition for Deep Research System.

This module defines the state graph and compiles the workflow
that orchestrates all agents in the research pipeline.
"""

from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.state import GraphState
from core.utils import get_logger
from agents import (
    decompose_node,
    plan_node,
    execution_node,
    selection_node,
    writing_node,
    review_node,
)
from agents.execution import should_continue_execution
from agents.review import review_router

logger = get_logger(__name__)


def create_workflow(use_checkpointer: bool = True) -> StateGraph:
    """
    Create the Deep Research workflow graph.
    
    The workflow follows the "Roma" pattern:
    1. Decompose: Break down the query into sub-tasks
    2. Plan: Organize and prioritize sub-tasks
    3. Execute: Gather data for each sub-task (loops until all done)
    4. Select: Filter and score collected data
    5. Write: Generate the research report
    6. Review: Check quality and decide next steps
    
    Args:
        use_checkpointer: Whether to use memory checkpointing
        
    Returns:
        Compiled StateGraph workflow
    """
    logger.info("Creating Deep Research workflow")
    
    # Create the graph with GraphState
    workflow = StateGraph(GraphState)
    
    # -------------------------------------------------------------------------
    # Add Nodes
    # -------------------------------------------------------------------------
    
    # Task decomposition
    workflow.add_node("decompose", decompose_node)
    
    # Task planning
    workflow.add_node("plan", plan_node)
    
    # Task execution (data gathering)
    workflow.add_node("execute", execution_node)
    
    # Data selection and citation
    workflow.add_node("select", selection_node)
    
    # Report writing
    workflow.add_node("write", writing_node)
    
    # Report review
    workflow.add_node("review", review_node)
    
    # -------------------------------------------------------------------------
    # Define Edges
    # -------------------------------------------------------------------------
    
    # Entry point: Start with decomposition
    workflow.set_entry_point("decompose")
    
    # Decompose -> Plan
    workflow.add_edge("decompose", "plan")
    
    # Plan -> Execute
    workflow.add_edge("plan", "execute")
    
    # Execute -> (conditional) Continue executing or move to selection
    workflow.add_conditional_edges(
        "execute",
        should_continue_execution,
        {
            "continue": "execute",  # Loop back to execute next task
            "done": "select",       # All tasks done, move to selection
        }
    )
    
    # Select -> Write
    workflow.add_edge("select", "write")
    
    # Write -> Review
    workflow.add_edge("write", "review")
    
    # Review -> (conditional) Based on review feedback
    workflow.add_conditional_edges(
        "review",
        review_router,
        {
            "plan": "plan",     # Need more data, go back to planning
            "writing": "write", # Rewrite the report
            "end": END,         # Approved, finish
        }
    )
    
    logger.info("Workflow graph created successfully")
    
    return workflow


def compile_workflow(
    use_checkpointer: bool = True,
    interrupt_before: list = None,
    interrupt_after: list = None,
) -> StateGraph:
    """
    Compile the workflow with optional checkpointing and interrupts.
    
    Args:
        use_checkpointer: Whether to use memory checkpointing
        interrupt_before: List of nodes to interrupt before
        interrupt_after: List of nodes to interrupt after
        
    Returns:
        Compiled workflow graph
    """
    workflow = create_workflow()
    
    # Configure compilation options
    compile_kwargs = {}
    
    if use_checkpointer:
        compile_kwargs["checkpointer"] = MemorySaver()
    
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before
    
    if interrupt_after:
        compile_kwargs["interrupt_after"] = interrupt_after
    
    # Compile the graph
    compiled = workflow.compile(**compile_kwargs)
    
    logger.info(
        "Workflow compiled",
        checkpointer=use_checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after
    )
    
    return compiled


def get_workflow_graph() -> StateGraph:
    """
    Get the compiled workflow graph (singleton pattern).
    
    Returns:
        Compiled workflow graph
    """
    return compile_workflow(use_checkpointer=True)


# Pre-compiled workflow for import
deep_research_workflow = compile_workflow(use_checkpointer=True)


def visualize_workflow() -> str:
    """
    Generate a Mermaid diagram of the workflow.
    
    Returns:
        Mermaid diagram string
    """
    mermaid = """
```mermaid
graph TD
    A[Start] --> B[Decompose]
    B --> C[Plan]
    C --> D[Execute]
    D --> E{All Tasks Done?}
    E -->|No| D
    E -->|Yes| F[Select]
    F --> G[Write]
    G --> H[Review]
    H --> I{Approved?}
    I -->|Need More Data| C
    I -->|Rewrite| G
    I -->|Yes| J[End]
    
    style B fill:#e1f5fe
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style F fill:#fce4ec
    style G fill:#f3e5f5
    style H fill:#e0f2f1
```
"""
    return mermaid


if __name__ == "__main__":
    # Print workflow visualization
    print(visualize_workflow())
    
    # Test workflow creation
    workflow = compile_workflow()
    print("\nWorkflow compiled successfully!")
    print(f"Nodes: {list(workflow.get_graph().nodes.keys())}")
