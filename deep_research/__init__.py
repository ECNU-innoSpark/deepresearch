"""
Deep Research System

A LangGraph-based multi-agent system for automated deep research.
Implements the "Roma" workflow pattern with task decomposition,
planning, execution, selection, writing, and review stages.
"""

__version__ = "0.1.0"
__author__ = "Deep Research Team"

from .core.state import GraphState, create_initial_state
from .workflow import compile_workflow, deep_research_workflow

__all__ = [
    "GraphState",
    "create_initial_state",
    "compile_workflow",
    "deep_research_workflow",
]
