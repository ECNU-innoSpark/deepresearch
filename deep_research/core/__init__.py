"""
Core module for Deep Research System.
Contains state definitions, LLM client, citation management, and utilities.
"""

from .state import GraphState, SubTask, RawDataItem, SelectedDataItem, ReviewFeedback
from .llm_client import LLMClient, get_llm_client
from .citation_manager import CitationManager, Citation
from .utils import setup_logging, get_logger

__all__ = [
    "GraphState",
    "SubTask",
    "RawDataItem",
    "SelectedDataItem",
    "ReviewFeedback",
    "LLMClient",
    "get_llm_client",
    "CitationManager",
    "Citation",
    "setup_logging",
    "get_logger",
]
