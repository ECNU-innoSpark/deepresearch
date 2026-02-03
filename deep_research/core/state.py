"""
LangGraph State Definitions for Deep Research System.

This module defines the global state that flows through the LangGraph workflow,
including all data structures for tasks, raw data, selected data, and review feedback.
"""

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class DataSourceType(str, Enum):
    """Type of data source."""
    SEARCH = "search"
    RAG = "rag"
    MCP = "mcp"


class TaskStatus(str, Enum):
    """Status of a sub-task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SubTask(BaseModel):
    """A decomposed sub-task from the original query."""
    
    id: str = Field(..., description="Unique identifier for the sub-task")
    question: str = Field(..., description="The sub-question to investigate")
    description: Optional[str] = Field(None, description="Additional context or description")
    priority: int = Field(default=0, description="Priority order (lower = higher priority)")
    dependencies: List[str] = Field(default_factory=list, description="IDs of dependent sub-tasks")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    preferred_sources: List[DataSourceType] = Field(
        default_factory=lambda: [DataSourceType.SEARCH, DataSourceType.RAG],
        description="Preferred data sources for this task"
    )


class RawDataItem(BaseModel):
    """Raw data item collected from various sources."""
    
    id: str = Field(..., description="Unique identifier for the data item")
    task_id: str = Field(..., description="ID of the sub-task this data relates to")
    source_type: DataSourceType = Field(..., description="Type of source (search/rag/mcp)")
    
    # Content
    content: str = Field(..., description="Main content/snippet")
    title: Optional[str] = Field(None, description="Title of the source")
    url: Optional[str] = Field(None, description="URL of the source")
    
    # Metadata
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    retrieved_at: datetime = Field(default_factory=datetime.now, description="Retrieval timestamp")


class SelectedDataItem(BaseModel):
    """Data item that has been selected and assigned a citation ID."""
    
    # Inherit from raw data
    raw_data_id: str = Field(..., description="Reference to original raw data item ID")
    task_id: str = Field(..., description="ID of the sub-task this data relates to")
    source_type: DataSourceType = Field(..., description="Type of source")
    
    # Content (may be processed/cleaned)
    content: str = Field(..., description="Content for citation")
    title: Optional[str] = Field(None, description="Title of the source")
    url: Optional[str] = Field(None, description="URL of the source")
    
    # Citation
    citation_id: str = Field(..., description="Unique citation ID (e.g., ref_1)")
    citation_key: str = Field(..., description="Citation key for in-text reference (e.g., [1])")
    
    # Scores
    relevance_score: float = Field(..., description="Relevance score")
    confidence_score: float = Field(..., description="Confidence score")
    
    # Selection reasoning
    selection_reason: Optional[str] = Field(None, description="Why this item was selected")


class ReviewFeedback(BaseModel):
    """Feedback from the review agent."""
    
    is_approved: bool = Field(..., description="Whether the report is approved")
    
    # Quality metrics
    format_check: bool = Field(default=True, description="Is Markdown format correct")
    citation_check: bool = Field(default=True, description="Are citations properly formatted")
    content_accuracy: bool = Field(default=True, description="Is content accurate (no hallucinations)")
    citation_count_check: bool = Field(default=True, description="Are there enough citations")
    
    # Counts
    total_citations: int = Field(default=0, description="Total number of citations used")
    search_citations: int = Field(default=0, description="Number of search-based citations")
    rag_citations: int = Field(default=0, description="Number of RAG-based citations")
    
    # Feedback details
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    
    # Routing decision
    route_to: Optional[Literal["plan", "writing", "end"]] = Field(
        None, 
        description="Where to route: 'plan' for more data, 'writing' for rewrite, 'end' if approved"
    )


class GraphState(BaseModel):
    """
    Global state for the Deep Research LangGraph workflow.
    
    This state is passed through all nodes in the graph and contains
    all necessary data for the research pipeline.
    """
    
    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------
    original_query: str = Field(..., description="User's original research question")
    
    # -------------------------------------------------------------------------
    # Task Decomposition & Planning
    # -------------------------------------------------------------------------
    sub_tasks: List[SubTask] = Field(
        default_factory=list, 
        description="Decomposed sub-tasks (max 5)"
    )
    task_plan: List[str] = Field(
        default_factory=list, 
        description="Ordered list of sub-task IDs representing execution plan"
    )
    current_task_index: int = Field(
        default=0, 
        description="Index of current task being processed"
    )
    
    # -------------------------------------------------------------------------
    # Data Collection
    # -------------------------------------------------------------------------
    raw_data: List[RawDataItem] = Field(
        default_factory=list, 
        description="All raw data collected from various sources"
    )
    
    # -------------------------------------------------------------------------
    # Data Selection
    # -------------------------------------------------------------------------
    selected_data: List[SelectedDataItem] = Field(
        default_factory=list, 
        description="Filtered and citation-assigned data items"
    )
    
    # -------------------------------------------------------------------------
    # Writing & Review
    # -------------------------------------------------------------------------
    draft_report: str = Field(
        default="", 
        description="Current draft of the research report"
    )
    final_report: str = Field(
        default="", 
        description="Final approved research report"
    )
    references_section: str = Field(
        default="", 
        description="Formatted references/bibliography section"
    )
    
    # -------------------------------------------------------------------------
    # Review & Iteration
    # -------------------------------------------------------------------------
    review_feedback: Optional[ReviewFeedback] = Field(
        default=None, 
        description="Feedback from the review agent"
    )
    revision_count: int = Field(
        default=0, 
        description="Number of revision cycles completed"
    )
    
    # -------------------------------------------------------------------------
    # Metadata & Control
    # -------------------------------------------------------------------------
    errors: List[str] = Field(
        default_factory=list, 
        description="Error messages encountered during processing"
    )
    workflow_status: str = Field(
        default="initialized", 
        description="Current workflow status"
    )
    
    # Messages for agent communication (LangGraph standard)
    messages: Annotated[List[Any], add_messages] = Field(
        default_factory=list,
        description="Message history for agent communication"
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


def create_initial_state(query: str) -> GraphState:
    """
    Create an initial GraphState from a user query.
    
    Args:
        query: The user's research question
        
    Returns:
        Initialized GraphState ready for the workflow
    """
    return GraphState(
        original_query=query,
        workflow_status="initialized"
    )
