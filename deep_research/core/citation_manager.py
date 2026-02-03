"""
Citation Manager for Deep Research System.

This module handles all citation-related operations including:
- Assigning unique citation IDs
- Maintaining ID -> source mappings
- Generating formatted reference lists
- Supporting multiple citation styles
"""

from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field


class CitationStyle(str, Enum):
    """Supported citation styles."""
    NUMERIC = "numeric"      # [1], [2], [3]
    ALPHA = "alpha"          # [ref_1], [ref_2]
    AUTHOR_YEAR = "author_year"  # [Author, Year]


class Citation(BaseModel):
    """A single citation entry."""
    
    id: str = Field(..., description="Unique citation ID (e.g., ref_1)")
    numeric_id: int = Field(..., description="Numeric ID for ordering")
    
    # Source information
    title: Optional[str] = Field(None, description="Title of the source")
    url: Optional[str] = Field(None, description="URL of the source")
    content_snippet: Optional[str] = Field(None, description="Brief content snippet")
    
    # Metadata
    source_type: str = Field(default="web", description="Type of source (search/rag/mcp)")
    author: Optional[str] = Field(None, description="Author(s) if available")
    date: Optional[str] = Field(None, description="Publication date if available")
    
    # Quality metrics
    relevance_score: float = Field(default=0.0, description="Relevance score")
    confidence_score: float = Field(default=0.0, description="Confidence score")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_citation_key(self, style: CitationStyle = CitationStyle.NUMERIC) -> str:
        """Get the in-text citation key based on style."""
        if style == CitationStyle.NUMERIC:
            return f"[{self.numeric_id}]"
        elif style == CitationStyle.ALPHA:
            return f"[{self.id}]"
        elif style == CitationStyle.AUTHOR_YEAR:
            author = self.author or "Unknown"
            year = self.date[:4] if self.date else "n.d."
            return f"[{author}, {year}]"
        return f"[{self.numeric_id}]"
    
    def format_reference(self, style: CitationStyle = CitationStyle.NUMERIC) -> str:
        """Format the citation as a reference entry."""
        parts = []
        
        # Number/ID
        if style == CitationStyle.NUMERIC:
            parts.append(f"[{self.numeric_id}]")
        else:
            parts.append(f"[{self.id}]")
        
        # Title
        if self.title:
            parts.append(f'"{self.title}"')
        
        # Author and date
        if self.author:
            parts.append(f"by {self.author}")
        if self.date:
            parts.append(f"({self.date})")
        
        # URL
        if self.url:
            parts.append(f"URL: {self.url}")
        
        # Source type
        parts.append(f"[Source: {self.source_type.upper()}]")
        
        return " ".join(parts)


class CitationManager:
    """
    Manages citations throughout the research workflow.
    
    This class maintains a registry of all citations, assigns unique IDs,
    and generates formatted reference lists.
    """
    
    _instance: Optional["CitationManager"] = None
    
    def __new__(cls) -> "CitationManager":
        """Singleton pattern for global citation management."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, style: CitationStyle = CitationStyle.NUMERIC):
        if self._initialized:
            return
        self._initialized = True
        self._style = style
        self._citations: Dict[str, Citation] = {}
        self._counter = 0
        self._url_to_id: Dict[str, str] = {}  # For deduplication
    
    @classmethod
    def get_instance(cls) -> "CitationManager":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        if cls._instance is not None:
            cls._instance._citations.clear()
            cls._instance._counter = 0
            cls._instance._url_to_id.clear()
    
    def set_style(self, style: CitationStyle) -> None:
        """Set the citation style."""
        self._style = style
    
    def add_citation(
        self,
        title: Optional[str] = None,
        url: Optional[str] = None,
        content_snippet: Optional[str] = None,
        source_type: str = "search",
        author: Optional[str] = None,
        date: Optional[str] = None,
        relevance_score: float = 0.0,
        confidence_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Citation:
        """
        Add a new citation and return it with assigned ID.
        
        If a citation with the same URL already exists, returns the existing one.
        
        Args:
            title: Title of the source
            url: URL of the source
            content_snippet: Brief content snippet
            source_type: Type of source (search/rag/mcp)
            author: Author(s) if available
            date: Publication date
            relevance_score: Relevance score
            confidence_score: Confidence score
            metadata: Additional metadata
            
        Returns:
            Citation object with assigned ID
        """
        # Check for duplicate by URL
        if url and url in self._url_to_id:
            return self._citations[self._url_to_id[url]]
        
        # Create new citation
        self._counter += 1
        citation_id = f"ref_{self._counter}"
        
        citation = Citation(
            id=citation_id,
            numeric_id=self._counter,
            title=title,
            url=url,
            content_snippet=content_snippet,
            source_type=source_type,
            author=author,
            date=date,
            relevance_score=relevance_score,
            confidence_score=confidence_score,
            metadata=metadata or {},
        )
        
        self._citations[citation_id] = citation
        if url:
            self._url_to_id[url] = citation_id
        
        return citation
    
    def get_citation(self, citation_id: str) -> Optional[Citation]:
        """Get a citation by ID."""
        return self._citations.get(citation_id)
    
    def get_citation_by_url(self, url: str) -> Optional[Citation]:
        """Get a citation by URL."""
        citation_id = self._url_to_id.get(url)
        if citation_id:
            return self._citations.get(citation_id)
        return None
    
    def get_all_citations(self) -> List[Citation]:
        """Get all citations sorted by numeric ID."""
        return sorted(self._citations.values(), key=lambda c: c.numeric_id)
    
    def get_citation_key(self, citation_id: str) -> str:
        """Get the in-text citation key for a citation ID."""
        citation = self._citations.get(citation_id)
        if citation:
            return citation.get_citation_key(self._style)
        return f"[{citation_id}]"
    
    def get_citation_count(self) -> int:
        """Get the total number of citations."""
        return len(self._citations)
    
    def get_citation_count_by_type(self, source_type: str) -> int:
        """Get the count of citations by source type."""
        return sum(1 for c in self._citations.values() if c.source_type == source_type)
    
    def generate_references_section(self, title: str = "参考文献") -> str:
        """
        Generate the formatted references section for the report.
        
        Args:
            title: Title for the references section
            
        Returns:
            Formatted Markdown references section
        """
        citations = self.get_all_citations()
        if not citations:
            return ""
        
        lines = [
            f"\n## {title}\n",
        ]
        
        for citation in citations:
            lines.append(citation.format_reference(self._style))
        
        return "\n".join(lines)
    
    def generate_references_markdown(self) -> str:
        """
        Generate references in Markdown footnote format.
        
        Returns:
            Formatted Markdown footnotes
        """
        citations = self.get_all_citations()
        if not citations:
            return ""
        
        lines = ["\n---\n"]
        
        for citation in citations:
            # Markdown footnote format: [^ref_1]: Reference text
            ref_text = []
            if citation.title:
                ref_text.append(f"**{citation.title}**")
            if citation.url:
                ref_text.append(f"[链接]({citation.url})")
            ref_text.append(f"_[来源: {citation.source_type.upper()}]_")
            
            lines.append(f"[^{citation.id}]: {' - '.join(ref_text)}")
        
        return "\n".join(lines)
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all citations to a dictionary."""
        return {
            "total_count": self.get_citation_count(),
            "by_type": {
                "search": self.get_citation_count_by_type("search"),
                "rag": self.get_citation_count_by_type("rag"),
                "mcp": self.get_citation_count_by_type("mcp"),
            },
            "citations": [c.model_dump() for c in self.get_all_citations()],
        }
    
    def clear(self) -> None:
        """Clear all citations."""
        self._citations.clear()
        self._url_to_id.clear()
        self._counter = 0


def get_citation_manager() -> CitationManager:
    """Get the global CitationManager instance."""
    return CitationManager.get_instance()
