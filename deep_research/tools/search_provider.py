"""
Search Provider for Deep Research System.

This module provides a unified interface for web search APIs
including Tavily, Google Custom Search, and Bing Search.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from core.utils import get_logger, generate_id

logger = get_logger(__name__)


class SearchResult(BaseModel):
    """A single search result."""
    
    id: str = Field(default_factory=lambda: generate_id("search"))
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Content snippet")
    content: Optional[str] = Field(None, description="Full content if available")
    
    # Scores
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Metadata
    source: str = Field(default="web", description="Source name")
    published_date: Optional[str] = Field(None, description="Publication date")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=datetime.now)


class SearchProviderType(str, Enum):
    """Supported search providers."""
    TAVILY = "tavily"
    GOOGLE = "google"
    BING = "bing"


class BaseSearchProvider(ABC):
    """Abstract base class for search providers."""
    
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute a search query."""
        pass
    
    @abstractmethod
    def search_sync(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute a synchronous search query."""
        pass


class TavilySearchProvider(BaseSearchProvider):
    """Tavily Search API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.tavily.com")
        self.search_depth = config.get("search_depth", "advanced")
        self.max_results = config.get("max_results", 10)
        self.include_answer = config.get("include_answer", True)
        self.include_raw_content = config.get("include_raw_content", False)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute async Tavily search."""
        max_results = kwargs.get("max_results", self.max_results)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": self.search_depth,
                    "max_results": max_results,
                    "include_answer": self.include_answer,
                    "include_raw_content": self.include_raw_content,
                },
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                content=item.get("raw_content"),
                relevance_score=item.get("score", 0.5),
                source="tavily",
                metadata={"answer": data.get("answer")},
            ))
        
        logger.info("Tavily search completed", results_count=len(results))
        return results
    
    def search_sync(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute synchronous Tavily search."""
        return asyncio.get_event_loop().run_until_complete(self.search(query, **kwargs))


class GoogleSearchProvider(BaseSearchProvider):
    """Google Custom Search API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key", "")
        self.cx = config.get("cx", "")
        self.base_url = config.get("base_url", "https://www.googleapis.com/customsearch/v1")
        self.max_results = config.get("max_results", 10)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute async Google Custom Search."""
        max_results = kwargs.get("max_results", self.max_results)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self.base_url,
                params={
                    "key": self.api_key,
                    "cx": self.cx,
                    "q": query,
                    "num": min(max_results, 10),  # Google API max is 10
                },
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("items", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="google",
                metadata={
                    "displayLink": item.get("displayLink"),
                    "formattedUrl": item.get("formattedUrl"),
                },
            ))
        
        logger.info("Google search completed", results_count=len(results))
        return results
    
    def search_sync(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute synchronous Google search."""
        return asyncio.get_event_loop().run_until_complete(self.search(query, **kwargs))


class BingSearchProvider(BaseSearchProvider):
    """Bing Search API provider."""
    
    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.bing.microsoft.com/v7.0/search")
        self.max_results = config.get("max_results", 10)
        self.market = config.get("market", "en-US")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute async Bing search."""
        max_results = kwargs.get("max_results", self.max_results)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self.base_url,
                params={
                    "q": query,
                    "count": max_results,
                    "mkt": self.market,
                },
                headers={"Ocp-Apim-Subscription-Key": self.api_key},
            )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for item in data.get("webPages", {}).get("value", []):
            results.append(SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
                source="bing",
                published_date=item.get("dateLastCrawled"),
                metadata={
                    "displayUrl": item.get("displayUrl"),
                    "language": item.get("language"),
                },
            ))
        
        logger.info("Bing search completed", results_count=len(results))
        return results
    
    def search_sync(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute synchronous Bing search."""
        return asyncio.get_event_loop().run_until_complete(self.search(query, **kwargs))


class MockSearchProvider(BaseSearchProvider):
    """Mock search provider for testing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Return mock search results."""
        logger.debug("Mock search executed")
        return [
            SearchResult(
                title=f"Mock Result 1 for: {query}",
                url=f"https://example.com/result1?q={query.replace(' ', '+')}",
                snippet=f"This is a mock search result snippet for the query: {query}. "
                        "It contains relevant information that would typically come from a web search.",
                relevance_score=0.85,
                source="mock",
            ),
            SearchResult(
                title=f"Mock Result 2 for: {query}",
                url=f"https://example.com/result2?q={query.replace(' ', '+')}",
                snippet=f"Another mock result for: {query}. "
                        "This demonstrates multiple results being returned.",
                relevance_score=0.75,
                source="mock",
            ),
            SearchResult(
                title=f"Mock Result 3 for: {query}",
                url=f"https://example.com/result3?q={query.replace(' ', '+')}",
                snippet=f"Third mock result about {query}. "
                        "Contains additional perspective on the topic.",
                relevance_score=0.65,
                source="mock",
            ),
        ]
    
    def search_sync(self, query: str, **kwargs) -> List[SearchResult]:
        """Return mock search results synchronously."""
        return asyncio.get_event_loop().run_until_complete(self.search(query, **kwargs))


class SearchProvider:
    """
    Unified search provider that delegates to specific implementations.
    
    This class serves as a factory and unified interface for all search providers.
    """
    
    def __init__(self, provider_type: Optional[str] = None, use_mock: bool = False):
        """
        Initialize the search provider.
        
        Args:
            provider_type: Type of provider (tavily, google, bing)
            use_mock: Whether to use mock provider for testing
        """
        self._config = get_config().get_search_config()
        self._provider_type = provider_type or self._config.get("provider", "tavily")
        self._use_mock = use_mock
        self._provider: Optional[BaseSearchProvider] = None
    
    @property
    def provider(self) -> BaseSearchProvider:
        """Get or create the underlying search provider."""
        if self._provider is None:
            if self._use_mock:
                self._provider = MockSearchProvider(self._config)
            elif self._provider_type == SearchProviderType.TAVILY:
                self._provider = TavilySearchProvider(self._config)
            elif self._provider_type == SearchProviderType.GOOGLE:
                self._provider = GoogleSearchProvider(self._config)
            elif self._provider_type == SearchProviderType.BING:
                self._provider = BingSearchProvider(self._config)
            else:
                logger.warning(f"Unknown provider type: {self._provider_type}, using mock")
                self._provider = MockSearchProvider(self._config)
        return self._provider
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute an async search query."""
        return await self.provider.search(query, **kwargs)
    
    def search_sync(self, query: str, **kwargs) -> List[SearchResult]:
        """Execute a synchronous search query."""
        return self.provider.search_sync(query, **kwargs)


# Global provider instance
_search_provider: Optional[SearchProvider] = None


def get_search_provider(use_mock: bool = False) -> SearchProvider:
    """
    Get the global search provider instance.
    
    Args:
        use_mock: Whether to use mock provider
        
    Returns:
        SearchProvider instance
    """
    global _search_provider
    if _search_provider is None or _search_provider._use_mock != use_mock:
        _search_provider = SearchProvider(use_mock=use_mock)
    return _search_provider
