"""
LLM Client for Deep Research System.

This module provides a unified interface for interacting with OpenAI-compatible APIs,
supporting different LLM configurations for different agents.

每个Agent可以配置完全不同的LLM API:
- base_url: API端点地址
- api_key: API密钥
- model: 模型名称
- temperature, max_tokens等参数
"""

import os
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config


class LLMConfig(BaseModel):
    """Configuration for an LLM instance.
    
    每个Agent可以有完全独立的配置，包括不同的API提供商。
    """
    
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: str = Field(default="")
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: int = Field(default=60, gt=0)
    
    # 额外配置（用于特殊API）
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"


class LLMClient:
    """
    Unified LLM client supporting OpenAI-compatible APIs.
    
    Features:
    - 每个Agent可使用完全不同的API (base_url, api_key, model)
    - 支持所有OpenAI兼容的第三方API
    - 自动重试机制
    - 结构化输出支持
    
    支持的API提供商示例:
    - OpenAI: https://api.openai.com/v1
    - DeepSeek: https://api.deepseek.com/v1
    - Qwen: https://dashscope.aliyuncs.com/compatible-mode/v1
    - Zhipu: https://open.bigmodel.cn/api/paas/v4
    - Moonshot: https://api.moonshot.cn/v1
    - Ollama: http://localhost:11434/v1
    - OneAPI: http://localhost:3000/v1
    """
    
    def __init__(self, agent_name: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            agent_name: Optional name of the agent to load specific config.
                       每个agent可以有完全不同的LLM配置。
        """
        self._agent_name = agent_name
        self._config = self._load_config()
        self._client: Optional[ChatOpenAI] = None
    
    def _load_config(self) -> LLMConfig:
        """Load LLM configuration from settings.
        
        配置优先级:
        1. Agent特定配置 (settings.yaml -> llm.agents.{agent_name})
        2. 默认配置 (settings.yaml -> llm.default)
        3. 环境变量 (OPENAI_API_KEY等)
        """
        config_loader = get_config()
        raw_config = config_loader.get_llm_config(self._agent_name)
        
        # 如果agent配置中没有api_key，尝试从环境变量获取
        if not raw_config.get("api_key"):
            # 根据base_url推断应该使用哪个环境变量
            base_url = raw_config.get("base_url", "")
            api_key = self._get_api_key_from_env(base_url)
            raw_config["api_key"] = api_key
        
        return LLMConfig(**raw_config)
    
    def _get_api_key_from_env(self, base_url: str) -> str:
        """根据base_url推断并获取对应的API密钥。
        
        Args:
            base_url: API端点地址
            
        Returns:
            对应的API密钥
        """
        # URL到环境变量的映射
        url_to_env = {
            "openai.com": "OPENAI_API_KEY",
            "deepseek.com": "DEEPSEEK_API_KEY",
            "dashscope.aliyuncs.com": "QWEN_API_KEY",
            "bigmodel.cn": "ZHIPU_API_KEY",
            "moonshot.cn": "MOONSHOT_API_KEY",
            "anthropic.com": "ANTHROPIC_API_KEY",
            "localhost": "OPENAI_API_KEY",  # 本地服务默认用OPENAI_API_KEY
        }
        
        for domain, env_var in url_to_env.items():
            if domain in base_url.lower():
                return os.getenv(env_var, "")
        
        # 默认使用OPENAI_API_KEY
        return os.getenv("OPENAI_API_KEY", "")
    
    @property
    def client(self) -> ChatOpenAI:
        """Get or create the ChatOpenAI client."""
        if self._client is None:
            self._client = ChatOpenAI(
                base_url=self._config.base_url,
                api_key=self._config.api_key,
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                timeout=self._config.timeout,
            )
        return self._client
    
    @property
    def model(self) -> BaseChatModel:
        """Get the underlying chat model (alias for client)."""
        return self.client
    
    def reload_config(self) -> None:
        """Reload configuration and recreate client."""
        self._config = self._load_config()
        self._client = None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def ainvoke(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> AIMessage:
        """
        Asynchronously invoke the LLM with messages.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            AI response message
        """
        return await self.client.ainvoke(messages, **kwargs)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> AIMessage:
        """
        Synchronously invoke the LLM with messages.
        
        Args:
            messages: List of messages to send
            **kwargs: Additional arguments to pass to the model
            
        Returns:
            AI response message
        """
        return self.client.invoke(messages, **kwargs)
    
    def invoke_with_prompt(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs: Any
    ) -> str:
        """
        Convenience method to invoke with system prompt and user message.
        
        Args:
            system_prompt: System prompt to use
            user_message: User's message
            **kwargs: Additional arguments
            
        Returns:
            AI response content as string
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self.invoke(messages, **kwargs)
        return response.content
    
    async def ainvoke_with_prompt(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs: Any
    ) -> str:
        """
        Async convenience method to invoke with system prompt and user message.
        
        Args:
            system_prompt: System prompt to use
            user_message: User's message
            **kwargs: Additional arguments
            
        Returns:
            AI response content as string
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = await self.ainvoke(messages, **kwargs)
        return response.content
    
    def invoke_structured(
        self,
        messages: List[BaseMessage],
        output_schema: type[BaseModel],
        **kwargs: Any
    ) -> BaseModel:
        """
        Invoke the LLM with structured output.
        
        Args:
            messages: List of messages to send
            output_schema: Pydantic model for structured output
            **kwargs: Additional arguments
            
        Returns:
            Parsed Pydantic model instance
        """
        structured_llm = self.client.with_structured_output(output_schema)
        return structured_llm.invoke(messages, **kwargs)
    
    async def ainvoke_structured(
        self,
        messages: List[BaseMessage],
        output_schema: type[BaseModel],
        **kwargs: Any
    ) -> BaseModel:
        """
        Async invoke the LLM with structured output.
        
        Args:
            messages: List of messages to send
            output_schema: Pydantic model for structured output
            **kwargs: Additional arguments
            
        Returns:
            Parsed Pydantic model instance
        """
        structured_llm = self.client.with_structured_output(output_schema)
        return await structured_llm.ainvoke(messages, **kwargs)
    
    def bind_tools(self, tools: List[Any]) -> BaseChatModel:
        """
        Bind tools to the LLM for function calling.
        
        Args:
            tools: List of tools to bind
            
        Returns:
            LLM with tools bound
        """
        return self.client.bind_tools(tools)
    
    def get_config(self) -> LLMConfig:
        """Get the current LLM configuration."""
        return self._config


# Cache for LLM clients by agent name
_llm_clients: Dict[Optional[str], LLMClient] = {}


def get_llm_client(agent_name: Optional[str] = None) -> LLMClient:
    """
    Get or create an LLM client for a specific agent.
    
    Args:
        agent_name: Name of the agent (or None for default)
        
    Returns:
        LLMClient instance
    """
    if agent_name not in _llm_clients:
        _llm_clients[agent_name] = LLMClient(agent_name)
    return _llm_clients[agent_name]


def clear_llm_clients() -> None:
    """Clear all cached LLM clients (useful for testing or config reload)."""
    _llm_clients.clear()
