"""
Configuration module for Deep Research System.
Provides YAML configuration loading and validation.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Loads and manages YAML configuration files with environment variable resolution."""
    
    _instance: Optional["ConfigLoader"] = None
    _settings: Dict[str, Any] = {}
    _tools_config: Dict[str, Any] = {}
    
    def __new__(cls) -> "ConfigLoader":
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config_dir = Path(__file__).parent
        self._load_configs()
    
    def _resolve_env_vars(self, value: Any) -> Any:
        """Recursively resolve environment variables in config values.
        
        Supports ${VAR_NAME} syntax for environment variable substitution.
        """
        if isinstance(value, str):
            # Pattern to match ${VAR_NAME}
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, value)
            for var_name in matches:
                env_value = os.getenv(var_name, "")
                value = value.replace(f"${{{var_name}}}", env_value)
            return value
        elif isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]
        return value
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML file and resolve environment variables."""
        filepath = self._config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        
        return self._resolve_env_vars(config)
    
    def _load_configs(self) -> None:
        """Load all configuration files."""
        try:
            self._settings = self._load_yaml("settings.yaml")
        except FileNotFoundError:
            self._settings = {}
        
        try:
            self._tools_config = self._load_yaml("tools_config.yaml")
        except FileNotFoundError:
            self._tools_config = {}
    
    def reload(self) -> None:
        """Reload all configuration files."""
        self._load_configs()
    
    @property
    def settings(self) -> Dict[str, Any]:
        """Get global settings configuration."""
        return self._settings
    
    @property
    def tools(self) -> Dict[str, Any]:
        """Get tools configuration."""
        return self._tools_config
    
    def get_llm_config(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get LLM configuration for a specific agent or default.
        
        Args:
            agent_name: Name of the agent (decompose, plan, execution, etc.)
            
        Returns:
            LLM configuration dictionary with merged default and agent-specific settings.
        """
        llm_config = self._settings.get("llm", {})
        default_config = llm_config.get("default", {}).copy()
        
        if agent_name:
            agents_config = llm_config.get("agents", {})
            agent_config = agents_config.get(agent_name, {})
            default_config.update(agent_config)
        
        return default_config
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration."""
        return self._settings.get("workflow", {})
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search provider configuration."""
        search_config = self._tools_config.get("search", {})
        provider = search_config.get("provider", "tavily")
        provider_config = search_config.get(provider, {})
        return {"provider": provider, **provider_config}
    
    def get_ragflow_config(self) -> Dict[str, Any]:
        """Get RAGFlow configuration."""
        return self._tools_config.get("ragflow", {})
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """Get MCP configuration."""
        return self._tools_config.get("mcp", {})


# Global config instance
config = ConfigLoader()


def get_config() -> ConfigLoader:
    """Get the global configuration instance."""
    return config
