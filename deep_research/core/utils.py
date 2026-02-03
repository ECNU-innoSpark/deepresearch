"""
Utility functions for Deep Research System.

This module provides common utilities including logging setup,
text processing, and helper functions.
"""

import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import sys

import structlog
from rich.console import Console
from rich.logging import RichHandler


# Rich console for pretty output
console = Console()


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_structured: bool = True,
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        use_structured: Whether to use structured logging (structlog)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create handlers
    handlers: List[logging.Handler] = []
    
    # Rich console handler for pretty terminal output
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(log_level)
    handlers.append(rich_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,
    )
    
    if use_structured:
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part
        
    Returns:
        Unique ID string
    """
    import uuid
    random_part = uuid.uuid4().hex[:length]
    if prefix:
        return f"{prefix}_{random_part}"
    return random_part


def hash_content(content: str) -> str:
    """
    Generate a hash for content deduplication.
    
    Args:
        content: Content to hash
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(content.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Text to search
        
    Returns:
        List of URLs found
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format a datetime as ISO string.
    
    Args:
        dt: Datetime to format (defaults to now)
        
    Returns:
        ISO formatted timestamp
    """
    if dt is None:
        dt = datetime.now()
    return dt.isoformat()


def safe_json_loads(text: str, default: Any = None) -> Any:
    """
    Safely parse JSON with fallback.
    
    Args:
        text: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Safely serialize to JSON.
    
    Args:
        obj: Object to serialize
        indent: Indentation level
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_file(path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Read a file's contents.
    
    Args:
        path: File path
        encoding: File encoding
        
    Returns:
        File contents
    """
    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def write_file(
    path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    mkdir: bool = True,
) -> None:
    """
    Write content to a file.
    
    Args:
        path: File path
        content: Content to write
        encoding: File encoding
        mkdir: Whether to create parent directories
    """
    path = Path(path)
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries (later values override earlier).
    
    Args:
        *dicts: Dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count (rough approximation).
    
    Uses a simple heuristic: ~4 characters per token for English,
    ~2 characters per token for Chinese.
    
    Args:
        text: Text to count
        
    Returns:
        Estimated token count
    """
    # Count Chinese characters
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    other_chars = len(text) - chinese_chars
    
    # Estimate: Chinese ~2 chars/token, Other ~4 chars/token
    return (chinese_chars // 2) + (other_chars // 4)


def print_banner(title: str, width: int = 60) -> None:
    """
    Print a formatted banner to console.
    
    Args:
        title: Banner title
        width: Banner width
    """
    console.print("=" * width, style="bold blue")
    console.print(f"{title:^{width}}", style="bold white")
    console.print("=" * width, style="bold blue")


def print_step(step_name: str, status: str = "running") -> None:
    """
    Print a workflow step status.
    
    Args:
        step_name: Name of the step
        status: Status (running, done, error)
    """
    status_colors = {
        "running": "yellow",
        "done": "green",
        "error": "red",
    }
    color = status_colors.get(status, "white")
    console.print(f"[{color}]● {step_name}[/{color}] - {status}")


def format_state_summary(state: Dict[str, Any], max_preview_len: int = 200) -> str:
    """
    将当前 GraphState 格式化为可读的中间结果摘要。
    
    Args:
        state: 状态字典 (来自 workflow.get_state().values 或 state_update)
        max_preview_len: 文本预览最大长度
        
    Returns:
        多行摘要字符串
    """
    lines = []
    
    # 子任务
    sub_tasks = state.get("sub_tasks") or []
    if sub_tasks:
        lines.append("  [子任务]")
        for i, t in enumerate(sub_tasks[:5], 1):
            if isinstance(t, dict):
                q = (t.get("question") or "")[:60]
                tid = t.get("id", "")
            else:
                q = (getattr(t, "question", "") or "")[:60]
                tid = getattr(t, "id", "")
            lines.append(f"    {i}. [{tid}] {q}...")
        if len(sub_tasks) > 5:
            lines.append(f"    ... 共 {len(sub_tasks)} 个子任务")
    
    # 执行计划
    task_plan = state.get("task_plan") or []
    if task_plan:
        lines.append(f"  [执行计划] {task_plan}")
    
    # 当前执行到第几个任务
    current_idx = state.get("current_task_index", 0)
    if task_plan:
        lines.append(f"  [当前任务索引] {current_idx + 1}/{len(task_plan)}")
    
    # 原始数据
    raw_data = state.get("raw_data") or []
    if raw_data:
        search_count = sum(1 for r in raw_data if _get_attr(r, "source_type") == "search")
        rag_count = sum(1 for r in raw_data if _get_attr(r, "source_type") == "rag")
        lines.append(f"  [原始数据] 共 {len(raw_data)} 条 (搜索: {search_count}, RAG: {rag_count})")
    
    # 筛选数据
    selected_data = state.get("selected_data") or []
    if selected_data:
        lines.append(f"  [筛选数据] 共 {len(selected_data)} 条")
    
    # 草稿预览
    draft = state.get("draft_report") or ""
    if draft:
        preview = draft[:max_preview_len].replace("\n", " ")
        if len(draft) > max_preview_len:
            preview += "..."
        lines.append(f"  [报告草稿] {len(draft)} 字")
        lines.append(f"    预览: {preview}")
    
    # 审查结果
    review = state.get("review_feedback")
    if review is not None:
        if isinstance(review, dict):
            approved = review.get("is_approved", False)
            route = review.get("route_to", "")
            issues = review.get("issues", [])
        else:
            approved = getattr(review, "is_approved", False)
            route = getattr(review, "route_to", "")
            issues = getattr(review, "issues", [])
        lines.append(f"  [审查] 通过={approved}, 路由={route}")
        if issues:
            for issue in issues[:3]:
                lines.append(f"    问题: {issue[:80]}...")
    
    # 错误
    errors = state.get("errors") or []
    if errors:
        lines.append(f"  [错误] {len(errors)} 条")
        for e in errors[-3:]:
            lines.append(f"    - {str(e)[:80]}")
    
    return "\n".join(lines) if lines else "  (无中间数据)"


def _get_attr(obj: Any, key: str) -> Any:
    """从对象或字典获取属性。"""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def print_state_summary(state: Dict[str, Any], title: str = "当前状态摘要") -> None:
    """
    打印当前状态的摘要到控制台。
    
    Args:
        state: 状态字典
        title: 标题
    """
    console.print(f"\n[bold cyan]━━━ {title} ━━━[/bold cyan]")
    console.print(format_state_summary(state))
    console.print("[dim]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n")
