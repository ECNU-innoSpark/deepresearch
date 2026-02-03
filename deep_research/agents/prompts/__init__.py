"""
Prompts module for Deep Research agents.

All system prompts are stored here for easy management and fine-tuning.
Prompts should NOT be hardcoded in agent logic files.
"""

from .decompose_prompts import DECOMPOSE_SYSTEM_PROMPT, DECOMPOSE_USER_TEMPLATE
from .plan_prompts import PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE
from .execution_prompts import EXECUTION_SYSTEM_PROMPT, EXECUTION_USER_TEMPLATE
from .selection_prompts import SELECTION_SYSTEM_PROMPT, SELECTION_USER_TEMPLATE
from .writing_prompts import WRITING_SYSTEM_PROMPT, WRITING_USER_TEMPLATE
from .review_prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_TEMPLATE

__all__ = [
    "DECOMPOSE_SYSTEM_PROMPT",
    "DECOMPOSE_USER_TEMPLATE",
    "PLAN_SYSTEM_PROMPT",
    "PLAN_USER_TEMPLATE",
    "EXECUTION_SYSTEM_PROMPT",
    "EXECUTION_USER_TEMPLATE",
    "SELECTION_SYSTEM_PROMPT",
    "SELECTION_USER_TEMPLATE",
    "WRITING_SYSTEM_PROMPT",
    "WRITING_USER_TEMPLATE",
    "REVIEW_SYSTEM_PROMPT",
    "REVIEW_USER_TEMPLATE",
]
