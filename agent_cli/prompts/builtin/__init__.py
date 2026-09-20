from .default import DEFAULT_SYSTEM_PROMPT
from .coder import CODER_SYSTEM_PROMPT
from .product_manager import PRODUCT_MANAGER_PROMPT
from .dba import DBA_PROMPT
from .devops import DEVOPS_PROMPT
from .tech_writer import TECH_WRITER_PROMPT

BUILTIN_PROMPTS = {
    "default": DEFAULT_SYSTEM_PROMPT,
    "coder": CODER_SYSTEM_PROMPT,
    "product_manager": PRODUCT_MANAGER_PROMPT,
    "dba": DBA_PROMPT,
    "devops": DEVOPS_PROMPT,
    "tech_writer": TECH_WRITER_PROMPT,
}

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "CODER_SYSTEM_PROMPT",
    "PRODUCT_MANAGER_PROMPT",
    "DBA_PROMPT",
    "DEVOPS_PROMPT",
    "TECH_WRITER_PROMPT",
    "BUILTIN_PROMPTS",
]
