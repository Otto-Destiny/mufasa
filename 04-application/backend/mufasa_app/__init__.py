"""MUFASA application layer — the laptop-local orchestrator.

It owns the machine's resources, not just the screen: retrieval budget, prompt
size, generation limits, one generation at a time across every client, cancel,
and validation before anything is displayed.
"""

from .config import Settings, get_settings
from .governor import Busy, Cancelled, Governor

__all__ = ["Busy", "Cancelled", "Governor", "Settings", "get_settings"]
__version__ = "0.1.0"
