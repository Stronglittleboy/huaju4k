"""
CLI Data Models

Data structures for command-line interface components.
"""

from dataclasses import dataclass
from typing import Any, Optional, Callable


@dataclass
class MenuOption:
    """Menu option data structure."""
    key: str
    title: str
    description: str
    action: Optional[Callable] = None
    enabled: bool = True
    
    def __str__(self) -> str:
        """String representation of menu option."""
        status = "" if self.enabled else " (disabled)"
        return f"{self.key}. {self.title}{status}"


@dataclass
class UserInput:
    """User input data structure."""
    value: str
    is_valid: bool
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate input data."""
        if not self.is_valid and self.error_message is None:
            self.error_message = "Invalid input"