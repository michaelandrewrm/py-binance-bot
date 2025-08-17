"""
OuterHandler utility module for unified console output with rich formatting.

This module provides a centralized way to handle console output with different
message levels, colors, and icons. It uses the Rich library for enhanced
console formatting and supports various message severity levels.

Example usage:
    handler = OuterHandler()
    handler.success("Operation completed successfully")
    handler.warning("This is a warning message")
    handler.error("An error occurred")
"""

from typing import Optional
from enum import Enum
from rich.console import Console

from constants.icons import Icon


class MessageLevel(Enum):
    """Message severity levels"""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    DEBUG = "debug"
    INFO = "info"


class OutputHandler:
    def __init__(self):
        self.console = Console()

    def message(
        self,
        text: str,
        level: MessageLevel = MessageLevel.INFO,
        style: Optional[str] = None,
        **kwargs,  # Accept additional keyword arguments
    ) -> None:
        """
        Send message to both console and logs with appropriate formatting

        Args:
            text: The message text
            level: Message severity level
            style: Rich console style (color/formatting)
            **kwargs: Additional data included in logs (will be sanitized and namespaced)

        Raises:
            ValueError: If both console_only and log_only are True
        """

        # Determine styles and log levels
        style_map = {
            MessageLevel.SUCCESS: "green",
            MessageLevel.INFO: "cyan",
            MessageLevel.WARNING: "yellow",
            MessageLevel.ERROR: "red",
            MessageLevel.CRITICAL: "red bold",
        }

        # Add emoji/icons for better UX
        icon_map = {
            MessageLevel.SUCCESS: Icon.SUCCESS,
            MessageLevel.WARNING: Icon.WARNING,
            MessageLevel.ERROR: Icon.ERROR,
            MessageLevel.CRITICAL: Icon.ALARM,
            MessageLevel.INFO: Icon.INFO,
            MessageLevel.DEBUG: Icon.SEARCH,
        }

        display_icon = icon_map.get(level, "")
        display_style = style if style else style_map.get(level, "white")
        display_text = f"{display_icon} {text}" if display_icon else text

        self.print(display_text, style=display_style, **kwargs)

    def success(self, text: str, **kwargs) -> None:
        """Success message (green)"""
        self.message(text, MessageLevel.SUCCESS, **kwargs)

    def info(self, text: str, **kwargs) -> None:
        """Info message (cyan)"""
        self.message(text, MessageLevel.INFO, **kwargs)

    def warning(self, text: str, **kwargs) -> None:
        """Warning message (yellow)"""
        self.message(text, MessageLevel.WARNING, **kwargs)

    def critical(self, text: str, **kwargs) -> None:
        """Warning message (yellow)"""
        self.message(text, MessageLevel.CRITICAL, **kwargs)

    def error(self, text: str, **kwargs) -> None:
        """Warning message (yellow)"""
        self.message(text, MessageLevel.ERROR, **kwargs)

    def print(self, text: str, **kwargs) -> None:
        """Warning message (yellow)"""
        self.console.print(text, **kwargs)
