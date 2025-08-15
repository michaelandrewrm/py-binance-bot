"""
Unified output handling for CLI operations

This module provides a single interface for both user feedback and logging,
eliminating the need for dual console.print() and logger calls.
"""

import logging
from typing import Optional, Any
from enum import Enum
from rich.console import Console


class MessageLevel(Enum):
    """Message severity levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    # UI-specific levels
    SUCCESS = "success"
    USER_ACTION = "user_action"
    CUSTOM = "custom"


class OutputHandler:
    """Unified handler for user output and logging"""

    def __init__(self, logger_name: str = __name__, show_console: bool = True):
        self.logger = logging.getLogger(logger_name)
        self.console = Console() if show_console else None
        self.show_console = show_console

    def message(
        self,
        text: str,
        level: MessageLevel = MessageLevel.INFO,
        console_style: Optional[str] = None,
        log_extra: Optional[dict] = None,
        console_only: bool = False,
        log_only: bool = False,
    ):
        """
        Send message to both console and logs with appropriate formatting

        Args:
            text: The message text
            level: Message severity level
            console_style: Rich console style (color/formatting)
            log_extra: Extra data for structured logging
            console_only: Only show on console, skip logging
            log_only: Only log, skip console output
        """

        # Determine styles and log levels
        style_map = {
            MessageLevel.SUCCESS: "green",
            MessageLevel.INFO: "cyan",
            MessageLevel.WARNING: "yellow",
            MessageLevel.ERROR: "red",
            MessageLevel.CRITICAL: "red bold",
            MessageLevel.USER_ACTION: "blue bold",
        }

        log_level_map = {
            MessageLevel.DEBUG: logging.DEBUG,
            MessageLevel.INFO: logging.INFO,
            MessageLevel.SUCCESS: logging.INFO,
            MessageLevel.WARNING: logging.WARNING,
            MessageLevel.ERROR: logging.ERROR,
            MessageLevel.CRITICAL: logging.CRITICAL,
            MessageLevel.USER_ACTION: logging.INFO,
            MessageLevel.CUSTOM: logging.INFO,
        }

        # Console output
        if self.show_console and not log_only:
            style = console_style or style_map.get(level, "white")

            # Add emoji/icons for better UX
            icon_map = {
                MessageLevel.SUCCESS: "✅",
                MessageLevel.WARNING: "⚠️",
                MessageLevel.ERROR: "❌",
                MessageLevel.CRITICAL: "🚨",
                MessageLevel.INFO: "ℹ️",
                MessageLevel.USER_ACTION: "👤",
            }

            icon = icon_map.get(level, "")
            display_text = f"{icon} {text}" if icon else text

            self.console.print(display_text, style=style)

        # Logging output
        if not console_only:
            log_level = log_level_map.get(level, logging.INFO)
            extra_data = log_extra or {}

            # Add context to log message
            log_text = text
            if level == MessageLevel.USER_ACTION:
                extra_data["audit"] = True

            self.logger.log(log_level, log_text, extra=extra_data)

    # Convenience methods
    def success(self, text: str, **kwargs):
        """Success message (green)"""
        self.message(text, MessageLevel.SUCCESS, **kwargs)

    def info(self, text: str, **kwargs):
        """Info message (cyan)"""
        self.message(text, MessageLevel.INFO, **kwargs)

    def warning(self, text: str, **kwargs):
        """Warning message (yellow)"""
        self.message(text, MessageLevel.WARNING, **kwargs)

    def error(self, text: str, **kwargs):
        """Error message (red)"""
        self.message(text, MessageLevel.ERROR, **kwargs)

    def critical(self, text: str, **kwargs):
        """Critical error message (red bold)"""
        self.message(text, MessageLevel.CRITICAL, **kwargs)

    def user_action(self, action: str, details: dict = None, **kwargs):
        """Log user action for audit trail"""
        text = f"User action: {action}"
        log_extra = {"audit": True, "action": action, "details": details or {}}
        self.message(text, MessageLevel.USER_ACTION, log_extra=log_extra, **kwargs)

    def custom(self, text: str, **kwargs):
        """Log custom message (white)"""
        self.message(text, MessageLevel.CUSTOM, **kwargs)

    def debug_only(self, text: str, **kwargs):
        """Debug message (log only, no console)"""
        self.message(text, MessageLevel.DEBUG, log_only=True, **kwargs)

    def console_only(self, text: str, style: str = None, **kwargs):
        """Console message only (no logging)"""
        self.message(
            text, MessageLevel.INFO, console_style=style, console_only=True, **kwargs
        )


# Global instances for different contexts
security_output = OutputHandler("ui.security")
session_output = OutputHandler("ui.session_manager")
