"""
Unified output handling for CLI operations

This module provides a single interface for both user-facing output and structured
logging, eliminating the need for dual console.print() and logger calls.

Example usage:
    output = OutputHandler(log_level=logging.DEBUG, log_file="app.log")
    output.success("Operation completed successfully")
    output.error("Failed to process", error=exception_obj, operation="data_processing")
"""

import json
import logging
import os
import sys
import threading
import traceback
from typing import Optional, Union, Dict, Any
from enum import Enum
from pathlib import Path
from rich.console import Console
from rich.traceback import Traceback
from rich.panel import Panel

# Safe reserved LogRecord attributes that shouldn't be in extras
SAFE_RESERVED = {
    "message",
    "asctime",
    "levelname",
    "name",
    "filename",
    "lineno",
    "funcName",
    "module",
    "pathname",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "getMessage",
    "exc_info",
    "exc_text",
    "stack_info",
    "args",
    "msg",
}


def _sanitize_extra(d: dict) -> dict:
    """Sanitize extra logging data to avoid collisions and ensure serializability"""
    out = {}
    for k, v in d.items():
        # Skip reserved LogRecord attributes
        if k in SAFE_RESERVED:
            continue

        # Ensure value is serializable - with robust fallback
        try:
            repr(v)  # Test if object can be represented
            out[k] = v
        except Exception:
            try:
                out[k] = str(v)  # Try converting to string
            except Exception:
                # Last resort: use object type and id
                out[k] = f"<{type(v).__name__} object at {hex(id(v))}>"

    # Namespace to avoid collisions with LogRecord attributes
    return {f"ctx_{k}": v for k, v in out.items()}


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        base = record.__dict__.copy()
        base["msg"] = record.getMessage()

        # Ensure all values are JSON serializable
        for k in list(base):
            if not isinstance(base[k], (str, int, float, bool, type(None))):
                base[k] = str(base[k])

        return json.dumps(base, ensure_ascii=False)


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


class ErrorContext:
    """Enhanced error context for better error reporting.

    Attributes:
        error: The original exception.
        operation: High-level operation name (e.g., "fetch_config").
        user_context: Friendly description for end users.
        technical_details: Dict echoed into logs for debugging (avoid PII).
        error_type: Exception class name.
        error_message: str(error).
        traceback: Full traceback derived from the exception.
    """

    def __init__(
        self,
        error: Exception,
        operation: Optional[str] = None,
        user_context: Optional[str] = None,
        technical_details: Optional[Dict[str, Any]] = None,
    ):
        self.error = error
        self.operation = operation or "Unknown operation"
        self.user_context = user_context or "An error occurred"
        self.technical_details = technical_details or {}
        self.error_type = type(error).__name__
        self.error_message = str(error)
        # Always derive traceback from the exception
        self.traceback = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )


class LoggerConfig:
    """Logger configuration with best practices and environment support"""

    @staticmethod
    def _get_env_log_level() -> int:
        """Get log level from environment variable"""
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_str, logging.INFO)

    @staticmethod
    def _get_env_log_file() -> Optional[Path]:
        """Get log file path from environment variable"""
        log_file_str = os.getenv("LOG_FILE")
        return Path(log_file_str) if log_file_str else None

    @staticmethod
    def _should_use_json_logs() -> bool:
        """Check if JSON logging is enabled via environment"""
        return os.getenv("JSON_LOGS", "").lower() in ("true", "1", "yes")

    @staticmethod
    def setup_logger(
        name: str,
        level: Optional[int] = None,
        log_file: Optional[Union[str, Path]] = None,
        format_string: Optional[str] = None,
        console_level: Optional[int] = None,
        file_level: int = logging.DEBUG,
        use_json: Optional[bool] = None,
    ) -> logging.Logger:
        """Set up logger with proper formatting and handlers

        Args:
            name: Logger name
            level: Overall log level (overrides environment)
            log_file: Log file path (overrides environment)
            format_string: Custom format string
            console_level: Console handler level (defaults to level)
            file_level: File handler level (defaults to DEBUG)
            use_json: Use JSON formatting (overrides environment)
        """

        logger = logging.getLogger(name)

        # Apply environment defaults if not explicitly provided
        if level is None:
            level = LoggerConfig._get_env_log_level()
        if log_file is None:
            log_file = LoggerConfig._get_env_log_file()
        if use_json is None:
            use_json = LoggerConfig._should_use_json_logs()

        console_level = console_level or level

        # If handlers already exist, optionally refresh level/handlers
        if logger.handlers:
            # Set logger to minimum level needed for any handler
            logger.setLevel(min(console_level, file_level, logging.DEBUG))

            # Update existing handlers' levels
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    handler.setLevel(console_level)
                elif isinstance(handler, logging.FileHandler):
                    handler.setLevel(file_level)

            # Add file handler if newly requested and not already present
            if log_file and not any(
                isinstance(h, logging.FileHandler) for h in logger.handlers
            ):
                if isinstance(log_file, str):
                    log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(file_level)

                # Choose formatter based on JSON preference
                if use_json:
                    file_handler.setFormatter(JsonFormatter())
                else:
                    format_str = format_string or (
                        "%(asctime)s - %(name)s - %(levelname)s - "
                        "%(filename)s:%(lineno)d - %(message)s"
                    )
                    file_handler.setFormatter(logging.Formatter(format_str))

                logger.addHandler(file_handler)
            return logger

        # Set logger to minimum level needed for any handler
        logger.setLevel(min(console_level, file_level, logging.DEBUG))

        # Default format with context (JSON not typically used for console)
        if not format_string:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )

        formatter = logging.Formatter(format_string)

        # Console handler aligned with Rich Console stderr
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            # Convert string to Path if needed
            if isinstance(log_file, str):
                log_file = Path(log_file)

            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(file_level)

            # Use JSON formatter for file if requested
            if use_json:
                file_handler.setFormatter(JsonFormatter())
            else:
                file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)

        return logger


class OutputHandler:
    """Unified handler for user output and logging with enhanced error handling"""

    def __init__(
        self,
        logger_name: str = __name__,
        show_console: bool = True,
        log_level: Optional[int] = None,
        log_file: Optional[Path] = None,
        use_json: Optional[bool] = None,
    ):

        self.show_console = show_console
        self.console = Console(stderr=True) if show_console else None

        # Set up logger with proper configuration
        self.logger = LoggerConfig.setup_logger(
            logger_name, level=log_level, log_file=log_file, use_json=use_json
        )

        # Thread-safe error tracking for better debugging
        self._error_lock = threading.Lock()
        self._error_count = 0
        self._last_error: Optional[ErrorContext] = None

    @property
    def error_count(self) -> int:
        """Thread-safe access to error count"""
        with self._error_lock:
            return self._error_count

    @property
    def last_error(self) -> Optional[ErrorContext]:
        """Thread-safe access to last error"""
        with self._error_lock:
            return self._last_error

    def _increment_error_count(self, error_context: ErrorContext) -> None:
        """Thread-safe error tracking update"""
        with self._error_lock:
            self._error_count += 1
            self._last_error = error_context

    def message(
        self,
        text: str,
        level: MessageLevel = MessageLevel.INFO,
        console_style: Optional[str] = None,
        log_extra: Optional[dict] = None,
        console_only: bool = False,
        log_only: bool = False,
        error_context: Optional[ErrorContext] = None,
        **kwargs,  # Accept additional keyword arguments
    ) -> None:
        """
        Send message to both console and logs with appropriate formatting

        Args:
            text: The message text
            level: Message severity level
            console_style: Rich console style (color/formatting)
            log_extra: Extra data for structured logging
            console_only: Only show on console, skip logging
            log_only: Only log, skip console output
            error_context: Enhanced error context for errors
            **kwargs: Additional data included in logs (will be sanitized and namespaced)

        Raises:
            ValueError: If both console_only and log_only are True
        """

        # Guard against mutually exclusive flags
        if console_only and log_only:
            raise ValueError("console_only and log_only cannot both be True")

        # Determine styles and log levels
        style_map = {
            MessageLevel.SUCCESS: "green",
            MessageLevel.INFO: "cyan",
            MessageLevel.WARNING: "yellow",
            MessageLevel.ERROR: "red",
            MessageLevel.CRITICAL: "red bold",
            MessageLevel.USER_ACTION: "blue bold",
            MessageLevel.CUSTOM: "white",  # Default style for CUSTOM
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

        # Console output with enhanced error handling
        if self.show_console and not log_only:
            self._render_console_message(
                text, level, console_style, style_map, error_context
            )

        # Logging output with structured data - sanitize and merge kwargs
        if not console_only:
            merged_log_extra = _sanitize_extra(log_extra or {})
            merged_log_extra.update(_sanitize_extra(kwargs))
            self._log_message(
                text, level, log_level_map, merged_log_extra, error_context
            )

    def _render_console_message(
        self,
        text: str,
        level: MessageLevel,
        console_style: Optional[str],
        style_map: Dict[MessageLevel, str],
        error_context: Optional[ErrorContext],
    ) -> None:
        """Render console message with appropriate styling"""

        style = console_style or style_map.get(level, "white")

        # Add emoji/icons for better UX
        icon_map = {
            MessageLevel.SUCCESS: "✅",
            MessageLevel.WARNING: "⚠️",
            MessageLevel.ERROR: "❌",
            MessageLevel.CRITICAL: "🚨",
            MessageLevel.INFO: "ℹ️",
            MessageLevel.USER_ACTION: "👤",
            MessageLevel.DEBUG: "🔍",
        }

        icon = icon_map.get(level, "")
        display_text = f"{icon} {text}" if icon else text

        # Enhanced error display
        if level in [MessageLevel.ERROR, MessageLevel.CRITICAL] and error_context:
            self._render_enhanced_error(display_text, error_context, style)
        else:
            try:
                self.console.print(display_text, style=style)
            except Exception:
                # Fallback for any display issues
                self.console.print(str(display_text), style=style)

    def _render_enhanced_error(
        self, text: str, error_context: ErrorContext, style: str
    ) -> None:
        """Render enhanced error display for better user experience"""

        # User-friendly error panel
        error_panel = Panel(
            f"[bold red]{text}[/bold red]\n\n"
            f"[dim]Context: {error_context.user_context}[/dim]\n"
            f"[dim]Operation: {error_context.operation}[/dim]\n"
            f"[dim]Error Type: {error_context.error_type}[/dim]",
            title="Error",
            title_align="left",
            border_style="red",
        )

        self.console.print(error_panel)

        # Show technical details only in debug mode
        if self.logger.level <= logging.DEBUG and error_context.traceback:
            self.console.print("\n[dim]Technical Details (Debug Mode):[/dim]")
            # Use Rich's traceback for better formatting
            tb = Traceback.from_exception(
                type(error_context.error),
                error_context.error,
                error_context.error.__traceback__,
            )
            self.console.print(tb)

    def _log_message(
        self,
        text: str,
        level: MessageLevel,
        log_level_map: Dict[MessageLevel, int],
        log_extra: Optional[dict],
        error_context: Optional[ErrorContext],
    ) -> None:
        """Log message with structured data and error context"""

        log_level = log_level_map.get(level, logging.INFO)
        extra_data = log_extra or {}

        # Enhanced logging for errors
        if error_context:
            extra_data.update(
                {
                    "ctx_error_type": error_context.error_type,
                    "ctx_error_message": error_context.error_message,
                    "ctx_operation": error_context.operation,
                    "ctx_technical_details": error_context.technical_details,
                    "ctx_traceback": error_context.traceback,
                }
            )
            self._increment_error_count(error_context)

        # Add context to log message
        log_text = text
        if level == MessageLevel.USER_ACTION:
            extra_data["ctx_audit"] = True

        # Use stacklevel to point to real caller instead of this wrapper
        self.logger.log(log_level, log_text, extra=extra_data, stacklevel=3)

    # Enhanced convenience methods with better error handling
    def success(self, text: str, **kwargs) -> None:
        """Success message (green)"""
        self.message(text, MessageLevel.SUCCESS, **kwargs)

    def info(self, text: str, **kwargs) -> None:
        """Info message (cyan)"""
        self.message(text, MessageLevel.INFO, **kwargs)

    def warning(self, text: str, **kwargs) -> None:
        """Warning message (yellow)"""
        self.message(text, MessageLevel.WARNING, **kwargs)

    def debug(self, text: str, **kwargs) -> None:
        """Debug message with technical context"""
        self.message(text, MessageLevel.DEBUG, **kwargs)

    def _emit_error(
        self,
        level: MessageLevel,
        text: str,
        error: Optional[Exception] = None,
        operation: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Common error emission logic to reduce duplication"""
        error_context = None
        if error:
            # Use dedicated technical_details instead of fishing from log_extra
            technical_details = kwargs.pop("technical_details", {})
            error_context = ErrorContext(
                error=error,
                operation=operation,
                user_context=text,
                technical_details=technical_details,
            )

        self.message(text, level, error_context=error_context, **kwargs)

    def error(
        self,
        text: str,
        error: Optional[Exception] = None,
        operation: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Enhanced error message with optional exception context"""
        self._emit_error(MessageLevel.ERROR, text, error, operation, **kwargs)

    def critical(
        self,
        text: str,
        error: Optional[Exception] = None,
        operation: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Enhanced critical error message with optional exception context"""
        self._emit_error(MessageLevel.CRITICAL, text, error, operation, **kwargs)

    def handle_exception(
        self,
        error: Exception,
        user_message: str,
        operation: str = "Unknown operation",
        show_traceback: bool = False,
        technical_details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Comprehensive exception handling with separation of concerns

        Args:
            error: The exception that occurred
            user_message: User-friendly error message
            operation: Description of the operation that failed
            show_traceback: Whether to show technical details to user
            technical_details: Additional technical context
        """

        # Create error context
        error_context = ErrorContext(
            error=error,
            operation=operation,
            user_context=user_message,
            technical_details=technical_details or {},
        )

        # Log full technical details for developers
        self.logger.error(
            f"Exception in {operation}: {str(error)}",
            extra=_sanitize_extra(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "operation": operation,
                    "technical_details": technical_details or {},
                    "traceback": traceback.format_exc(),
                }
            ),
            exc_info=True,
            stacklevel=2,
        )

        # Show user-friendly message to end users
        if show_traceback or self.logger.level <= logging.DEBUG:
            # Developer mode - show technical details
            self.message(
                f"{user_message}\n\nTechnical details: {str(error)}",
                MessageLevel.ERROR,
                error_context=error_context,
            )
        else:
            # End user mode - clean message only
            self.message(user_message, MessageLevel.ERROR, error_context=error_context)

    def user_action(
        self, action: str, details: Optional[dict] = None, **kwargs
    ) -> None:
        """Log user action for audit trail with enhanced context"""
        text = f"[AUDIT] User action: {action}"
        log_extra = {
            "audit": True,
            "action": action,
            "details": details or {},
            "user_session": kwargs.get("user_session"),
            "timestamp": kwargs.get("timestamp"),
        }
        # Remove these from kwargs to avoid double-inclusion
        kwargs.pop("user_session", None)
        kwargs.pop("timestamp", None)
        self.message(text, MessageLevel.USER_ACTION, log_extra=log_extra, **kwargs)

    def custom(self, text: str, **kwargs) -> None:
        """Custom message with flexible styling"""
        self.message(text, MessageLevel.CUSTOM, **kwargs)

    def debug_only(self, text: str, **kwargs) -> None:
        """Debug message (log only, no console)"""
        self.message(text, MessageLevel.DEBUG, log_only=True, **kwargs)

    def console_only(self, text: str, style: Optional[str] = None, **kwargs) -> None:
        """Console message only (no logging)"""
        self.message(
            text,
            MessageLevel.INFO,
            console_style=style,
            console_only=True,
            **kwargs,
        )

    def note(self, text: str, **kwargs) -> None:
        """Dimmed info message for less important information"""
        self.message(text, MessageLevel.INFO, console_style="dim", **kwargs)

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors for debugging and monitoring (thread-safe)"""
        with self._error_lock:
            return {
                "total_errors": self._error_count,
                "last_error": (
                    {
                        "type": self._last_error.error_type,
                        "message": self._last_error.error_message,
                        "operation": self._last_error.operation,
                    }
                    if self._last_error
                    else None
                ),
            }

    def reset_error_tracking(self) -> None:
        """Reset error tracking counters (thread-safe)"""
        with self._error_lock:
            self._error_count = 0
            self._last_error = None
