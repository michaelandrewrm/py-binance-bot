"""
Enhanced error handling and validation for CLI operations

This module provides decorators and validation functions for consistent
error handling across the trading bot CLI interface.
"""

import time
import re
import typer
import json

from pathlib import Path
from typing import Optional, Any, Dict
from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any, Callable, Optional
from datetime import timedelta

from .output import OutputHandler

logger = OutputHandler("ui.error_handling")


class CLIError(Exception):
    """Base exception for CLI-related errors"""

    pass


class ValidationError(CLIError):
    """Exception for input validation errors"""

    pass


class ConfigurationError(CLIError):
    """Exception for configuration-related errors"""

    pass


def handle_cli_errors(func: Callable) -> Callable:
    """Decorator for consistent CLI error handling with detailed logging"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except typer.Abort:
            # User cancelled - exit cleanly without error message
            raise
        except ValidationError as e:
            logger.error(f"Invalid input: {e}")
            raise typer.Exit(1)
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            raise typer.Exit(1)
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise typer.Exit(1)
        except PermissionError as e:
            logger.error(f"Permission denied: {e}")
            logger.info(
                "💡 Try running with appropriate permissions",
                console_style="yellow",
            )
            raise typer.Exit(1)
        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            logger.info(
                "💡 Check your internet connection and try again",
                console_style="yellow",
            )
            raise typer.Exit(1)
        except TimeoutError as e:
            logger.error(f"Operation timed out: {e}")
            logger.info(
                "💡 Try again or check system resources", console_style="yellow"
            )
            raise typer.Exit(1)
        except KeyboardInterrupt:
            logger.warning("⏹️  Operation cancelled by user")
            logger.user_action("User cancelled operation", {"function": func.__name__})
            raise typer.Abort()
        except Exception as e:
            # Log full traceback for debugging, with contextual info for user
            logger.critical(
                f"Unexpected error: {e}",
                log_extra={"function": func.__name__, "traceback": True},
            )
            logger.info(
                "💡 Check logs for detailed error information",
                console_style="yellow",
            )
            logger.console_only(
                "💡 Consider reporting this issue if it persists", style="dim"
            )
            raise typer.Exit(1)

    return wrapper


def validate_required_config(config: dict, required_fields: list[str]) -> None:
    """Validate that required configuration fields are present"""
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ConfigurationError(
            f"Missing required configuration fields: {', '.join(missing_fields)}"
        )


def parse_duration_safe(duration_str: str) -> timedelta:
    """Parse duration string with comprehensive validation and error handling"""
    if not duration_str or not isinstance(duration_str, str):
        raise ValidationError("Duration string cannot be empty")

    duration_str = duration_str.strip().lower()

    # Validate format with more robust regex
    pattern = r"^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$"
    match = re.match(pattern, duration_str)

    if not match:
        raise ValidationError(
            f"Invalid duration format: '{duration_str}'. "
            f"Expected format: '1d2h30m45s' (days, hours, minutes, seconds)"
        )

    days, hours, minutes, seconds = match.groups()

    # Convert to integers with validation
    try:
        total_seconds = 0

        if days:
            days_val = int(days)
            if days_val > 365:  # Reasonable limit
                raise ValidationError("Duration cannot exceed 365 days")
            if days_val < 0:
                raise ValidationError("Days cannot be negative")
            total_seconds += days_val * 24 * 3600

        if hours:
            hours_val = int(hours)
            if hours_val > 23:
                raise ValidationError("Hours component cannot exceed 23")
            if hours_val < 0:
                raise ValidationError("Hours cannot be negative")
            total_seconds += hours_val * 3600

        if minutes:
            minutes_val = int(minutes)
            if minutes_val > 59:
                raise ValidationError("Minutes component cannot exceed 59")
            if minutes_val < 0:
                raise ValidationError("Minutes cannot be negative")
            total_seconds += minutes_val * 60

        if seconds:
            seconds_val = int(seconds)
            if seconds_val > 59:
                raise ValidationError("Seconds component cannot exceed 59")
            if seconds_val < 0:
                raise ValidationError("Seconds cannot be negative")
            total_seconds += seconds_val

    except ValueError as e:
        raise ValidationError(f"Invalid duration values: {e}")

    if total_seconds <= 0:
        raise ValidationError("Duration must be greater than 0")

    if total_seconds > 365 * 24 * 3600:  # Max 1 year
        raise ValidationError("Duration cannot exceed 1 year")

    return timedelta(seconds=total_seconds)


def format_duration(duration: timedelta) -> str:
    """Format timedelta as human-readable string"""
    if not isinstance(duration, timedelta):
        return "Invalid duration"

    total_seconds = int(duration.total_seconds())
    if total_seconds < 0:
        return "Invalid duration"

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 and not parts:  # Only show seconds if no larger units
        parts.append(f"{seconds}s")

    return " ".join(parts) if parts else "0s"


def confirm_operation(
    message: str,
    default: bool = False,
    danger: bool = False,
    require_explicit: bool = False,
) -> bool:
    """Enhanced confirmation with safety features for dangerous operations"""

    if danger:
        logger.critical(f"DANGER: {message}")
        if require_explicit:
            # For very dangerous operations, require typing "yes"
            response = typer.prompt(
                "Type 'yes' to confirm this dangerous operation", default="no"
            )
            return response.lower() == "yes"
    else:
        logger.console_only(f"❓ {message}")

    return typer.confirm("Are you sure?", default=default)


def validate_positive_int(
    value: str, min_val: int = 1, max_val: Optional[int] = None
) -> int:
    """Validate positive integer input with optional range checking"""
    try:
        int_val = int(value)
    except ValueError:
        raise ValidationError(f"'{value}' is not a valid integer")

    if int_val < min_val:
        raise ValidationError(f"Value must be at least {min_val}")

    if max_val is not None and int_val > max_val:
        raise ValidationError(f"Value must not exceed {max_val}")

    return int_val


def safe_json_load(file_path: str) -> dict:
    """Safely load JSON file with proper error handling"""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    if not path.is_file():
        raise ValidationError(f"Path is not a file: {file_path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValidationError("Configuration file must contain a JSON object")

        return data

    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {file_path}: {e}")
    except PermissionError:
        raise PermissionError(f"Cannot read configuration file: {file_path}")


def retry_operation(
    operation: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """Retry an operation with exponential backoff"""

    last_exception = None

    for attempt in range(max_retries):
        try:
            return operation()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Operation failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                logger.debug_only(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Operation failed after {max_retries} attempts: {e}")

    raise last_exception


def log_user_action(action: str, details: dict = None):
    """Log user actions for audit trail"""
    log_data = {
        "action": action,
        "timestamp": None,  # Will be added by logger
        "details": details or {},
    }

    # Use the unified output for user actions (audit trail)
    logger.user_action(action, log_data)


class ProgressTracker:
    """Simple progress tracking for long operations"""

    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.output = logger  # Use unified output

    def update(self, step: int = None, message: str = None):
        """Update progress"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        percentage = (self.current_step / self.total_steps) * 100
        progress_bar = "█" * int(percentage // 5) + "░" * (20 - int(percentage // 5))

        status = message or f"Step {self.current_step}/{self.total_steps}"
        # Use console_only to avoid logging every progress update
        self.output.console_only(
            f"\r{self.description}: [{progress_bar}] {percentage:.1f}% - {status}",
            style="cyan",
        )

    def complete(self, message: str = "Complete"):
        """Mark as complete"""
        self.current_step = self.total_steps
        self.update(message=message)
        self.output.console_only("")  # New line
