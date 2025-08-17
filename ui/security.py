"""
Security utilities for CLI operations

This module provides secure credential storage, encryption, and validation
functions for the trading bot CLI interface.
"""

import os
import base64
import json
import re
import typer

from pathlib import Path
from typing import Tuple, Dict, Any
from decimal import Decimal, InvalidOperation

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

from .output import OutputHandler


logger = OutputHandler("ui.security")


class SecurityError(Exception):
    """Custom exception for security-related errors"""

    pass


def check_encryption_available() -> bool:
    """Check if encryption is available and configured"""
    if not ENCRYPTION_AVAILABLE:
        logger.warning("Cryptography library not available. Install with: pip install cryptography")
        logger.warning("Falling back to basic credential storage")
        return False

    return True


def get_encryption_key() -> bytes:
    """Generate encryption key from environment variable or user password"""
    if not ENCRYPTION_AVAILABLE:
        raise SecurityError("Cryptography library not available")

    # First try environment variable for automated deployments
    env_key = os.getenv("TRADING_BOT_ENCRYPTION_KEY")
    if env_key:
        try:
            return base64.urlsafe_b64decode(env_key.encode())
        except Exception as e:
            logger.error(f"Invalid encryption key in environment: {e}")
            raise SecurityError("Invalid encryption key in environment")

    # Fall back to password-based key derivation for interactive use
    password = typer.prompt("Enter encryption password", hide_input=True)
    if len(password) < 8:
        raise SecurityError("Password must be at least 8 characters long")

    # Use a fixed salt for simplicity - in production, store salt separately
    salt = b"trading_bot_salt_2024"

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,  # Strong iteration count
    )

    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def encrypt_credentials(api_key: str, api_secret: str) -> Dict[str, str]:
    """Encrypt API credentials for secure storage"""
    if not ENCRYPTION_AVAILABLE:
        logger.warning("Storing credentials without encryption")
        return {"api_key": api_key, "api_secret": api_secret, "encrypted": False}

    try:
        key = get_encryption_key()
        f = Fernet(key)

        return {
            "api_key_encrypted": f.encrypt(api_key.encode()).decode(),
            "api_secret_encrypted": f.encrypt(api_secret.encode()).decode(),
            "encrypted": True,
        }
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise SecurityError(f"Failed to encrypt credentials: {e}")


def decrypt_credentials(config: Dict[str, Any]) -> Tuple[str, str]:
    """Decrypt API credentials from secure storage"""
    # Handle unencrypted legacy format
    if not config.get("encrypted", True):
        return config.get("api_key", ""), config.get("api_secret", "")

    # Handle encrypted format
    if not ENCRYPTION_AVAILABLE:
        raise SecurityError(
            "Cannot decrypt credentials: cryptography library not available"
        )

    try:
        key = get_encryption_key()
        f = Fernet(key)

        api_key = f.decrypt(config["api_key_encrypted"].encode()).decode()
        api_secret = f.decrypt(config["api_secret_encrypted"].encode()).decode()

        return api_key, api_secret
    except KeyError as e:
        raise SecurityError(f"Missing encrypted credential: {e}")
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise SecurityError(f"Failed to decrypt credentials: {e}")


def validate_trading_symbol(symbol: str) -> str:
    """Validate and normalize trading symbol format"""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol cannot be empty")

    # Remove whitespace and convert to uppercase
    symbol = symbol.strip().upper()

    # Basic validation: letters and numbers only, reasonable length
    if not re.match(r"^[A-Z0-9]{6,12}$", symbol):
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. "
            f"Expected format like 'BTCUSDT' (6-12 alphanumeric characters)"
        )

    return symbol


def validate_decimal_input(
    value: str,
    min_value: Decimal = None,
    max_value: Decimal = None,
    field_name: str = "value",
) -> Decimal:
    """Validate and convert decimal input with range checking"""
    if not value or not isinstance(value, str):
        raise ValueError(f"{field_name} cannot be empty")

    try:
        decimal_value = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        raise ValueError(f"Invalid {field_name}: '{value}'. Must be a valid number")

    if decimal_value <= 0:
        raise ValueError(f"{field_name} must be positive")

    if min_value is not None and decimal_value < min_value:
        raise ValueError(f"{field_name} must be at least {min_value}")

    if max_value is not None and decimal_value > max_value:
        raise ValueError(f"{field_name} must not exceed {max_value}")

    return decimal_value


def validate_api_key_format(api_key: str) -> bool:
    """Basic validation for API key format"""
    if not api_key or not isinstance(api_key, str):
        return False

    # Basic checks: reasonable length and alphanumeric
    return len(api_key) >= 32 and api_key.replace("-", "").replace("_", "").isalnum()


def secure_file_write(file_path: Path, data: Dict[str, Any]) -> None:
    """Write data to file with secure permissions"""
    try:
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Set restrictive permissions before writing (Unix systems)
        if os.name != "nt":  # Not Windows
            file_path.touch(mode=0o600)  # Owner read/write only

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        # Double-check permissions after writing
        if os.name != "nt":
            os.chmod(file_path, 0o600)

    except PermissionError:
        raise SecurityError(f"Permission denied writing to {file_path}")
    except Exception as e:
        raise SecurityError(f"Failed to write secure file: {e}")


def check_file_permissions(file_path: Path) -> bool:
    """Check if file has secure permissions"""
    if not file_path.exists():
        return True  # File doesn't exist yet

    if os.name == "nt":  # Windows
        return True  # Skip permission check on Windows

    file_stat = file_path.stat()
    # Check if readable by group/others (should only be readable by owner)
    return not bool(file_stat.st_mode & 0o077)


def mask_sensitive_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a display-safe version of configuration with masked sensitive data"""
    display_config = config.copy()

    # Enhanced credential masking
    sensitive_keys = [
        "api_key_encrypted",
        "api_secret_encrypted",
        "api_key",
        "api_secret",
        "password",
        "token",
    ]

    for key in sensitive_keys:
        if key in display_config:
            value = str(display_config[key])
            if "secret" in key.lower() or "password" in key.lower():
                display_config[key] = "***HIDDEN***"
            else:
                # Show first 4 characters only for keys/tokens
                if len(value) > 8:
                    display_config[key] = f"{value[:4]}...***"
                else:
                    display_config[key] = "***"

    return display_config


def validate_file_path(path_str: str, must_exist: bool = False) -> Path:
    """Validate and sanitize file paths to prevent path traversal"""
    if not path_str:
        raise ValueError("Path cannot be empty")

    # Convert to Path object and resolve
    try:
        path = Path(path_str).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {e}")

    # Basic path traversal protection - ensure path doesn't go outside project
    cwd = Path.cwd().resolve()
    try:
        path.relative_to(cwd)
    except ValueError:
        # Allow absolute paths in user home directory or data directories
        home = Path.home().resolve()
        if not (str(path).startswith(str(home)) or str(path).startswith("/tmp")):
            raise ValueError(f"Path outside allowed directories: {path}")

    if must_exist and not path.exists():
        raise ValueError(f"Path does not exist: {path}")

    return path
