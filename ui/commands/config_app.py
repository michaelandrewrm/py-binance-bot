"""
Configuration commands for the trading bot CLI
"""

import typer

from typing import List, Dict
from datetime import datetime, timezone
from pathlib import Path
from rich.table import Table

from ui.error_handling import (
    ValidationError,
    ConfigurationError,
    handle_cli_errors,
    confirm_operation,
    safe_json_load,
    log_user_action,
)
from ui.security import (
    SecurityError,
    encrypt_credentials,
    validate_api_key_format,
    secure_file_write,
    check_file_permissions,
    mask_sensitive_data,
    validate_file_path,
    check_encryption_available,
)
from utils.output import OutputHandler
from constants.icons import Icon

logger = OutputHandler()

# Configuration commands will be registered to this app instance
config_app = typer.Typer(help="Configuration management")


@config_app.command("init")
@handle_cli_errors
def init_config(
    api_key: str = typer.Option(
        ..., prompt=True, hide_input=True, help="Binance API Key"
    ),
    api_secret: str = typer.Option(
        ..., prompt=True, hide_input=True, help="Binance API Secret"
    ),
    testnet: bool = typer.Option(True, help="Use Binance Testnet"),
    data_dir: str = typer.Option("data", help="Data directory"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Initialize trading bot configuration with encrypted credentials"""

    # Validate API key format
    if not validate_api_key_format(api_key):
        raise ValidationError(
            "Invalid API key format. Must be at least 32 characters and alphanumeric."
        )

    # Validate and resolve data directory path
    try:
        data_path = validate_file_path(data_dir)
        if not data_path.is_absolute():
            data_path = Path.cwd() / data_dir
        data_path = data_path.resolve()
    except ValueError as e:
        raise ValidationError(f"Invalid data directory: {e}")

    # Check if encryption is available
    encryption_available = check_encryption_available()

    # Encrypt credentials if possible
    try:
        if encryption_available:
            encrypted_creds = encrypt_credentials(api_key, api_secret)
        else:
            logger.warning(
                "Encryption not available - credentials will be stored in plain text"
            )
            if not confirm_operation(
                "Continue without encryption?", default=False, danger=True
            ):
                logger.critical("Configuration cancelled for security reasons")
                raise typer.Abort()
            encrypted_creds = {
                "api_key": api_key,
                "api_secret": api_secret,
                "encrypted": False,
            }

        config = {
            **encrypted_creds,
            "testnet": testnet,
            "data_dir": str(data_path),
            "log_level": log_level,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        config_path = Path("config.json")
        secure_file_write(config_path, config)

        logger.success(f"Secure configuration saved to {config_path}")

        # Initialize data directory
        data_path.mkdir(parents=True, exist_ok=True)
        logger.success(f"Data directory created: {data_path}")

        # Log the action
        log_user_action(
            "config_initialized",
            {
                "testnet": testnet,
                "data_dir": str(data_path),
                "encryption_used": encryption_available,
            },
        )

    except SecurityError as e:
        raise ConfigurationError(f"Security error: {e}")


@config_app.command("show")
@handle_cli_errors
def show_config():
    """Show current configuration with enhanced security"""
    config_path = Path("config.json")

    if not config_path.exists():
        logger.error("Configuration file not found. Run 'config init' first.")
        raise typer.Exit(1)

    # Check file permissions
    if not check_file_permissions(config_path):
        logger.warning("Configuration file has insecure permissions!")
        logger.info("Consider running: chmod 600 config.json")

    try:
        config = safe_json_load(str(config_path))
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration file: {e}")

    # Create display configuration with proper credential masking
    display_config = mask_sensitive_data(config)

    table = Table(title="Trading Bot Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in display_config.items():
        table.add_row(key, str(value))

    logger.print(table)

    # Show security status
    if config.get("encrypted", True):
        logger.print(f"{Icon.LOCK} Credentials are encrypted")
    else:
        logger.warning("Credentials are stored in plain text")


@config_app.command("validate")
@handle_cli_errors
def validate_config():
    """Validate current secured configuration"""

    # Check for different configuration file types
    config_json_path = Path("config.json")

    logger.print(f"{Icon.SEARCH} Validating Trading Bot Configuration")
    logger.print("=" * 50)

    validation_results = []

    logger.print(f"{Icon.FILE} Checking configuration file...")
    json_issues = _validate_json_config(config_json_path)
    validation_results.extend(json_issues)

    logger.print(f"{Icon.LOCK} Checking security configuration and sensitive files...")
    security_issues = _validate_security_settings()
    validation_results.extend(security_issues)

    logger.info(f"{Icon.FOLDER} Checking directories...")
    directory_issues = _validate_directories()
    validation_results.extend(directory_issues)

    # Display results
    _display_validation_results(validation_results)

    # Summary
    logger.print(f"\n{Icon.BAR_CHART} Validation Summary")
    logger.print("=" * 50)

    error_count = sum(1 for issue in validation_results if issue["type"] == "error")
    warning_count = sum(1 for issue in validation_results if issue["type"] == "warning")
    info_count = sum(1 for issue in validation_results if issue["type"] == "info")

    if error_count == 0:
        logger.success("Configuration validation passed!")
        logger.info(f"   • Warnings: {warning_count}")
        logger.info(f"   • Info messages: {info_count}")
    else:
        logger.error(f"Configuration validation failed with {error_count} errors")
        logger.info(f"   • Warnings: {warning_count}")
        logger.info(f"   • Info messages: {info_count}")

        logger.print(f"\n{Icon.LIGHT_BULB} Recommended Actions:")
        logger.info("   • Fix the errors listed above")
        logger.info("   • Run 'config init' to reconfigure if needed")
        logger.info("   • Check file permissions: chmod 600 config.json")

        raise typer.Exit(1)


# Helper functions
def _validate_json_config(config_path: Path) -> List[Dict[str, str]]:
    """Validate JSON-based configuration file"""
    issues = []

    if not config_path.exists():
        issues.append(
            {
                "type": "warning",
                "category": "JSON Config",
                "message": f"{config_path} not found",
            }
        )
        return issues

    try:
        # Check file permissions
        if not check_file_permissions(config_path):
            issues.append(
                {
                    "type": "warning",
                    "category": "Security",
                    "message": f"{config_path} has insecure permissions. Run: chmod 600 {config_path}",
                }
            )

        # Load and validate JSON structure
        config = safe_json_load(str(config_path))

        # Check if credentials are encrypted
        is_encrypted = config.get("encrypted", True)

        # Required fields validation - adjust based on encryption
        if is_encrypted:
            required_fields = [
                "api_key_encrypted",
                "api_secret_encrypted",
                "testnet",
                "data_dir",
                "log_level",
            ]
            credential_fields = ["api_key_encrypted", "api_secret_encrypted"]
        else:
            required_fields = [
                "api_key",
                "api_secret",
                "testnet",
                "data_dir",
                "log_level",
            ]
            credential_fields = ["api_key", "api_secret"]

        missing_fields = []

        for field in required_fields:
            if field not in config:
                missing_fields.append(field)
            elif field in credential_fields:
                if not config[field] or len(config[field].strip()) == 0:
                    missing_fields.append(field + " (empty)")

        if missing_fields:
            issues.append(
                {
                    "type": "error",
                    "category": "JSON Config",
                    "message": f"Missing or insufficient required fields: {', '.join(missing_fields)}",
                }
            )

        # Validate specific fields
        if "data_dir" in config:
            data_path = Path(config["data_dir"])
            if not data_path.exists():
                issues.append(
                    {
                        "type": "warning",
                        "category": "JSON Config",
                        "message": f"Data directory does not exist: {data_path}",
                    }
                )

        if "log_level" in config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if config["log_level"].upper() not in valid_levels:
                issues.append(
                    {
                        "type": "error",
                        "category": "JSON Config",
                        "message": f"Invalid log level: {config['log_level']}. Must be one of: {', '.join(valid_levels)}",
                    }
                )

        # Validate API key format (only for unencrypted)
        if not is_encrypted and "api_key" in config and config["api_key"]:
            if not validate_api_key_format(config["api_key"]):
                issues.append(
                    {
                        "type": "error",
                        "category": "JSON Config",
                        "message": "API key format is invalid",
                    }
                )

        # Validate testnet setting
        if "testnet" in config:
            if not isinstance(config["testnet"], bool):
                issues.append(
                    {
                        "type": "error",
                        "category": "JSON Config",
                        "message": f"testnet must be true or false, got: {config['testnet']}",
                    }
                )

        # Check encryption status
        if is_encrypted:
            issues.append(
                {
                    "type": "info",
                    "category": "JSON Config",
                    "message": f"{Icon.SUCCESS} Credentials are encrypted",
                }
            )
        else:
            issues.append(
                {
                    "type": "warning",
                    "category": "JSON Config",
                    "message": f"{Icon.WARNING} Credentials are stored in plain text",
                }
            )

    except Exception as e:
        issues.append(
            {
                "type": "error",
                "category": "JSON Config",
                "message": f"Failed to parse JSON configuration: {e}",
            }
        )

    return issues


def _validate_security_settings() -> List[Dict[str, str]]:
    """Validate security settings and sensitive files"""
    issues = []

    # Check if encryption is available
    encryption_available = check_encryption_available()
    if encryption_available:
        issues.append(
            {
                "type": "info",
                "category": "Security",
                "message": f"{Icon.SUCCESS} Encryption capabilities are available",
            }
        )
    else:
        issues.append(
            {
                "type": "warning",
                "category": "Security",
                "message": "Encryption not available - install cryptography package for better security",
            }
        )

    # Check for common security files
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists():
        try:
            with open(gitignore_path, "r") as f:
                content = f.read()

            security_patterns = [
                "*.json",
                "config.json",
                ".env",
                "*.key",
                "*.pem",
            ]
            missing_patterns = []
            for pattern in security_patterns:
                if pattern not in content:
                    missing_patterns.append(pattern)

            if missing_patterns:
                issues.append(
                    {
                        "type": "warning",
                        "category": "Security",
                        "message": f"Consider adding these patterns to .gitignore: {', '.join(missing_patterns)}",
                    }
                )

        except Exception as e:
            issues.append(
                {
                    "type": "warning",
                    "category": "Security",
                    "message": f"Could not read .gitignore file: {e}",
                }
            )
    else:
        issues.append(
            {
                "type": "warning",
                "category": "Security",
                "message": "No .gitignore file found - create one to protect sensitive files",
            }
        )

    return issues


def _validate_directories() -> List[Dict[str, str]]:
    """Validate required directories"""
    issues = []

    # Required directories
    required_dirs = [
        ("data", "Data storage directory"),
        ("data_cache", "Data cache directory"),
        ("logs", "Log files directory (optional)"),
    ]

    for dir_name, description in required_dirs:
        dir_path = Path(dir_name)

        if dir_path.exists():
            issues.append(
                {
                    "type": "info",
                    "category": "Directories",
                    "message": f"{Icon.SUCCESS} {description} exists: {dir_path}",
                }
            )
        else:
            issues.append(
                {
                    "type": "info",
                    "category": "Directories",
                    "message": f"{Icon.FOLDER} {description} will be created: {dir_path}",
                }
            )

    # Check database file
    db_path = Path("trading_bot.db")
    if db_path.exists():
        issues.append(
            {
                "type": "info",
                "category": "Database",
                "message": f"{Icon.SUCCESS} Database file exists: {db_path}",
            }
        )
    else:
        issues.append(
            {
                "type": "info",
                "category": "Database",
                "message": f"{Icon.BAR_CHART} Database will be created on first run: {db_path}",
            }
        )

    return issues


def _display_validation_results(results: List[Dict[str, str]]) -> None:
    """Display validation results in a formatted table"""

    if not results:
        logger.success("No validation issues found!")
        return

    # Group results by type
    errors = [r for r in results if r["type"] == "error"]
    warnings = [r for r in results if r["type"] == "warning"]
    info = [r for r in results if r["type"] == "info"]

    # Display errors
    if errors:
        logger.error("\nErrors Found:")
        for error in errors:
            logger.print(f"   [{error['category']}] {error['message']}")

    # Display warnings
    if warnings:
        logger.warning("\nWarnings:")
        for warning in warnings:
            logger.print(f"   [{warning['category']}] {warning['message']}")

    # Display info messages
    if info:
        logger.info(f"\nInformation:")
        for item in info:
            logger.print(f"   [{item['category']}] {item['message']}")
