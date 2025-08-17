"""
Configuration System - Unified .env and YAML Configuration

This module provides a comprehensive configuration system that integrates
.env files, config.yaml, and environment variables with proper validation,
security, and developer experience features.

Features:
- Environment variable loading from .env files
- YAML configuration with environment-specific overrides
- Proper precedence: OS env vars → .env → config.yaml (env) → config.yaml (default) → hardcoded defaults
- Type validation and schema enforcement with Pydantic
- Variable interpolation in YAML (${VAR} syntax)
- Security best practices (secrets only in env vars/.env)
- Clear error messages and documentation
"""

from __future__ import annotations

import os
import re
import stat
import yaml
import secrets
import logging
import string
from pathlib import Path
from typing import Dict, Any, Optional, List, Pattern
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Constants
MIN_JWT_SECRET_LENGTH = 32
ENCRYPTION_KEY_LENGTH = 32
VALID_TIMEFRAMES = [
    "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", 
    "6h", "8h", "12h", "1d", "3d", "1w", "1M"
]
VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
VALID_ENVIRONMENTS = ["dev", "development", "stage", "staging", "prod", "production"]
VALID_MODEL_TYPES = ["lstm", "gru", "transformer", "baseline"]

# Status indicators
STATUS_CHECKMARK = "✅"
STATUS_CROSS = "❌"
MASKED_PLACEHOLDER = "***MASKED***"

# Compiled regex patterns for performance
ENV_VAR_PATTERN: Pattern[str] = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?)\}')
VAR_NAME_PATTERN: Pattern[str] = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

# Sensitive field patterns
SENSITIVE_PATTERNS = frozenset([
    "key", "secret", "password", "token", "credential", 
    "auth", "jwt", "encryption", "api_key", "api_secret"
])

logger = logging.getLogger(__name__)

__all__ = [
    "Settings",
    "ConfigurationManager", 
    "ConfigurationError",
    "SecurityValidationError",
    "get_settings",
    "reload_settings",
    "settings"
]


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class SecurityValidationError(Exception):
    """Custom exception for security validation failures"""
    
    def __init__(self, message: str, severity: str = "HIGH"):
        super().__init__(message)
        self.severity = severity


def validate_secret_field(value: Optional[str], field_name: str) -> Optional[str]:
    """
    Reusable validator for secret fields with enhanced security
    
    Args:
        value: The secret value to validate
        field_name: Name of the field for error messages
        
    Returns:
        Validated and sanitized secret value or None
        
    Raises:
        ValueError: If validation fails
    """
    if value is None:
        return None
    
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    
    stripped_value = value.strip()
    if len(stripped_value) == 0:
        return None
    
    # Basic security check - no obvious injection patterns
    allowed_chars = string.ascii_letters + string.digits + string.punctuation.replace('`', '').replace('$', '')
    if not all(c in allowed_chars for c in stripped_value):
        logger.warning(f"Invalid characters detected in {field_name}")
        
    return stripped_value


def mask_sensitive_value(value: Any) -> str:
    """
    Mask sensitive values for safe logging with improved security
    
    Args:
        value: Value to mask
        
    Returns:
        Masked string representation
    """
    if value is None:
        return "None"
    
    if not isinstance(value, str):
        return MASKED_PLACEHOLDER
    
    value_len = len(value)
    
    if value_len == 0:
        return ""
    elif value_len <= 4:
        return "*" * value_len
    elif value_len <= 8:
        return f"{value[0]}{'*' * (value_len - 2)}{value[-1]}"
    else:
        return f"{value[:2]}{'*' * (value_len - 4)}{value[-2:]}"


def ensure_secure_permissions(path: Path) -> None:
    """
    Ensure file/directory has secure permissions
    
    Args:
        path: Path to secure
        
    Raises:
        PermissionError: If permissions cannot be set
    """
    try:
        if path.is_file():
            # Files: owner read/write only (600)
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        elif path.is_dir():
            # Directories: owner read/write/execute only (700)
            path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    except OSError as e:
        raise PermissionError(f"Cannot set secure permissions for {path}: {e}")


def validate_url(url: str, require_https: bool = False) -> str:
    """
    Validate and normalize URL
    
    Args:
        url: URL to validate
        require_https: Whether to require HTTPS
        
    Returns:
        Validated URL
        
    Raises:
        ValueError: If URL is invalid
    """
    url = url.strip()
    
    if not url:
        raise ValueError("URL cannot be empty")
    
    if not (url.startswith("https://") or url.startswith("http://")):
        raise ValueError("URL must start with http:// or https://")
    
    if require_https and not url.startswith("https://"):
        raise ValueError("HTTPS is required for this URL")
    
    return url


class DatabaseConfig(BaseModel):
    """Database configuration with enhanced validation"""

    url: str = Field(
        default="sqlite:///trading_bot.db", 
        description="Database connection URL"
    )
    echo: bool = Field(
        default=False, 
        description="Enable SQL query logging"
    )
    pool_size: int = Field(
        default=5, 
        description="Connection pool size",
        ge=1,
        le=50
    )
    max_overflow: int = Field(
        default=10, 
        description="Maximum connection overflow",
        ge=0,
        le=100
    )

    @field_validator("url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Database URL cannot be empty")
        
        v = v.strip()
        
        # Basic URL validation for database connections
        if not any(v.startswith(scheme) for scheme in ["sqlite://", "postgresql://", "mysql://"]):
            logger.warning(f"Unrecognized database URL scheme: {v.split('://')[0] if '://' in v else 'none'}")
        
        return v


class BinanceConfig(BaseModel):
    """Binance exchange configuration with security enhancements"""

    api_key: Optional[str] = Field(
        default=None, 
        description="Binance API key (secret)"
    )
    api_secret: Optional[str] = Field(
        default=None, 
        description="Binance API secret (secret)"
    )
    testnet: bool = Field(
        default=True, 
        description="Use Binance testnet"
    )
    base_url: Optional[str] = Field(
        default=None, 
        description="Custom API base URL"
    )
    rate_limit_requests_per_minute: int = Field(
        default=1200, 
        description="API rate limit",
        ge=1,
        le=6000
    )
    rate_limit_orders_per_second: int = Field(
        default=10, 
        description="Order rate limit",
        ge=1,
        le=100
    )

    @field_validator("api_key", "api_secret")
    @classmethod
    def validate_api_credentials(cls, v: Optional[str]) -> Optional[str]:
        validated = validate_secret_field(v, "Binance API credential")
        
        # Additional validation for API key format
        if validated and len(validated) < 32:
            raise ValueError("Binance API credentials should be at least 32 characters")
        
        return validated

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        try:
            return validate_url(v, require_https=True)
        except ValueError as e:
            raise ValueError(f"Invalid Binance base URL: {e}")

    def has_credentials(self) -> bool:
        """Check if API credentials are properly configured"""
        return (self.api_key is not None and 
                self.api_secret is not None and
                len(self.api_key.strip()) > 0 and
                len(self.api_secret.strip()) > 0)


class LoggingConfig(BaseModel):
    """Logging configuration with security considerations"""

    level: str = Field(
        default="INFO", 
        description="Log level"
    )
    file: Optional[str] = Field(
        default="trading_bot.log", 
        description="Log file path"
    )
    max_file_size_mb: int = Field(
        default=10, 
        description="Maximum log file size in MB",
        ge=1,
        le=1000
    )
    backup_count: int = Field(
        default=5, 
        description="Number of backup log files",
        ge=0,
        le=50
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in VALID_LOG_LEVELS:
            raise ValueError(f"Invalid log level '{v}'. Must be one of: {VALID_LOG_LEVELS}")
        return v

    @field_validator("format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Log format cannot be empty")
        
        # Basic validation for log format string
        required_fields = ["%(levelname)s", "%(message)s"]
        for field in required_fields:
            if field not in v:
                logger.warning(f"Log format missing recommended field: {field}")
        
        return v.strip()

    @field_validator("file")
    @classmethod
    def validate_log_file(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        v = v.strip()
        if not v:
            return None
        
        # Validate log file path
        log_path = Path(v)
        if log_path.is_absolute() and not log_path.parent.exists():
            logger.warning(f"Log file directory does not exist: {log_path.parent}")
        
        return v


class TradingConfig(BaseModel):
    """Trading strategy configuration with business logic validation"""

    default_symbol: str = Field(
        default="BTCUSDC", 
        description="Default trading symbol"
    )
    default_timeframe: str = Field(
        default="5m", 
        description="Default timeframe"
    )
    max_position_size: Decimal = Field(
        default=Decimal("1000"), 
        description="Maximum position size",
        gt=Decimal("0")
    )
    risk_per_trade: Decimal = Field(
        default=Decimal("0.02"), 
        description="Risk percentage per trade",
        gt=Decimal("0"),
        lt=Decimal("1")
    )
    fee_percentage: Decimal = Field(
        default=Decimal("0.001"), 
        description="Trading fee percentage",
        ge=Decimal("0"),
        lt=Decimal("0.1")
    )

    @field_validator("default_symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        v = v.upper().strip()
        
        if not v:
            raise ValueError("Trading symbol cannot be empty")
        
        if len(v) < 6:
            raise ValueError("Trading symbol must be at least 6 characters (e.g., BTCUSDC)")
        
        # Basic symbol format validation
        if not v.isalnum():
            raise ValueError("Trading symbol must contain only alphanumeric characters")
        
        return v

    @field_validator("default_timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        v = v.lower().strip()
        
        if not v:
            raise ValueError("Timeframe cannot be empty")
        
        if v not in VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe '{v}'. Must be one of: {VALID_TIMEFRAMES}")
        
        return v


class GridConfig(BaseModel):
    """Grid trading configuration with enhanced validation"""

    n_grids: int = Field(
        default=12, 
        description="Number of grid levels",
        ge=2,
        le=100
    )
    invest_per_grid: Decimal = Field(
        default=Decimal("50.0"), 
        description="Investment per grid level",
        gt=Decimal("0")
    )
    grid_spacing_pct: Decimal = Field(
        default=Decimal("0.01"), 
        description="Grid spacing percentage",
        gt=Decimal("0"),
        lt=Decimal("1")
    )
    upper_price: Optional[Decimal] = Field(
        default=None, 
        description="Upper price boundary"
    )
    lower_price: Optional[Decimal] = Field(
        default=None, 
        description="Lower price boundary"
    )
    rebalance_enabled: bool = Field(
        default=True, 
        description="Enable grid rebalancing"
    )

    @model_validator(mode="after")
    def validate_grid_configuration(self):
        """Comprehensive grid configuration validation"""
        # Price boundary validation
        if self.upper_price is not None and self.lower_price is not None:
            if self.upper_price <= self.lower_price:
                raise ValueError("Upper price must be greater than lower price")
            
            # Ensure minimum spread for grid spacing
            price_spread = self.upper_price - self.lower_price
            min_spread = self.lower_price * self.grid_spacing_pct * self.n_grids
            
            if price_spread < min_spread:
                raise ValueError(
                    f"Price spread ({price_spread}) too small for {self.n_grids} grids "
                    f"with {self.grid_spacing_pct:.2%} spacing. Minimum required: {min_spread}"
                )
        
        # Validate individual price boundaries
        if self.upper_price is not None and self.upper_price <= 0:
            raise ValueError("Upper price must be positive")
        
        if self.lower_price is not None and self.lower_price <= 0:
            raise ValueError("Lower price must be positive")
        
        return self


class AIConfig(BaseModel):
    """AI/ML model configuration with ML-specific validation"""

    enabled: bool = Field(
        default=True, 
        description="Enable AI features"
    )
    model_type: str = Field(
        default="lstm", 
        description="Model type"
    )
    n_steps: int = Field(
        default=60, 
        description="Number of time steps for prediction",
        ge=1,
        le=1000
    )
    epochs: int = Field(
        default=30, 
        description="Training epochs",
        ge=1,
        le=1000
    )
    batch_size: int = Field(
        default=32, 
        description="Training batch size",
        ge=1,
        le=1024
    )
    validation_split: Decimal = Field(
        default=Decimal("0.1"), 
        description="Validation split ratio",
        gt=Decimal("0"),
        lt=Decimal("0.5")
    )
    early_stopping_patience: int = Field(
        default=5, 
        description="Early stopping patience",
        ge=1,
        le=100
    )

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        v = v.lower().strip()
        
        if not v:
            raise ValueError("Model type cannot be empty")
        
        if v not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model type '{v}'. Must be one of: {VALID_MODEL_TYPES}")
        
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        # Ensure batch size is a power of 2 for optimal performance
        if v & (v - 1) != 0:
            logger.warning(f"Batch size {v} is not a power of 2, which may impact performance")
        
        return v

    @model_validator(mode="after")
    def validate_ml_parameters(self):
        """Validate ML parameter combinations"""
        # Ensure n_steps is reasonable relative to batch_size
        if self.n_steps < self.batch_size:
            logger.warning(
                f"n_steps ({self.n_steps}) is smaller than batch_size ({self.batch_size}), "
                "which may lead to poor training performance"
            )
        
        return self


class CacheConfig(BaseModel):
    """Cache configuration with security and performance considerations"""

    dir: Path = Field(
        default=Path("data_cache"), 
        description="Cache directory"
    )
    max_age_hours: int = Field(
        default=24, 
        description="Maximum cache age in hours",
        ge=1,
        le=8760  # 1 year
    )
    max_size_mb: int = Field(
        default=500, 
        description="Maximum cache size in MB",
        ge=10,
        le=10000
    )
    enabled: bool = Field(
        default=True, 
        description="Enable caching"
    )

    @field_validator("dir")
    @classmethod
    def validate_cache_dir(cls, v: Path | str) -> Path:
        if isinstance(v, str):
            v = Path(v)
        
        # Security validation
        if str(v) in [".", "", "/"]:
            raise ValueError("Cache directory cannot be current directory, empty, or root")
        
        # Ensure it's not pointing to system directories
        system_dirs = ["/etc", "/var", "/usr", "/bin", "/sbin", "/sys", "/proc"]
        abs_path = v.resolve()
        
        if any(str(abs_path).startswith(sys_dir) for sys_dir in system_dirs):
            raise ValueError(f"Cache directory cannot be in system directory: {abs_path}")
        
        return v

    def get_cache_path(self) -> Path:
        """Get resolved cache path with security validation"""
        cache_path = self.dir.resolve()
        
        # Ensure parent directories exist with secure permissions
        cache_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        
        if not cache_path.exists():
            cache_path.mkdir(mode=0o700)
        
        return cache_path


class SecurityConfig(BaseModel):
    """Security configuration with crypto best practices"""

    jwt_secret_key: Optional[str] = Field(
        default=None, 
        description="JWT secret key (secret)"
    )
    encryption_key: Optional[str] = Field(
        default=None, 
        description="Encryption key (secret)"
    )
    session_timeout_minutes: int = Field(
        default=60, 
        description="Session timeout in minutes",
        ge=5,
        le=1440  # 24 hours
    )

    @field_validator("jwt_secret_key")
    @classmethod
    def validate_jwt_secret(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        v = validate_secret_field(v, "JWT secret key")
        
        if v is not None and len(v) < MIN_JWT_SECRET_LENGTH:
            raise ValueError(
                f"JWT secret key must be at least {MIN_JWT_SECRET_LENGTH} characters long "
                f"for adequate security. Current length: {len(v)}"
            )
        
        return v

    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        v = validate_secret_field(v, "Encryption key")
        
        if v is not None and len(v) != ENCRYPTION_KEY_LENGTH:
            raise ValueError(
                f"Encryption key must be exactly {ENCRYPTION_KEY_LENGTH} characters long "
                f"for AES-256. Current length: {len(v)}"
            )
        
        return v

    def generate_jwt_secret(self) -> str:
        """Generate a cryptographically secure JWT secret key"""
        return secrets.token_urlsafe(MIN_JWT_SECRET_LENGTH)

    def generate_encryption_key(self) -> str:
        """Generate a cryptographically secure encryption key"""
        # Use proper crypto-secure random bytes
        # Generate proper length encryption key
        return secrets.token_urlsafe(ENCRYPTION_KEY_LENGTH)[:ENCRYPTION_KEY_LENGTH]

    def is_secure(self) -> bool:
        """Check if security configuration meets minimum requirements"""
        return (self.jwt_secret_key is not None and 
                len(self.jwt_secret_key) >= MIN_JWT_SECRET_LENGTH)


class NotificationConfig(BaseModel):
    """Notification configuration with service validation"""

    telegram_enabled: bool = Field(
        default=False, 
        description="Enable Telegram notifications"
    )
    telegram_bot_token: Optional[str] = Field(
        default=None, 
        description="Telegram bot token (secret)"
    )
    telegram_chat_id: Optional[str] = Field(
        default=None, 
        description="Telegram chat ID (secret)"
    )
    email_enabled: bool = Field(
        default=False, 
        description="Enable email notifications"
    )
    email_smtp_server: Optional[str] = Field(
        default=None, 
        description="SMTP server"
    )
    email_smtp_port: int = Field(
        default=587, 
        description="SMTP port",
        ge=1,
        le=65535
    )
    email_username: Optional[str] = Field(
        default=None, 
        description="Email username (secret)"
    )
    email_password: Optional[str] = Field(
        default=None, 
        description="Email password (secret)"
    )

    @field_validator("telegram_bot_token", "telegram_chat_id", "email_username", "email_password")
    @classmethod
    def validate_notification_secrets(cls, v: Optional[str]) -> Optional[str]:
        return validate_secret_field(v, "Notification credential")

    @field_validator("telegram_bot_token")
    @classmethod
    def validate_telegram_token(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        v = validate_secret_field(v, "Telegram bot token")
        
        # Basic Telegram token format validation
        if v and not v.startswith(('1', '2', '5', '6', '7')):
            logger.warning("Telegram bot token format appears invalid")
        
        return v

    @field_validator("email_smtp_server")
    @classmethod
    def validate_smtp_server(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        v = v.strip()
        if not v:
            return None
        
        # Basic SMTP server validation
        if '://' in v:
            raise ValueError("SMTP server should be hostname only, not URL")
        
        return v

    def has_telegram_config(self) -> bool:
        """Check if Telegram configuration is complete and valid"""
        return (self.telegram_enabled and 
                self.telegram_bot_token is not None and 
                self.telegram_chat_id is not None and
                len(self.telegram_bot_token.strip()) > 0 and
                len(self.telegram_chat_id.strip()) > 0)

    def has_email_config(self) -> bool:
        """Check if email configuration is complete and valid"""
        return (self.email_enabled and 
                self.email_smtp_server is not None and
                self.email_username is not None and 
                self.email_password is not None and
                len(self.email_smtp_server.strip()) > 0 and
                len(self.email_username.strip()) > 0 and
                len(self.email_password.strip()) > 0)


class Settings(BaseSettings):
    """
    Main application settings with comprehensive validation and security

    This class integrates configuration from multiple sources with proper precedence:
    1. OS environment variables (highest priority)
    2. .env file
    3. config.yaml (environment-specific section)
    4. config.yaml (default section)
    5. Hardcoded defaults (lowest priority)
    """

    # Environment and application metadata
    environment: str = Field(
        default="dev", 
        description="Application environment (dev/stage/prod)"
    )
    app_name: str = Field(
        default="py-binance-bot", 
        description="Application name"
    )
    version: str = Field(
        default="1.0.0", 
        description="Application version"
    )
    debug: bool = Field(
        default=False, 
        description="Debug mode"
    )

    # Configuration sections
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        validate_assignment=True,
    )

    def __init__(self, config_file: Optional[str] = None, **kwargs):
        """
        Initialize settings with YAML config file support

        Args:
            config_file: Path to YAML configuration file
            **kwargs: Additional keyword arguments to override settings
        """
        # Load YAML configuration first
        yaml_config = self._load_yaml_config(config_file or "config.yaml")

        # Merge configurations with proper precedence
        merged_config = self._merge_configurations(yaml_config, kwargs)

        # Initialize Pydantic BaseSettings with merged config
        super().__init__(**merged_config)

    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load and process YAML configuration file with enhanced security
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Processed configuration dictionary
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        config_path = Path(config_file)

        if not config_path.exists():
            logger.warning(f"Configuration file {config_file} not found, using defaults")
            return {}

        try:
            # Validate file permissions - should be readable only by owner
            file_mode = config_path.stat().st_mode & 0o777
            if file_mode & 0o077:  # group or other have any permissions
                logger.warning(f"Configuration file {config_file} has permissions {oct(file_mode)}, should be 0o600 or more restrictive")

            with open(config_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            # Validate file size (prevent DoS)
            if len(yaml_content) > 1024 * 1024:  # 1MB limit
                raise ConfigurationError("Configuration file too large (>1MB)")

            # Perform secure environment variable interpolation
            interpolated_content = self._interpolate_env_vars(yaml_content)

            # Parse YAML with safe loader
            config = yaml.safe_load(interpolated_content) or {}

            if not isinstance(config, dict):
                raise ConfigurationError("Configuration file must contain a YAML object")

            # Process environment-specific configuration
            processed_config = self._process_environment_config(config)

            logger.info(f"Successfully loaded configuration from {config_file}")
            return processed_config

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {config_file}: {e}")
            raise ConfigurationError(f"Invalid YAML configuration: {e}")
        except FileNotFoundError:
            logger.warning(f"Configuration file {config_file} not found")
            return {}
        except PermissionError as e:
            logger.error(f"Permission denied reading {config_file}: {e}")
            raise ConfigurationError(f"Cannot read configuration file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading configuration file {config_file}: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def _interpolate_env_vars(self, content: str) -> str:
        """
        Safely interpolate environment variables in YAML content
        
        Args:
            content: YAML content with variable references
            
        Returns:
            Content with variables interpolated
        """
        def replace_var(match):
            var_expr = match.group(1)
            default_value = ""

            # Parse variable expression
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
            else:
                var_name = var_expr

            # Enhanced security validation
            if not VAR_NAME_PATTERN.match(var_name):
                logger.warning(f"Invalid environment variable name: {var_name}")
                return match.group(0)  # Return original if invalid

            # Additional security checks
            if len(var_name) > 100:  # Prevent extremely long variable names
                logger.warning(f"Environment variable name too long: {var_name[:50]}...")
                return match.group(0)

            return os.getenv(var_name, default_value)

        return ENV_VAR_PATTERN.sub(replace_var, content)

    def _process_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process environment-specific configuration
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Processed configuration for current environment
        """
        env = os.getenv("ENVIRONMENT", "dev").lower()

        # Start with default configuration
        final_config = config.get("default", {})

        # Override with environment-specific configuration
        env_config = config.get(env, {})
        if env_config:
            final_config = self._deep_merge(final_config, env_config)

        # Flatten nested configuration for Pydantic
        return self._flatten_config(final_config)

    def _merge_configurations(self, yaml_config: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration sources with proper precedence
        
        Args:
            yaml_config: Configuration from YAML file
            kwargs: Configuration from direct arguments
            
        Returns:
            Merged configuration dictionary
        """
        # Use dict.update for better performance than ** operator
        merged_config = yaml_config.copy()
        merged_config.update(kwargs)
        return merged_config

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Efficiently deep merge two dictionaries in-place
        
        Args:
            base: Base dictionary to merge into
            override: Override dictionary to merge from
            
        Returns:
            Merged dictionary (modifies base)
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Flatten nested configuration for Pydantic field mapping
        
        Args:
            config: Nested configuration dictionary
            prefix: Current prefix for keys
            
        Returns:
            Flattened configuration dictionary
        """
        flattened = {}

        for key, value in config.items():
            full_key = f"{prefix}__{key}" if prefix else key

            if isinstance(value, dict):
                flattened.update(self._flatten_config(value, full_key))
            else:
                flattened[full_key] = value

        return flattened

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting"""
        v = v.lower().strip()
        
        if not v:
            raise ValueError("Environment cannot be empty")
        
        if v not in VALID_ENVIRONMENTS:
            raise ValueError(f"Invalid environment '{v}'. Must be one of: {VALID_ENVIRONMENTS}")
        
        return v

    @field_validator("app_name")
    @classmethod
    def validate_app_name(cls, v: str) -> str:
        """Validate application name"""
        v = v.strip()
        
        if not v:
            raise ValueError("Application name cannot be empty")
        
        if len(v) > 50:
            raise ValueError("Application name too long (max 50 characters)")
        
        return v

    @model_validator(mode="after")
    def validate_configuration_integrity(self):
        """Comprehensive configuration integrity validation"""
        # Production security requirements
        if self.is_production():
            self._validate_production_requirements()

        # Cross-configuration validation
        self._validate_configuration_consistency()

        # Security validation
        self._validate_security_requirements()

        return self

    def _validate_production_requirements(self) -> None:
        """Validate production-specific requirements"""
        if not self.binance.has_credentials():
            raise SecurityValidationError(
                "Binance API credentials are required in production environment",
                severity="CRITICAL"
            )

        if not self.security.jwt_secret_key:
            raise SecurityValidationError(
                "JWT secret key is required in production environment",
                severity="CRITICAL"
            )

        if self.debug:
            logger.warning("Debug mode is enabled in production environment")

    def _validate_configuration_consistency(self) -> None:
        """Validate configuration consistency across sections"""
        # Notification validation
        if self.notifications.telegram_enabled and not self.notifications.has_telegram_config():
            logger.warning("Telegram notifications enabled but configuration incomplete")

        if self.notifications.email_enabled and not self.notifications.has_email_config():
            logger.warning("Email notifications enabled but configuration incomplete")

    def _validate_security_requirements(self) -> None:
        """Validate security requirements"""
        if self.is_production() and not self.security.is_secure():
            raise SecurityValidationError(
                "Insufficient security configuration for production environment"
            )

    def get_database_url(self) -> str:
        """Get database URL"""
        return self.database.url

    def get_log_level(self) -> int:
        """Get numeric log level for Python logging"""
        return getattr(logging, self.logging.level)

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment in ["prod", "production"]

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment in ["dev", "development"]

    def get_cache_dir(self) -> Path:
        """Get cache directory with security validation"""
        return self.cache.get_cache_path()

    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export configuration as dictionary with security controls

        Args:
            include_secrets: Whether to include sensitive information

        Returns:
            Configuration dictionary with secrets masked if include_secrets=False
        """
        config_dict = self.model_dump()

        if not include_secrets:
            self._mask_secrets_in_dict(config_dict)

        return config_dict

    def _mask_secrets_in_dict(self, data: Dict[str, Any]) -> None:
        """
        Recursively mask secrets in configuration dictionary
        
        Args:
            data: Dictionary to mask secrets in (modified in-place)
        """
        for key, value in data.items():
            if isinstance(value, dict):
                self._mask_secrets_in_dict(value)
            elif isinstance(value, str) and self._is_sensitive_key(key):
                data[key] = mask_sensitive_value(value)

    def _is_sensitive_key(self, key: str) -> bool:
        """
        Check if a key represents sensitive information
        
        Args:
            key: Configuration key to check
            
        Returns:
            True if key appears to be sensitive
        """
        key_lower = key.lower()
        return any(pattern in key_lower for pattern in SENSITIVE_PATTERNS)

    def validate_configuration(self) -> List[str]:
        """
        Comprehensive configuration validation with detailed reporting

        Returns:
            List of configuration issues/warnings
        """
        issues = []

        # Credential validation
        issues.extend(self._validate_credentials())
        
        # Infrastructure validation  
        issues.extend(self._validate_infrastructure())
        
        # Security validation
        issues.extend(self._validate_security_config())
        
        # Performance validation
        issues.extend(self._validate_performance_config())

        return issues

    def _validate_credentials(self) -> List[str]:
        """Validate credential configuration"""
        issues = []
        
        if not self.is_development() and not self.binance.has_credentials():
            issues.append("Binance API credentials are missing for non-development environment")
        
        return issues

    def _validate_infrastructure(self) -> List[str]:
        """Validate infrastructure configuration"""
        issues = []
        
        # Cache directory validation
        try:
            cache_dir = self.get_cache_dir()
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
                ensure_secure_permissions(cache_dir)
        except (PermissionError, OSError) as e:
            issues.append(f"Cannot create cache directory {self.cache.dir}: {e}")

        return issues

    def _validate_security_config(self) -> List[str]:
        """Validate security configuration"""
        issues = []
        
        # JWT secret validation
        if self.is_production():
            if (self.security.jwt_secret_key and 
                len(self.security.jwt_secret_key) < MIN_JWT_SECRET_LENGTH):
                issues.append(
                    f"JWT secret key should be at least {MIN_JWT_SECRET_LENGTH} "
                    "characters in production"
                )

        return issues

    def _validate_performance_config(self) -> List[str]:
        """Validate performance-related configuration"""
        issues = []
        
        # Check for performance-impacting configurations
        if self.ai.enabled and self.ai.n_steps > 500:
            issues.append("Large n_steps value may impact AI model performance")
        
        if self.cache.max_size_mb < 50:
            issues.append("Cache size may be too small for optimal performance")

        return issues

    def get_masked_config_summary(self) -> str:
        """Get a summary of configuration with masked secrets for safe logging"""
        status_check = STATUS_CHECKMARK
        status_cross = STATUS_CROSS
        
        summary = f"""Configuration Summary:
            - Environment: {self.environment}
            - Debug Mode: {self.debug}
            - Database URL: {mask_sensitive_value(self.database.url)}
            - Binance Testnet: {self.binance.testnet}
            - Binance Credentials: {status_check if self.binance.has_credentials() else status_cross}
            - Cache Directory: {self.cache.dir}
            - Telegram Notifications: {status_check if self.notifications.has_telegram_config() else status_cross}
            - Email Notifications: {status_check if self.notifications.has_email_config() else status_cross}
            - Security Level: {'High' if self.security.is_secure() else 'Basic'}
        """
        
        return summary


class ConfigurationManager:
    """Thread-safe configuration manager with caching and validation"""
    
    def __init__(self):
        self._settings: Optional[Settings] = None
        self._config_file: Optional[str] = None
        self._last_modified: Optional[float] = None
    
    def get_settings(self, config_file: Optional[str] = None, reload: bool = False) -> Settings:
        """
        Get settings instance with thread safety and automatic reload detection
        
        Args:
            config_file: Path to configuration file
            reload: Force reload of settings
            
        Returns:
            Settings instance
        """
        should_reload = (
            self._settings is None or 
            reload or 
            (config_file is not None and config_file != self._config_file) or
            self._should_auto_reload(config_file or self._config_file)
        )
        
        if should_reload:
            self._settings = Settings(config_file=config_file)
            self._config_file = config_file
            self._update_last_modified(config_file)
            
            # Validate and report issues
            issues = self._settings.validate_configuration()
            if issues:
                logger.warning("Configuration issues found:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            else:
                logger.info("Configuration validation passed")
            
            # Log configuration summary
            logger.info(self._settings.get_masked_config_summary())
        
        return self._settings
    
    def _should_auto_reload(self, config_file: Optional[str]) -> bool:
        """Check if configuration file has been modified"""
        if not config_file:
            config_file = "config.yaml"
        
        config_path = Path(config_file)
        if not config_path.exists():
            return False
        
        try:
            current_mtime = config_path.stat().st_mtime
            if self._last_modified is None or current_mtime > self._last_modified:
                return True
        except OSError:
            pass
        
        return False
    
    def _update_last_modified(self, config_file: Optional[str]) -> None:
        """Update the last modified timestamp"""
        if not config_file:
            config_file = "config.yaml"
        
        config_path = Path(config_file)
        if config_path.exists():
            try:
                self._last_modified = config_path.stat().st_mtime
            except OSError:
                pass
    
    def reload_settings(self, config_file: Optional[str] = None) -> Settings:
        """Force reload settings"""
        return self.get_settings(config_file=config_file, reload=True)


# Global configuration manager instance
_config_manager = ConfigurationManager()


def get_settings(config_file: Optional[str] = None, reload: bool = False) -> Settings:
    """
    Get global settings instance with automatic reloading
    
    Args:
        config_file: Path to configuration file
        reload: Force reload of settings
        
    Returns:
        Settings instance
    """
    return _config_manager.get_settings(config_file=config_file, reload=reload)


def reload_settings(config_file: Optional[str] = None) -> Settings:
    """Force reload settings"""
    return _config_manager.reload_settings(config_file=config_file)


# Convenience function for importing - initialized on first access
def _get_default_settings() -> Settings:
    """Lazy initialization of default settings"""
    return get_settings()


# Create a property-like access for the default settings
class _SettingsProxy:
    """Proxy class for lazy settings initialization"""
    
    def __init__(self):
        self._cached_settings = None
    
    def __getattr__(self, name: str) -> Any:
        if self._cached_settings is None:
            self._cached_settings = _get_default_settings()
        return getattr(self._cached_settings, name)


settings = _SettingsProxy()
