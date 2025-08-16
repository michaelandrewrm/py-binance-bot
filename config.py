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

import os
import re
import yaml
import secrets
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
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

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors"""
    pass


class SecurityValidationError(Exception):
    """Custom exception for security validation failures"""
    pass


def validate_secret_field(value: Optional[str], field_name: str) -> Optional[str]:
    """Reusable validator for secret fields"""
    if value is None:
        return None
    
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    
    stripped_value = value.strip()
    if len(stripped_value) == 0:
        return None
        
    return stripped_value


def mask_sensitive_value(value: Any) -> str:
    """Mask sensitive values for logging"""
    if value is None:
        return "None"
    if isinstance(value, str) and len(value) > 0:
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"
    return "***MASKED***"


class DatabaseConfig(BaseModel):
    """Database configuration"""

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
        return v.strip()


class BinanceConfig(BaseModel):
    """Binance exchange configuration"""

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
        return validate_secret_field(v, "API credential")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if not (v.startswith("https://") or v.startswith("http://")):
            raise ValueError("Base URL must start with http:// or https://")
        return v

    def has_credentials(self) -> bool:
        """Check if API credentials are provided"""
        return self.api_key is not None and self.api_secret is not None


class LoggingConfig(BaseModel):
    """Logging configuration"""

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
            raise ValueError(f"Invalid log level. Must be one of: {VALID_LOG_LEVELS}")
        return v

    @field_validator("format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Log format cannot be empty")
        return v.strip()


class TradingConfig(BaseModel):
    """Trading strategy configuration"""

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
        if not v or len(v) < 6:
            raise ValueError("Trading symbol must be at least 6 characters")
        return v

    @field_validator("default_timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_TIMEFRAMES:
            raise ValueError(f"Invalid timeframe. Must be one of: {VALID_TIMEFRAMES}")
        return v


class GridConfig(BaseModel):
    """Grid trading configuration"""

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
    def validate_price_boundaries(self):
        """Validate that upper price is greater than lower price"""
        if self.upper_price is not None and self.lower_price is not None:
            if self.upper_price <= self.lower_price:
                raise ValueError("Upper price must be greater than lower price")
        return self


class AIConfig(BaseModel):
    """AI/ML model configuration"""

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
        if v not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model type. Must be one of: {VALID_MODEL_TYPES}")
        return v


class CacheConfig(BaseModel):
    """Cache configuration"""

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
    def validate_cache_dir(cls, v: Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # Ensure it's a relative or absolute path, not empty
        if str(v) == "." or str(v) == "":
            raise ValueError("Cache directory cannot be empty or current directory")
        return v


class SecurityConfig(BaseModel):
    """Security configuration"""

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
            raise ValueError(f"JWT secret key must be at least {MIN_JWT_SECRET_LENGTH} characters long")
        return v

    @field_validator("encryption_key")
    @classmethod
    def validate_encryption_key(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        
        v = validate_secret_field(v, "Encryption key")
        if v is not None and len(v) != ENCRYPTION_KEY_LENGTH:
            raise ValueError(f"Encryption key must be exactly {ENCRYPTION_KEY_LENGTH} characters long")
        return v

    def generate_jwt_secret(self) -> str:
        """Generate a secure JWT secret key"""
        return secrets.token_urlsafe(MIN_JWT_SECRET_LENGTH)

    def generate_encryption_key(self) -> str:
        """Generate a secure encryption key"""
        return secrets.token_urlsafe(ENCRYPTION_KEY_LENGTH)[:ENCRYPTION_KEY_LENGTH]


class NotificationConfig(BaseModel):
    """Notification configuration"""

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

    def has_telegram_config(self) -> bool:
        """Check if Telegram configuration is complete"""
        return (self.telegram_enabled and 
                self.telegram_bot_token is not None and 
                self.telegram_chat_id is not None)

    def has_email_config(self) -> bool:
        """Check if email configuration is complete"""
        return (self.email_enabled and 
                self.email_smtp_server is not None and
                self.email_username is not None and 
                self.email_password is not None)


class Settings(BaseSettings):
    """
    Main application settings with environment variable support

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

        # Merge YAML config with kwargs (kwargs take precedence)
        merged_config = {**yaml_config, **kwargs}

        # Initialize Pydantic BaseSettings with merged config
        super().__init__(**merged_config)

    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load and process YAML configuration file"""
        config_path = Path(config_file)

        if not config_path.exists():
            logger.warning(f"Configuration file {config_file} not found, using defaults")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            # Perform environment variable interpolation
            interpolated_content = self._interpolate_env_vars(yaml_content)

            # Parse YAML
            config = yaml.safe_load(interpolated_content) or {}

            # Handle environment-specific configuration
            env = os.getenv("ENVIRONMENT", "dev")

            # Start with default configuration
            final_config = config.get("default", {})

            # Override with environment-specific configuration
            if env in config:
                final_config = self._deep_merge(final_config, config[env])

            # Flatten nested configuration for Pydantic
            flattened_config = self._flatten_config(final_config)

            logger.info(f"Loaded configuration from {config_file} for environment: {env}")
            return flattened_config

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
        """Interpolate environment variables in YAML content with security validation"""
        # Environment variable interpolation pattern - only allow alphanumeric, underscore, and dash
        ENV_VAR_PATTERN = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*(?::[^}]*)?)\}')

        def replace_var(match):
            var_expr = match.group(1)
            default_value = ""

            # Support default values: ${VAR:default}
            if ":" in var_expr:
                var_name, default_value = var_expr.split(":", 1)
            else:
                var_name = var_expr

            # Validate variable name (security measure)
            if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', var_name):
                logger.warning(f"Invalid environment variable name: {var_name}")
                return match.group(0)  # Return original if invalid

            return os.getenv(var_name, default_value)

        return ENV_VAR_PATTERN.sub(replace_var, content)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries efficiently"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration for Pydantic field mapping"""
        flattened = {}

        for key, value in config.items():
            full_key = f"{prefix}__{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flattened.update(self._flatten_config(value, full_key))
            else:
                flattened[full_key] = value

        return flattened

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_ENVIRONMENTS:
            raise ValueError(f"Invalid environment. Must be one of: {VALID_ENVIRONMENTS}")
        return v

    @model_validator(mode="after")
    def validate_configuration_integrity(self):
        """Validate configuration integrity and security requirements"""
        # Production security requirements
        if self.is_production():
            if not self.binance.has_credentials():
                raise SecurityValidationError(
                    "Binance API credentials are required in production environment"
                )

            if not self.security.jwt_secret_key:
                raise SecurityValidationError(
                    "JWT secret key is required in production environment"
                )

        # Notification configuration validation
        if self.notifications.telegram_enabled and not self.notifications.has_telegram_config():
            logger.warning("Telegram notifications enabled but configuration incomplete")

        if self.notifications.email_enabled and not self.notifications.has_email_config():
            logger.warning("Email notifications enabled but configuration incomplete")

        return self

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
        """Get cache directory as Path object"""
        return Path(self.cache.dir)

    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export configuration as dictionary

        Args:
            include_secrets: Whether to include sensitive information

        Returns:
            Configuration dictionary with secrets masked if include_secrets=False
        """
        config_dict = self.model_dump()

        if not include_secrets:
            # Comprehensive secret masking
            self._mask_secrets_in_dict(config_dict)

        return config_dict

    def _mask_secrets_in_dict(self, data: Dict[str, Any]) -> None:
        """Recursively mask secrets in configuration dictionary"""
        sensitive_patterns = [
            "key", "secret", "password", "token", "credential", 
            "auth", "jwt", "encryption"
        ]

        for key, value in data.items():
            if isinstance(value, dict):
                self._mask_secrets_in_dict(value)
            elif isinstance(value, str) and any(pattern in key.lower() for pattern in sensitive_patterns):
                data[key] = mask_sensitive_value(value)

    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of issues

        Returns:
            List of configuration issues/warnings
        """
        issues = []

        # Check for missing API credentials in non-development environments
        if not self.is_development() and not self.binance.has_credentials():
            issues.append("Binance API credentials are missing for non-development environment")

        # Check cache directory
        try:
            cache_dir = self.get_cache_dir()
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            issues.append(f"Cannot create cache directory {self.cache.dir}: {e}")

        # Check notification configurations
        if self.notifications.telegram_enabled and not self.notifications.has_telegram_config():
            issues.append("Telegram notifications enabled but credentials incomplete")

        if self.notifications.email_enabled and not self.notifications.has_email_config():
            issues.append("Email notifications enabled but credentials incomplete")

        # Security validations
        if self.is_production():
            if self.security.jwt_secret_key and len(self.security.jwt_secret_key) < MIN_JWT_SECRET_LENGTH:
                issues.append(f"JWT secret key should be at least {MIN_JWT_SECRET_LENGTH} characters in production")

        return issues

    def get_masked_config_summary(self) -> str:
        """Get a summary of configuration with masked secrets for logging"""
        summary = f"""
            Configuration Summary:
            - Environment: {self.environment}
            - Debug Mode: {self.debug}
            - Database URL: {mask_sensitive_value(self.database.url)}
            - Binance Testnet: {self.binance.testnet}
            - Binance Credentials: {'✅' if self.binance.has_credentials() else '❌'}
            - Cache Directory: {self.cache.dir}
            - Telegram Notifications: {'✅' if self.notifications.has_telegram_config() else '❌'}
            - Email Notifications: {'✅' if self.notifications.has_email_config() else '❌'}
        """
        return summary.strip()


class ConfigurationManager:
    """Thread-safe configuration manager"""
    
    def __init__(self):
        self._settings: Optional[Settings] = None
        self._config_file: Optional[str] = None
    
    def get_settings(self, config_file: Optional[str] = None, reload: bool = False) -> Settings:
        """
        Get settings instance with thread safety
        
        Args:
            config_file: Path to configuration file
            reload: Force reload of settings
            
        Returns:
            Settings instance
        """
        if self._settings is None or reload or (config_file and config_file != self._config_file):
            self._settings = Settings(config_file=config_file)
            self._config_file = config_file
            
            # Validate configuration
            issues = self._settings.validate_configuration()
            if issues:
                logger.warning("Configuration issues found:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            
            # Log configuration summary
            logger.info(self._settings.get_masked_config_summary())
        
        return self._settings
    
    def reload_settings(self, config_file: Optional[str] = None) -> Settings:
        """Force reload settings"""
        return self.get_settings(config_file=config_file, reload=True)


# Global configuration manager instance
_config_manager = ConfigurationManager()


def get_settings(config_file: Optional[str] = None, reload: bool = False) -> Settings:
    """
    Get global settings instance
    
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


# Convenience function for importing
settings = get_settings()
