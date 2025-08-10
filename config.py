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
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from decimal import Decimal
from datetime import timedelta
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

logger = logging.getLogger(__name__)

# Environment variable interpolation pattern
ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: str = Field(default="sqlite:///trading_bot.db", description="Database connection URL")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum connection overflow")

class BinanceConfig(BaseModel):
    """Binance exchange configuration"""
    api_key: Optional[str] = Field(default=None, description="Binance API key (secret)")
    api_secret: Optional[str] = Field(default=None, description="Binance API secret (secret)")
    testnet: bool = Field(default=True, description="Use Binance testnet")
    base_url: Optional[str] = Field(default=None, description="Custom API base URL")
    rate_limit_requests_per_minute: int = Field(default=1200, description="API rate limit")
    rate_limit_orders_per_second: int = Field(default=10, description="Order rate limit")
    
    @field_validator('api_key', 'api_secret')
    @classmethod
    def validate_api_credentials(cls, v):
        if v is not None and len(v.strip()) == 0:
            return None
        return v

class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field(default="INFO", description="Log level")
    file: Optional[str] = Field(default="trading_bot.log", description="Log file path")
    max_file_size_mb: int = Field(default=10, description="Maximum log file size in MB")
    backup_count: int = Field(default=5, description="Number of backup log files")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    
    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

class TradingConfig(BaseModel):
    """Trading strategy configuration"""
    default_symbol: str = Field(default="BTCUSDC", description="Default trading symbol")
    default_timeframe: str = Field(default="5m", description="Default timeframe")
    max_position_size: Decimal = Field(default=Decimal("1000"), description="Maximum position size")
    risk_percentage: Decimal = Field(default=Decimal("0.02"), description="Risk percentage per trade")
    fee_percentage: Decimal = Field(default=Decimal("0.001"), description="Trading fee percentage")
    
    @field_validator('default_timeframe')
    @classmethod
    def validate_timeframe(cls, v):
        valid_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
        if v not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of: {valid_timeframes}")
        return v

class GridConfig(BaseModel):
    """Grid trading configuration"""
    n_grids: int = Field(default=12, description="Number of grid levels")
    invest_per_grid: Decimal = Field(default=Decimal("50.0"), description="Investment per grid level")
    grid_spacing_pct: Decimal = Field(default=Decimal("0.01"), description="Grid spacing percentage")
    upper_price: Optional[Decimal] = Field(default=None, description="Upper price boundary")
    lower_price: Optional[Decimal] = Field(default=None, description="Lower price boundary")
    rebalance_enabled: bool = Field(default=True, description="Enable grid rebalancing")
    
    @field_validator('n_grids')
    @classmethod
    def validate_n_grids(cls, v):
        if v < 2 or v > 100:
            raise ValueError("Number of grids must be between 2 and 100")
        return v

class AIConfig(BaseModel):
    """AI/ML model configuration"""
    enabled: bool = Field(default=True, description="Enable AI features")
    model_type: str = Field(default="lstm", description="Model type")
    n_steps: int = Field(default=60, description="Number of time steps for prediction")
    epochs: int = Field(default=30, description="Training epochs")
    batch_size: int = Field(default=32, description="Training batch size")
    validation_split: Decimal = Field(default=Decimal("0.1"), description="Validation split ratio")
    early_stopping_patience: int = Field(default=5, description="Early stopping patience")
    
    @field_validator('model_type')
    @classmethod
    def validate_model_type(cls, v):
        valid_types = ['lstm', 'gru', 'transformer', 'baseline']
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid model type. Must be one of: {valid_types}")
        return v.lower()

class CacheConfig(BaseModel):
    """Cache configuration"""
    dir: Path = Field(default=Path("data_cache"), description="Cache directory")
    max_age_hours: int = Field(default=24, description="Maximum cache age in hours")
    max_size_mb: int = Field(default=500, description="Maximum cache size in MB")
    enabled: bool = Field(default=True, description="Enable caching")

class SecurityConfig(BaseModel):
    """Security configuration"""
    jwt_secret_key: Optional[str] = Field(default=None, description="JWT secret key (secret)")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key (secret)")
    session_timeout_minutes: int = Field(default=60, description="Session timeout in minutes")
    
    @field_validator('encryption_key')
    @classmethod
    def validate_encryption_key(cls, v):
        if v is not None and len(v) != 32:
            raise ValueError("Encryption key must be exactly 32 characters long")
        return v

class NotificationConfig(BaseModel):
    """Notification configuration"""
    telegram_enabled: bool = Field(default=False, description="Enable Telegram notifications")
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram bot token (secret)")
    telegram_chat_id: Optional[str] = Field(default=None, description="Telegram chat ID (secret)")
    email_enabled: bool = Field(default=False, description="Enable email notifications")
    email_smtp_server: Optional[str] = Field(default=None, description="SMTP server")
    email_smtp_port: int = Field(default=587, description="SMTP port")
    email_username: Optional[str] = Field(default=None, description="Email username (secret)")
    email_password: Optional[str] = Field(default=None, description="Email password (secret)")

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
    environment: str = Field(default="dev", description="Application environment (dev/stage/prod)")
    app_name: str = Field(default="py-binance-bot", description="Application name")
    version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
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
            with open(config_path, 'r', encoding='utf-8') as f:
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
            
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file}: {e}")
            raise ValueError(f"Failed to load configuration: {e}")
    
    def _interpolate_env_vars(self, content: str) -> str:
        """Interpolate environment variables in YAML content"""
        def replace_var(match):
            var_name = match.group(1)
            default_value = ""
            
            # Support default values: ${VAR:default}
            if ":" in var_name:
                var_name, default_value = var_name.split(":", 1)
            
            return os.getenv(var_name, default_value)
        
        return ENV_VAR_PATTERN.sub(replace_var, content)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
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
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        valid_envs = ['dev', 'development', 'stage', 'staging', 'prod', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v.lower()
    
    @model_validator(mode='after')
    def validate_required_secrets(self):
        """Validate that required secrets are provided in production"""
        env = self.environment
        
        if env in ['prod', 'production']:
            # Check required secrets for production
            if not self.binance.api_key or not self.binance.api_secret:
                raise ValueError("Binance API credentials are required in production")
            
            if not self.security.jwt_secret_key:
                raise ValueError("JWT secret key is required in production")
        
        return self
    
    def get_database_url(self) -> str:
        """Get database URL with environment variable substitution"""
        return self.database.url
    
    def get_log_level(self) -> int:
        """Get numeric log level for Python logging"""
        return getattr(logging, self.logging.level)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment in ['prod', 'production']
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment in ['dev', 'development']
    
    def get_cache_dir(self) -> Path:
        """Get cache directory as Path object"""
        return Path(self.cache.dir)
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export configuration as dictionary
        
        Args:
            include_secrets: Whether to include sensitive information
            
        Returns:
            Configuration dictionary
        """
        config_dict = self.dict()
        
        if not include_secrets:
            # Remove sensitive fields
            sensitive_fields = [
                'binance.api_key', 'binance.api_secret',
                'security.jwt_secret_key', 'security.encryption_key',
                'notifications.telegram_bot_token', 'notifications.telegram_chat_id',
                'notifications.email_username', 'notifications.email_password'
            ]
            
            for field_path in sensitive_fields:
                parts = field_path.split('.')
                current = config_dict
                for part in parts[:-1]:
                    if part in current:
                        current = current[part]
                    else:
                        break
                else:
                    if parts[-1] in current:
                        current[parts[-1]] = "***HIDDEN***"
        
        return config_dict
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of issues
        
        Returns:
            List of configuration issues/warnings
        """
        issues = []
        
        # Check for missing API credentials in non-development environments
        if not self.is_development():
            if not self.binance.api_key or not self.binance.api_secret:
                issues.append("Binance API credentials are missing")
        
        # Check cache directory permissions
        cache_dir = self.get_cache_dir()
        if not cache_dir.exists():
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                issues.append(f"Cannot create cache directory: {cache_dir}")
        
        # Validate grid configuration
        if self.grid.upper_price and self.grid.lower_price:
            if self.grid.upper_price <= self.grid.lower_price:
                issues.append("Grid upper price must be greater than lower price")
        
        # Check notification configuration
        if self.notifications.telegram_enabled:
            if not self.notifications.telegram_bot_token or not self.notifications.telegram_chat_id:
                issues.append("Telegram notifications enabled but credentials missing")
        
        return issues

# Global settings instance
_settings: Optional[Settings] = None

def get_settings(config_file: Optional[str] = None, reload: bool = False) -> Settings:
    """
    Get global settings instance
    
    Args:
        config_file: Path to configuration file
        reload: Force reload of settings
        
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None or reload:
        _settings = Settings(config_file=config_file)
        
        # Validate configuration
        issues = _settings.validate_configuration()
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    return _settings

def reload_settings(config_file: Optional[str] = None) -> Settings:
    """Force reload settings"""
    return get_settings(config_file=config_file, reload=True)

# Convenience function for importing
settings = get_settings()
