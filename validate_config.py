#!/usr/bin/env python3
"""
Configuration Validation Tool

This script validates the configuration setup and provides helpful feedback
for developers setting up their environment.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Third-party imports (with optional import handling)
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    import pydantic

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import pydantic_settings

    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    PYDANTIC_SETTINGS_AVAILABLE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Project imports (with optional import handling)
try:
    from config import get_settings

    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


def check_file_exists(file_path: str) -> bool:
    """Check if a file exists"""
    return Path(file_path).exists()


def check_env_file() -> Dict[str, Any]:
    """Check .env file configuration"""
    result = {"status": "success", "messages": [], "warnings": [], "errors": []}

    env_file = Path(".env")
    env_example = Path(".env.example")

    if not env_example.exists():
        result["errors"].append(".env.example file is missing")
        result["status"] = "error"
    else:
        result["messages"].append("✓ .env.example file found")

    if not env_file.exists():
        result["warnings"].append(
            ".env file not found - using environment variables and defaults"
        )
        result["messages"].append("Create .env from .env.example for local development")
    else:
        result["messages"].append("✓ .env file found")

        # Check for common required variables
        if not DOTENV_AVAILABLE:
            result["errors"].append("python-dotenv not installed")
            result["status"] = "error"
        else:
            load_dotenv()

            required_vars = [
                "BINANCE_API_KEY",
                "BINANCE_API_SECRET",
                "JWT_SECRET_KEY",
            ]

            missing_vars = []
            for var in required_vars:
                value = os.getenv(var)
                if not value or value.strip() == "" or "your_" in value.lower():
                    missing_vars.append(var)

            if missing_vars:
                result["warnings"].extend(
                    [
                        f"Environment variable '{var}' is not set or contains placeholder value"
                        for var in missing_vars
                    ]
                )
            else:
                result["messages"].append("✓ Required environment variables are set")

    return result


def check_yaml_config() -> Dict[str, Any]:
    """Check YAML configuration file"""
    result = {"status": "success", "messages": [], "warnings": [], "errors": []}

    config_file = Path("config.yaml")

    if not config_file.exists():
        result["errors"].append("config.yaml file is missing")
        result["status"] = "error"
        return result

    result["messages"].append("✓ config.yaml file found")

    if not YAML_AVAILABLE:
        result["errors"].append("PyYAML not installed")
        result["status"] = "error"
        return result

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            result["errors"].append("config.yaml is empty or invalid")
            result["status"] = "error"
            return result

        # Check for required sections
        required_sections = ["default"]
        optional_sections = ["dev", "stage", "prod"]

        for section in required_sections:
            if section not in config:
                result["errors"].append(
                    f"Missing required section '{section}' in config.yaml"
                )
                result["status"] = "error"
            else:
                result["messages"].append(f"✓ Found section '{section}'")

        for section in optional_sections:
            if section in config:
                result["messages"].append(f"✓ Found section '{section}'")

        # Check for environment variable interpolation syntax
        yaml_content = config_file.read_text()
        if "${" in yaml_content and "}" in yaml_content:
            result["messages"].append("✓ Environment variable interpolation detected")

    except yaml.YAMLError as e:
        result["errors"].append(f"Invalid YAML syntax: {e}")
        result["status"] = "error"
    except Exception as e:
        result["errors"].append(f"Error reading config.yaml: {e}")
        result["status"] = "error"

    return result


def check_dependencies() -> Dict[str, Any]:
    """Check required Python dependencies"""
    result = {"status": "success", "messages": [], "warnings": [], "errors": []}

    # Check availability using the flags set at import time
    dependencies_status = {
        "pydantic": PYDANTIC_AVAILABLE,
        "pydantic_settings": PYDANTIC_SETTINGS_AVAILABLE,
        "yaml": YAML_AVAILABLE,
        "dotenv": DOTENV_AVAILABLE,
    }

    missing_packages = []

    for package, available in dependencies_status.items():
        if available:
            result["messages"].append(f"✓ {package} is installed")
        else:
            missing_packages.append(package)

    if missing_packages:
        result["errors"].extend(
            [f"Missing required package: {pkg}" for pkg in missing_packages]
        )
        result["status"] = "error"
        # Convert package names for pip install (pydantic_settings -> pydantic-settings)
        pip_packages = [
            pkg.replace("_", "-") if pkg == "pydantic_settings" else pkg
            for pkg in missing_packages
        ]
        pip_packages = ["PyYAML" if pkg == "yaml" else pkg for pkg in pip_packages]
        pip_packages = [
            "python-dotenv" if pkg == "dotenv" else pkg for pkg in pip_packages
        ]
        result["messages"].append(
            "Install missing packages with: pip install " + " ".join(pip_packages)
        )

    return result


def check_configuration_loading() -> Dict[str, Any]:
    """Test configuration loading"""
    result = {"status": "success", "messages": [], "warnings": [], "errors": []}

    if not CONFIG_AVAILABLE:
        result["errors"].append("Failed to import config module")
        result["status"] = "error"
        return result

    try:
        # Test loading configuration
        settings = get_settings()
        result["messages"].append("✓ Configuration loaded successfully")

        # Test basic attributes
        if hasattr(settings, "app_name"):
            result["messages"].append(f"✓ App name: {settings.app_name}")

        if hasattr(settings, "environment"):
            result["messages"].append(f"✓ Environment: {settings.environment}")

        # Test validation
        if hasattr(settings, "validate_configuration"):
            issues = settings.validate_configuration()
            if issues:
                result["warnings"].extend(
                    [f"Configuration issue: {issue}" for issue in issues]
                )
            else:
                result["messages"].append("✓ Configuration validation passed")

        # Test export
        if hasattr(settings, "export_config"):
            config_export = settings.export_config(include_secrets=False)
            if config_export:
                result["messages"].append("✓ Configuration export working")

    except Exception as e:
        result["errors"].append(f"Failed to load configuration: {e}")
        result["status"] = "error"

    return result


def print_results(title: str, results: Dict[str, Any]):
    """Print formatted results"""
    print(f"\n{title}")
    print("=" * len(title))

    # Print messages
    for msg in results["messages"]:
        print(f"  {msg}")

    # Print warnings
    for warning in results["warnings"]:
        print(f"  ⚠️  {warning}")

    # Print errors
    for error in results["errors"]:
        print(f"  ❌ {error}")

    # Print status
    status_emoji = {"success": "✅", "warning": "⚠️", "error": "❌"}
    print(
        f"\nStatus: {status_emoji.get(results['status'], '❓')} {results['status'].upper()}"
    )


def main():
    """Run configuration validation"""
    print("CONFIGURATION VALIDATION TOOL")
    print("=" * 50)

    # Check dependencies first
    dep_results = check_dependencies()
    print_results("DEPENDENCY CHECK", dep_results)

    if dep_results["status"] == "error":
        print("\n❌ Cannot proceed - please install missing dependencies")
        return 1

    # Check .env file
    env_results = check_env_file()
    print_results("ENVIRONMENT FILE CHECK", env_results)

    # Check YAML config
    yaml_results = check_yaml_config()
    print_results("YAML CONFIGURATION CHECK", yaml_results)

    # Test configuration loading
    if yaml_results["status"] != "error":
        config_results = check_configuration_loading()
        print_results("CONFIGURATION LOADING TEST", config_results)
    else:
        config_results = {"status": "error"}

    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)

    all_results = [dep_results, env_results, yaml_results, config_results]

    if all(r["status"] == "success" for r in all_results):
        print("✅ All checks passed! Configuration system is ready.")
        return 0
    elif any(r["status"] == "error" for r in all_results):
        print("❌ Critical errors found. Please fix the issues above.")
        return 1
    else:
        print(
            "⚠️  Warnings found. Configuration will work but consider addressing the warnings."
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
