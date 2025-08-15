#!/usr/bin/env python3
"""
Dependency Compatibility Checker

This script checks for deprecated dependencies and suggests modern alternatives.
Run this script before upgrading dependencies to identify potential issues.
"""

import subprocess
import sys
from typing import Dict, List, Tuple

# Known deprecated packages and their replacements
DEPRECATED_PACKAGES = {
    "keras-tuner": {
        "replacement": "optuna",
        "reason": "Better performance, more flexible optimization algorithms",
        "migration_guide": "Replace keras_tuner.RandomSearch with optuna.create_study()",
    },
    "pytz": {
        "replacement": "pendulum or zoneinfo",
        "reason": "Built-in zoneinfo (Python 3.9+) or pendulum for better timezone handling",
        "migration_guide": "Replace pytz.timezone() with zoneinfo.ZoneInfo() or pendulum",
    },
    "python-dateutil": {
        "replacement": "pendulum",
        "reason": "Better API, immutable objects, better timezone support",
        "migration_guide": "Replace dateutil.parser.parse() with pendulum.parse()",
    },
    "marshmallow": {
        "replacement": "pydantic",
        "reason": "Better performance, type safety, runtime validation",
        "migration_guide": "Replace marshmallow.Schema with pydantic.BaseModel",
    },
    "schedule": {
        "replacement": "APScheduler",
        "reason": "More robust, supports persistence, better error handling",
        "migration_guide": "Replace schedule.every().minutes.do() with APScheduler jobs",
    },
}

# Packages with security vulnerabilities in older versions
SECURITY_CRITICAL = {
    "cryptography": "45.0.0",
    "requests": "2.32.0",
    "pyyaml": "6.0.2",
    "tensorflow": "2.18.0",
    "numpy": "2.0.0",  # Address security issues in older versions
}

# Packages with breaking changes in major versions
BREAKING_CHANGES = {
    "numpy": {
        "version": "2.0.0",
        "changes": [
            "np.bool, np.int, np.float, etc. removed (use np.bool_, np.int_, etc.)",
            "np.str_ behavior changed",
            "Some deprecated functions removed",
        ],
    },
    "pandas": {
        "version": "2.0.0",
        "changes": [
            "Default dtype for object columns changed",
            "Some deprecated methods removed",
            "PyArrow backend improvements",
        ],
    },
    "tensorflow": {
        "version": "2.16.0",
        "changes": [
            "Keras 3.0 integration",
            "Some legacy APIs deprecated",
            "Performance improvements",
        ],
    },
}


def check_installed_packages() -> List[Tuple[str, str]]:
    """Check currently installed packages and their versions."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        packages = []
        for line in result.stdout.strip().split("\n")[2:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    packages.append((parts[0], parts[1]))
        return packages
    except subprocess.CalledProcessError:
        return []


def check_deprecated_packages(installed_packages: List[Tuple[str, str]]) -> List[str]:
    """Check for deprecated packages in installed packages."""
    deprecated_found = []
    installed_names = [pkg[0].lower() for pkg in installed_packages]

    for pkg_name in DEPRECATED_PACKAGES:
        if pkg_name.lower() in installed_names:
            deprecated_found.append(pkg_name)

    return deprecated_found


def check_security_vulnerabilities(
    installed_packages: List[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
    """Check for packages with known security vulnerabilities."""
    vulnerabilities = []

    for pkg_name, pkg_version in installed_packages:
        if pkg_name.lower() in SECURITY_CRITICAL:
            required_version = SECURITY_CRITICAL[pkg_name.lower()]
            # Simple version comparison (not perfect but works for basic cases)
            if pkg_version < required_version:
                vulnerabilities.append((pkg_name, pkg_version, required_version))

    return vulnerabilities


def print_deprecation_report(deprecated: List[str]):
    """Print report of deprecated packages."""
    if not deprecated:
        print("✅ No deprecated packages found!")
        return

    print("⚠️  DEPRECATED PACKAGES FOUND:")
    print("=" * 60)

    for pkg in deprecated:
        info = DEPRECATED_PACKAGES[pkg]
        print(f"\n📦 {pkg}")
        print(f"   Replacement: {info['replacement']}")
        print(f"   Reason: {info['reason']}")
        print(f"   Migration: {info['migration_guide']}")


def print_security_report(vulnerabilities: List[Tuple[str, str, str]]):
    """Print report of security vulnerabilities."""
    if not vulnerabilities:
        print("✅ No known security vulnerabilities found!")
        return

    print("\n🔒 SECURITY VULNERABILITIES FOUND:")
    print("=" * 60)

    for pkg_name, current_version, required_version in vulnerabilities:
        print(f"\n🚨 {pkg_name}")
        print(f"   Current: {current_version}")
        print(f"   Required: {required_version}+")
        print(f"   Action: pip install --upgrade {pkg_name}>={required_version}")


def print_breaking_changes_report():
    """Print report of packages with breaking changes."""
    print("\n⚡ PACKAGES WITH POTENTIAL BREAKING CHANGES:")
    print("=" * 60)

    for pkg_name, info in BREAKING_CHANGES.items():
        print(f"\n📋 {pkg_name} {info['version']}+")
        print("   Breaking changes:")
        for change in info["changes"]:
            print(f"   • {change}")


def main():
    """Main function to run all checks."""
    print("🔍 DEPENDENCY COMPATIBILITY CHECKER")
    print("=" * 60)

    # Check installed packages
    print("Scanning installed packages...")
    installed_packages = check_installed_packages()

    if not installed_packages:
        print("❌ Could not retrieve installed packages. Make sure pip is available.")
        return

    print(f"Found {len(installed_packages)} installed packages.\n")

    # Check for deprecated packages
    deprecated = check_deprecated_packages(installed_packages)
    print_deprecation_report(deprecated)

    # Check for security vulnerabilities
    vulnerabilities = check_security_vulnerabilities(installed_packages)
    print_security_report(vulnerabilities)

    # Print breaking changes info
    print_breaking_changes_report()

    # Summary and recommendations
    print("\n📋 SUMMARY AND RECOMMENDATIONS:")
    print("=" * 60)

    if deprecated:
        print(f"• {len(deprecated)} deprecated package(s) found - consider migrating")

    if vulnerabilities:
        print(
            f"• {len(vulnerabilities)} security vulnerability(ies) found - UPDATE IMMEDIATELY"
        )

    print("• Review breaking changes before upgrading major versions")
    print("• Test thoroughly in a development environment before production deployment")
    print("• Consider using virtual environments to isolate dependencies")

    print("\n💡 NEXT STEPS:")
    print("1. Update requirements.txt with new versions")
    print("2. Create a virtual environment: python -m venv trading_bot_env")
    print(
        "3. Activate environment: source trading_bot_env/bin/activate (Linux/Mac) or trading_bot_env\\Scripts\\activate (Windows)"
    )
    print("4. Install dependencies: pip install -r requirements.txt")
    print("5. Run tests to ensure compatibility: python -m pytest tests/")


if __name__ == "__main__":
    main()
