#!/usr/bin/env python3
"""
Dependency compatibility test script.
Tests that all major dependencies can be imported and basic functionality works.
"""

import sys
import traceback
from typing import List, Tuple


def test_import(module_name: str, alias: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        if alias:
            exec(f"import {module_name} as {alias}")
        else:
            exec(f"import {module_name}")
        return True, f"✅ {module_name} imported successfully"
    except ImportError as e:
        return False, f"❌ {module_name} failed to import: {str(e)}"
    except Exception as e:
        return False, f"❌ {module_name} unexpected error: {str(e)}"


def test_basic_functionality():
    """Test basic functionality of key packages."""
    results = []

    # Test numpy
    try:
        import numpy as np

        arr = np.array([1, 2, 3])
        assert arr.sum() == 6
        results.append("✅ numpy basic operations work")
    except Exception as e:
        results.append(f"❌ numpy basic operations failed: {str(e)}")

    # Test pandas
    try:
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert len(df) == 3
        results.append("✅ pandas basic operations work")
    except Exception as e:
        results.append(f"❌ pandas basic operations failed: {str(e)}")

    # Test tensorflow
    try:
        import tensorflow as tf

        # Test simple tensor operation
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        z = tf.add(x, y)
        assert z.numpy().tolist() == [5, 7, 9]
        results.append("✅ tensorflow basic operations work")
    except Exception as e:
        results.append(f"❌ tensorflow basic operations failed: {str(e)}")

    # Test scikit-learn
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.datasets import make_regression

        X, y = make_regression(n_samples=10, n_features=1, noise=0.1, random_state=42)
        model = LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)
        assert score > 0.9  # Should have high R² for simple synthetic data
        results.append("✅ scikit-learn basic operations work")
    except Exception as e:
        results.append(f"❌ scikit-learn basic operations failed: {str(e)}")

    # Test modern replacements
    try:
        import pendulum

        now = pendulum.now()
        assert now.year >= 2024
        results.append("✅ pendulum (pytz replacement) works")
    except Exception as e:
        results.append(f"❌ pendulum basic operations failed: {str(e)}")

    try:
        import optuna

        study = optuna.create_study()
        assert study is not None
        results.append("✅ optuna (keras-tuner replacement) works")
    except Exception as e:
        results.append(f"❌ optuna basic operations failed: {str(e)}")

    try:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)
        assert model.name == "test"
        results.append("✅ pydantic (marshmallow replacement) works")
    except Exception as e:
        results.append(f"❌ pydantic basic operations failed: {str(e)}")

    return results


def main():
    """Main test function."""
    print("🧪 DEPENDENCY COMPATIBILITY TEST")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print()

    # Core packages to test
    packages = [
        "numpy",
        "pandas",
        "tensorflow",
        "sklearn",
        "matplotlib",
        "requests",
        "aiohttp",
        "websockets",
        "httpx",
        "pyarrow",
        "sqlalchemy",
        "pendulum",
        "optuna",
        "joblib",
        "scipy",
        "plotly",
        "seaborn",
        "typer",
        "rich",
        "click",
        "streamlit",
        "fastapi",
        "uvicorn",
        "pydantic",
        "yaml",
        "structlog",
        "pytest",
        "cryptography",
        "orjson",
        "psutil",
        "ccxt",
        "apscheduler",
        "black",
        "flake8",
        "mypy",
    ]

    print("📦 IMPORT TESTS")
    print("-" * 40)

    passed = 0
    failed = 0
    failed_packages = []

    for package in packages:
        success, message = test_import(package)
        print(message)
        if success:
            passed += 1
        else:
            failed += 1
            failed_packages.append(package)

    print()
    print("🔧 FUNCTIONALITY TESTS")
    print("-" * 40)

    functionality_results = test_basic_functionality()
    for result in functionality_results:
        print(result)
        if "✅" in result:
            passed += 1
        else:
            failed += 1

    print()
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed_packages:
        print(f"🚨 Failed packages: {', '.join(failed_packages)}")

    if failed == 0:
        print("\n🎉 All tests passed! Dependencies are properly configured.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
