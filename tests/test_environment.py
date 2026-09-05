"""
Initialization test: verifies that the test runner is functional and the
Python environment meets the project requirements.

This test exists only to confirm that the development environment is
correctly configured. It does not test any platform functionality.
It should be easy to remove or replace as the test suite grows.
"""

import sys


def test_python_version_meets_requirement():
    """
    The project requires Python 3.13 or later.

    This test documents and enforces the version constraint defined in
    pyproject.toml so that misconfigured environments fail explicitly.
    """
    assert sys.version_info >= (3, 13), (
        f"Python 3.13 or later is required. Current version: "
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


def test_runner_is_functional():
    """
    Trivial assertion confirming the test runner itself is operational.

    This test has no platform significance. It exists as a sentinel
    for the initialization state of the repository.
    """
    assert True
