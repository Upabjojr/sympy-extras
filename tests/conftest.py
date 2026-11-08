"""Fixtures for the tests that run against both implementations.

The ``impl`` fixture yields the name of the pure Python package
``sympy_extras`` and of the Rust extension module ``sympy_extras_rs``,
whose submodules have the same names (``impl + '.polys.densearith'``); the
Rust run is skipped when the extension has not been built (``maturin
develop`` in ``crates/sympy-extras-py``). The ``rust`` fixture gives the
extension module itself, for the tests that compare the two directly.
"""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest

PYTHON: str = "sympy_extras"
RUST: str = "sympy_extras_rs"
IMPLEMENTATIONS: list[str] = [PYTHON, RUST]


def load(name: str) -> ModuleType:
    """The module ``name``; the test is skipped when it is not built."""
    try:
        return importlib.import_module(name)
    except ImportError:
        pytest.skip(f"{name} is not built")


@pytest.fixture(params=IMPLEMENTATIONS, ids=["python", "rust"])
def impl(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    load(name)
    return name


@pytest.fixture
def rust() -> ModuleType:
    return load(RUST)
