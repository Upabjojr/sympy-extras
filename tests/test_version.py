from __future__ import annotations

import sympy_extras

from .conftest import load


def test_version(impl: str) -> None:
    assert load(impl).__version__ == sympy_extras.__version__
