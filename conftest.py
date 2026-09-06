"""pytest configuration for sympy-extras.

The doctests in the docstrings follow the conventions of SymPy's own doctest
runner, which prints the result of each expression with ``str`` rather than
``repr`` (so that ``QQ(2, 5)`` shows as ``2/5`` and not ``MPQ(2,5)``). The
same display hook is installed here while a doctest item runs. The standard
``doctest`` runner resets ``sys.displayhook`` to ``sys.__displayhook__``
before executing an example, so both are replaced.
"""
from __future__ import annotations

import sys

import pytest
from _pytest.doctest import DoctestItem


def _str_displayhook(value):
    if value is not None:
        print(str(value))
        import builtins
        builtins._ = value


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if isinstance(item, DoctestItem):
        item._sympy_extras_displayhook = (sys.displayhook, sys.__displayhook__)
        sys.displayhook = sys.__displayhook__ = _str_displayhook


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    hooks = getattr(item, "_sympy_extras_displayhook", None)
    if hooks is not None:
        sys.displayhook, sys.__displayhook__ = hooks
