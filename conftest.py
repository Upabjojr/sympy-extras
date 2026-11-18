"""pytest configuration for sympy-extras.

The doctests in the docstrings follow the conventions of SymPy's own doctest
runner, which prints the result of each expression with ``str`` rather than
``repr`` (so that ``QQ(2, 5)`` shows as ``2/5`` and not ``MPQ(2,5)``). The
same display hook is installed here while a doctest item runs. The standard
``doctest`` runner resets ``sys.displayhook`` to ``sys.__displayhook__``
before executing an example, so both are replaced.

The working precision of mpmath (``mpmath.mp.prec``) is global: a test
setting ``mpmath.mp.dps`` changed every later test of the same process (a
doctest printing an ``mpf`` showed 20 digits instead of 15 in CI, which
runs the whole suite in one process). It is restored after each item.
"""
from __future__ import annotations

import sys
from typing import Callable, Optional

import mpmath
import pytest
from _pytest.doctest import DoctestItem

#: the display hooks replaced while a doctest item runs
_DISPLAYHOOKS = pytest.StashKey[tuple[Callable[[object], None], Callable[[object], None]]]()
#: the precision of mpmath before the item ran
_MPMATH_PREC = pytest.StashKey[int]()


def _str_displayhook(value: object) -> None:
    if value is not None:
        print(str(value))
        import builtins
        setattr(builtins, '_', value)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    item.stash[_MPMATH_PREC] = mpmath.mp.prec
    if isinstance(item, DoctestItem):
        item.stash[_DISPLAYHOOKS] = (sys.displayhook, sys.__displayhook__)
        sys.displayhook = sys.__displayhook__ = _str_displayhook


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: Optional[pytest.Item]) -> None:
    prec = item.stash.get(_MPMATH_PREC, None)
    if prec is not None:
        mpmath.mp.prec = prec
    hooks = item.stash.get(_DISPLAYHOOKS, None)
    if hooks is not None:
        sys.displayhook, sys.__displayhook__ = hooks
