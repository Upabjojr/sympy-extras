"""The pretty printer of the Rust core (``printing/pretty``) against
``sympy_extras.printing.pretty``: the expressions of the LaTeX agree tests
(numbers, names, sums, products, powers, functions, relations, logic,
derivatives, piecewise, sums, integrals, sets, limits, order terms) in
Unicode and in ASCII, with the settings of the printer; the nodes not
ported yet are a skip on the Rust side."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from ..conftest import load
from .test_latex_agree import _cases as _latex_cases, _outcome, _random
from .test_latex_objects_agree import _cases as _object_cases

SETTINGS: list[dict[str, Any]] = [
    {'use_unicode': True}, {'use_unicode': False}, {'use_unicode': True, 'root_notation': False},
    {'use_unicode': True, 'use_unicode_sqrt_char': False}, {'use_unicode': True, 'imaginary_unit': 'j'},
    {'use_unicode': True, 'order': 'none'}, {'use_unicode': False, 'order': 'none'},
]


def _cases() -> list[Any]:
    return _latex_cases() + _object_cases()


def _python(e: Any, settings: dict[str, Any]) -> str:
    pretty = importlib.import_module('sympy_extras.printing.pretty.pretty').pretty
    return pretty(e, wrap_line=False, **settings)


@pytest.mark.parametrize("index", range(len(_cases())))
def test_pretty(rust: Any, index: int) -> None:
    rc = load("sympy_extras_rs.core")
    e = _cases()[index]
    for settings in SETTINGS:
        got = _outcome(lambda: rc.pretty(e, dict(settings, wrap_line=False)))
        if got[0] == 'SKIP':
            continue
        assert got == _outcome(lambda: _python(e, settings)), (e, settings)


@pytest.mark.parametrize("seed", range(6))
def test_pretty_random(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    different = []
    for e in _random(seed):
        for settings in SETTINGS[:2]:
            got = _outcome(lambda: rc.pretty(e, dict(settings, wrap_line=False)))
            if got[0] != 'SKIP' and got != _outcome(lambda: _python(e, settings)):
                different.append((e, settings, got))
    assert not different, different[:3]
