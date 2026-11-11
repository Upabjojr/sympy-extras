"""``gcd_terms``, ``factor_terms``, ``as_content_primitive`` and
``decompose_power`` of the Rust core against
``sympy_extras.core.exprtools``, with their options, on random
expressions: polynomial ones, quotients by sums, and noncommutative
combinations (results by structure, classes and reprs)."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from ..polys.test_polyutils_agree import random_expression
from .test_arithmetic_agree import shape

Options = dict[str, bool]


def checks(rc: Any) -> dict[str, tuple[Callable[[Any, Options], Any], Callable[[Any, Options], Any], list[Options]]]:
    from sympy_extras.core import exprtools as et
    return {
        "gcd_terms": (lambda e, o: et.gcd_terms(e, **o), lambda e, o: rc.gcd_terms(e, **o),
                      [{}, {"clear": False}, {"fraction": False}, {"isprimitive": True}]),
        "factor_terms": (lambda e, o: et.factor_terms(e, **o), lambda e, o: rc.factor_terms(e, **o),
                         [{}, {"clear": True}, {"fraction": True}, {"sign": False}]),
        "as_content_primitive": (lambda e, o: e.as_content_primitive(**o),
                                 lambda e, o: rc.as_content_primitive(e, **o), [{}, {"clear": False}]),
        "decompose_power": (lambda e, o: et.decompose_power(e), lambda e, o: rc.decompose_power(e), [{}]),
    }


def agree(got: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        return all(agree(g, x) for g, x in zip(got, expected))
    if isinstance(expected, int):
        return bool(got == expected)
    return bool(shape(got) == shape(expected) and repr(got) == repr(expected))


@pytest.mark.parametrize("seed", range(20))
def test_exprtools_agree(rust: Any, seed: int) -> None:
    from sympy_extras.core.symbol import Symbol
    rc = load("sympy_extras_rs.core")
    A, B = Symbol('A', commutative=False), Symbol('B', commutative=False)
    rng = random.Random(stable_seed("exprtools", seed))
    compared = 0
    for _ in range(15):
        e = random_expression(rng, 3)
        if rng.random() < 0.3:
            e = e / (random_expression(rng, 2) + 1)
        if rng.random() < 0.15:
            e = e * A + B * e
        for name, (python, rust_function, options) in checks(rc).items():
            for o in options:
                try:
                    expected = python(e, o)
                except (ValueError, TypeError, ZeroDivisionError):
                    continue
                try:
                    got = rust_function(e, o)
                except NotImplementedError:
                    continue
                assert agree(got, expected), (name, o, e, got, expected)
                compared += 1
    assert compared >= 100
