"""The cylindrical algebraic decomposition of the Rust port (the lifting
phase over the projection sets) against
``sympy_extras.polys.cad.lifting.cylindrical_algebraic_decomposition``:
random polynomials in one, two and three variables, both projection
operators and the default (McCallum, Hong when not well-oriented),
compared through the reprs of the cells, their signs, the projection
sets and the sizes of the levels, and the class of the error."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load, stable_seed
from .test_cad_projection_agree import random_polynomial


def outcome(f: Callable[[], Any]) -> Any:
    try:
        return f()
    except NotImplementedError:
        return 'SKIP'
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def python_report(polys: list[Any], gens: list[Any], method: Any) -> dict[str, Any]:
    from sympy_extras.polys.cad.lifting import cylindrical_algebraic_decomposition
    cad = cylindrical_algebraic_decomposition(polys, gens, method)
    return {"repr": repr(cad), "method": cad.method,
            "projection": [[str(p) for p in level] for level in cad.projection],
            "cells": ["%r %s" % (cell, ",".join(str(s) for s in cell.signs)) for cell in cad.cells],
            "sizes": [len(level) for level in cad.levels()]}


def check(rc: Any, polys: list[Any], gens: list[Any], method: Any) -> bool:
    got = outcome(lambda: dict(rc.cad_report(polys, gens, method)))
    if got == 'SKIP':
        return False
    expected = outcome(lambda: python_report(polys, gens, method))
    assert got == expected, (polys, gens, method)
    return True


@pytest.mark.parametrize("seed", range(30))
def test_cad(rust: Any, seed: int) -> None:
    from sympy_extras.core import Symbol
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("cad lifting", seed))
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    gens = rng.choice([[x], [x, y], [x, y], [x, y, z]])
    polys = [random_polynomial(rng, gens) for _ in range(rng.randint(1, 2 if len(gens) == 3 else 3))]
    compared = 0
    for method in (None, "mccallum", "hong") if len(gens) < 3 else (None,):
        compared += check(rc, polys, gens, method)
    assert compared >= 1


def test_examples(rust: Any) -> None:
    from sympy_extras.core import Symbol
    rc = load("sympy_extras_rs.core")
    x, y, z = Symbol('x'), Symbol('y'), Symbol('z')
    check(rc, [x**2 + y**2 - 1], [x, y], None)
    check(rc, [x*y - 1, x - y], [x, y], "hong")
    check(rc, [x*z - y, x], [x, y, z], "mccallum")
    check(rc, [x*z - y, x], [x, y, z], None)
    check(rc, [x**2 - 2], [x], None)
    check(rc, [x], [], None)
    check(rc, [x], [x], "nonsense")
