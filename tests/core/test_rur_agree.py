"""``sympy_extras.polys.rur`` against its Rust port: the separating form
(the small forms, then the random ones of Python's ``random.Random``),
the squarefree and characteristic polynomials, the denominator and the
numerators of the coordinates, their reduction modulo the polynomial,
and the exact solutions (all of them and the real ones), with the
multiplicities; the errors of systems of positive dimension and with
symbolic coefficients."""

from __future__ import annotations

from typing import Any

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _systems() -> list[tuple[list[Any], list[Any]]]:
    from sympy_extras.core import Symbol, symbols
    x, y, z = symbols('x y z')
    a = Symbol('a')
    return [([x**2 + y**2 - 1, x*y], [x, y]), ([x**2 + y**2 - 1, x - y], [x, y]), ([x**2 + 1, y - x], [x, y]),
            ([(x - 1)**2*(x + 2), y - x**2], [x, y]), ([x**2 + 4*y**2 - 4, x*y - 1], [x, y]),
            ([x**2 - 2, y - x - 1], [x, y]), ([x**2 - 2, y - x**3], [x, y]), ([x - 1, x - 2], [x]),
            ([x*y], [x, y]), ([x**2 - a], [x]), ([x**2, x*y, y**2], [x, y]), ([x**2 - 3], [x]),
            ([x**2 - 2, y**2 - 3], [x, y]), ([x**2 + y**2 - 5, x*y - 2], [x, y]),
            ([x**3 - x, y**2 - 1, z - x*y], [x, y, z]),
            # symmetric systems where the small forms fail
            ([x + y + z, x*y + y*z + z*x, x*y*z - 1], [x, y, z]),
            ([x**2 - 1, y**2 - 1, z**2 - 1], [x, y, z]),
            ([x + y + z + symbols('w'), x*y + y*z + z*symbols('w') + symbols('w')*x,
              x*y*z + y*z*symbols('w') + z*symbols('w')*x + symbols('w')*x*y, x*y*z*symbols('w') - 1],
             [x, y, z, symbols('w')])]


@pytest.mark.parametrize("index", range(18))
def test_rur(rust: Any, index: int) -> None:
    from sympy_extras.core import Symbol, Tuple
    from sympy_extras.polys.rur import rur
    rc = load("sympy_extras_rs.core")
    polys, gens = _systems()[index]
    t = Symbol('t')

    def expected() -> Any:
        R = rur(polys, *gens, t=t)
        return [str(R.form), str(R.polynomial.to_expr()), str(R.characteristic.to_expr()),
                str(R.denominator.to_expr()), [str(g.to_expr()) for g in R.numerators],
                [str(g.to_expr()) for g in R.polynomials()],
                [[str(p[s]) for s in R.symbols] for p in R.solutions()],
                [[str(p[s]) for s in R.symbols] for p in R.solutions(real=True)]]

    def got() -> Any:
        r = rc.logic_call('rur', [Tuple(*polys), Tuple(*gens), t])
        items = list(r.args)
        return [str(v) for v in items[:4]] + [[str(v) for v in items[4].args], [str(v) for v in items[5].args],
                                               [[str(v) for v in p.args] for p in items[6].args],
                                               [[str(v) for v in p.args] for p in items[7].args]]
    assert outcome(got) == outcome(expected), polys
