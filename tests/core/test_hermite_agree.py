"""``sympy_extras.polys.hermite`` against its Rust port: Hermite's
quadratic form on the quotient algebra of zero-dimensional ideals over
``QQ`` (the products of standard monomials, the traces), the counts of
real and complex solutions, Tarski queries, sign counts, Ben-Or–Kozen–Reif
sign determination and the decision of conjunctions of relations."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _systems() -> list[tuple[list[Any], list[Any]]]:
    from sympy_extras.core import symbols
    x, y, z = symbols('x y z')
    return [([x**2 + y**2 - 1, x - y], [x, y]), ([x**2 + y**2 - 1, x + y - 2], [x, y]),
            ([x**2 + y**2 - 2, x + y - 2], [x, y]), ([x**2 + 1], [x]), ([x**2 + 1, y**2 + 1], [x, y]),
            ([(x - 1)**3*(x + 2), y**2], [x, y]), ([x**3 - x, y - x**2], [x, y]), ([x**2 - 1, y**2 - 1], [x, y]),
            ([x**2 - 1, y**2 - 4, z**2 - 9], [x, y, z]), ([x**2 - 2, y - x], [x, y]), ([x, x + 1], [x, y]),
            ([x**3 - 3*x + 1, y**2 - x - 2], [x, y]), ([x**2*y - 1, y**2 - 2*x], [x, y])]


@pytest.mark.parametrize("index", range(13))
def test_hermite(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, Symbol, Tuple
    from sympy_extras.polys import hermite as H
    rc = load("sympy_extras_rs.core")
    polys, gens = _systems()[index]

    def rust(op: str, *extra: Any) -> Callable[[], Any]:
        return lambda: rc.logic_call('hermite_form', [Symbol(op), Tuple(*polys), Tuple(*gens), *extra])

    def ints(t: Any) -> Any:
        return tuple(int(v) for v in t.args)
    assert outcome(lambda: int(rust('count_real')())) == outcome(lambda: H.count_real_solutions(polys, *gens))
    assert outcome(lambda: int(rust('count_complex')())) == outcome(lambda: H.count_complex_solutions(polys, *gens))
    x = gens[0]
    tests = [2*x - 1, x**2*gens[-1], x + gens[-1], gens[-1] - 1]
    for g in tests:
        assert outcome(lambda: int(rust('tarski', g)())) == outcome(lambda: H.tarski_query(polys, g, *gens)), g
        assert outcome(lambda: ints(rust('sign_counts', g)())) == \
            outcome(lambda: tuple(H.real_sign_counts(polys, g, *gens))), g
    for chosen in (tests[:1], tests[:2], tests):
        assert outcome(lambda: {ints(s): int(c) for s, c in (p.args for p in rust('sign_determination',
                                                                                   Tuple(*chosen))().args)}) == \
            outcome(lambda: H.sign_determination(polys, chosen, *gens)), chosen

    def python_matrix(g: Any = 1) -> Any:
        M = H.hermite_matrix(polys, g, *gens)
        return [[str(M[i, j]) for j in range(M.cols)] for i in range(M.rows)]
    assert outcome(lambda: [[str(v) for v in r.args] for r in rust('matrix')().args]) == outcome(python_matrix)
    assert outcome(lambda: [[str(v) for v in r.args] for r in rust('matrix', tests[0])().args]) == \
        outcome(lambda: python_matrix(tests[0]))


def test_decide(rust: Any) -> None:
    from sympy_extras.core import Eq, Ne, Symbol, Tuple, symbols
    from sympy_extras.polys.hermite import decide_zero_dimensional
    rc = load("sympy_extras_rs.core")
    x, y = symbols('x y')
    circle = Eq(x**2 + y**2, 1) & Eq(x - y, 0)
    formulas = [(circle & (x > 0), [x, y]), (circle & (x > 1), [x, y]), (circle & (x >= 1), [x, y]),
                (circle & (x*y < 1), [x, y]), (circle & (x*y <= 1), [x, y]), (circle & Ne(x**2, 1), [x, y]),
                (circle & (x > 0) & (y < 0), [x, y]), (Eq(x, 1) & Eq(x, 2) & Eq(y, 0), [x, y]), (Eq(x**2, 2), [x]),
                (Eq(x**2 + y**2, 2) & (x > 0), [x, y]), (Eq(x**2, 2) & Eq(y**3, x), [x, y]),
                (Eq(x**2, 2) & (x**3 < 1) & Ne(y, 0) & Eq(y**2, 3), [x, y])]
    for formula, gens in formulas:
        def got() -> Any:
            r = rc.logic_call('hermite_form', [Symbol('decide'), Tuple(), Tuple(*gens), formula])
            return None if isinstance(r, Tuple) else bool(r)
        assert outcome(got) == outcome(lambda: decide_zero_dimensional(formula, gens)), formula
