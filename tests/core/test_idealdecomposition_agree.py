"""``sympy_extras.polys.idealdecomposition`` against its Rust port: the
radical in every dimension, the equidimensional parts, the minimal primes
(split by separating linear forms), the primary decomposition of Gianni,
Trager and Zacharias with Shimoyama and Yokoyama's separators, the
associated primes, the primary and prime tests, over ``QQ``; the refusal
of the other fields."""

from __future__ import annotations

from typing import Any, Callable

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _ideals() -> list[tuple[list[Any], list[Any], int]]:
    from sympy_extras.core import symbols
    x, y, z = symbols('x y z')
    cone = x**2 + y**2 - z**2
    return [
        ([x**2*y, x*y**3], [x, y], 0),
        ([x**2, x*y], [x, y], 0),
        ([cone**2, (x - y)**3*cone], [x, y, z], 0),
        ([x**2 - y**2], [x, y], 0),
        ([x**2*z - y**2*z, x**3*y - x*y**3, z**2*(x - y)], [x, y, z], 0),
        ([x*z, x*y], [x, y, z], 0),
        ([x*z - y**2, x**2 - y*z], [x, y, z], 0),
        ([x**2 - 2*y**2], [x, y], 0),
        ([x**4 - y**4, z**2 - x**2], [x, y, z], 0),
        ([x**2 - z, y**2 - z], [x, y, z], 0),
        ([x**2 - y**3], [x, y], 0),
        ([x**2, x*y], [x, y, z], 0),
        ([x*y, x*z, y*z], [x, y, z], 0),
        ([x*z, y**2*z, (x - 1)**2*y], [x, y, z], 0),
        ([(x**2 - 2)**2, (x**2 - 2)*y], [x, y], 0),
        ([x**2, y**3], [x, y], 0),
        ([x**2, y], [x, y, z], 0),
        ([x*y], [x, y], 0),
        ([x**2 - 1, y**2 - 4], [x, y], 0),
        ([x*y], [x, y], 5),
        ([x, x + 1], [x, y], 0),
    ]


OPERATIONS = ['radical', 'equidimensional_parts', 'minimal_primes', 'primary_decomposition', 'associated_primes',
              'is_primary', 'is_prime', 'is_radical']


@pytest.mark.parametrize("index", range(21))
def test_decompositions(rust: Any, index: int) -> None:
    from sympy_extras.core import Integer, Symbol, Tuple
    from sympy_extras.polys.domains import GF, QQ
    from sympy_extras.polys.ideals import Ideal
    rc = load("sympy_extras_rs.core")
    gens, variables, p = _ideals()[index]
    domain = QQ if p == 0 else GF(p)

    def basis(J: Any) -> list[str]:
        return [str(J.to_expr(g)) for g in J.groebner_basis()]

    def python(op: str) -> Callable[[], Any]:
        def call() -> Any:
            result = getattr(Ideal(gens, *variables, domain=domain), op)()
            if isinstance(result, bool):
                return str(result)
            if op == 'radical':
                return basis(result)
            if op == 'primary_decomposition':
                return [[basis(q), basis(p_)] for q, p_ in result]
            return [basis(J) for J in result]
        return call

    def rust(op: str) -> Callable[[], Any]:
        def call() -> Any:
            r = rc.logic_call('ideal', [Symbol(op), Tuple(*gens), Tuple(*variables), Symbol('grevlex'), Integer(p)])
            if not isinstance(r, Tuple):
                return str(r)
            if op == 'radical':
                return [str(v) for v in r.args]
            if op == 'primary_decomposition':
                return [[[str(v) for v in q.args], [str(v) for v in p_.args]] for q, p_ in r.args]
            return [[str(v) for v in J.args] for J in r.args]
        return call
    for op in OPERATIONS:
        assert outcome(rust(op)) == outcome(python(op)), (op, gens, p)
