"""``sympy_extras.matrices.domainmatrix`` against its Rust port: random
matrices over ``ZZ`` and ``QQ`` (square and not, singular and of low
rank, empty), compared on the determinant, rank, ``rref_den``,
``rref``, null space, ``inv_den``, ``inv``, ``solve_den``, the
characteristic polynomial, the adjugate, products, sums, powers and the
printed form (or the error)."""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pytest

from ..conftest import load, stable_seed


def to_core(v: Any) -> Any:
    """The nested results of the Python class as nodes of the core."""
    from sympy_extras.core import Integer, Rational
    from sympy_extras.core.containers import Tuple
    from sympy_extras.matrices.domainmatrix import DomainMatrix
    if v is None:
        return Tuple()
    if isinstance(v, DomainMatrix):
        return Tuple(*[Tuple(*[to_core(a) for a in r]) for r in v.rows])
    if isinstance(v, (tuple, list)):
        return Tuple(*[to_core(a) for a in v])
    if isinstance(v, Fraction):
        return Rational(v.numerator, v.denominator)
    if isinstance(v, int):
        return Integer(v)
    raise TypeError(v)


def result(f: Any) -> str:
    try:
        return str(f())
    except Exception as exc:
        return 'ERROR ' + type(exc).__name__


def random_matrix(rng: random.Random, n: int, m: int, field: bool) -> list[list[Any]]:
    rows = [[(Fraction(rng.randint(-4, 4), rng.randint(1, 3)) if field else rng.randint(-5, 5)) for _ in range(m)]
            for _ in range(n)]
    if n >= 2 and rng.random() < 0.3:
        k = rng.randint(-2, 2)
        rows[-1] = [a + k*b for a, b in zip(rows[0], rows[1])]  # a dependent row
    return rows


@pytest.mark.parametrize("seed", range(60))
def test_domainmatrix(rust: Any, seed: int) -> None:
    from sympy_extras.core import Integer, Rational, Symbol
    from sympy_extras.core.containers import Tuple
    from sympy_extras.matrices.domainmatrix import DomainMatrix
    from sympy_extras.polys.domains import QQ, ZZ
    rc = load("sympy_extras_rs.core")
    rng = random.Random(stable_seed("domainmatrix", seed))
    for _ in range(4):
        field = rng.random() < 0.5
        K, name = (QQ, 'QQ') if field else (ZZ, 'ZZ')
        n = rng.randint(1, 4)
        m = n if rng.random() < 0.6 else rng.randint(1, 4)
        rows = random_matrix(rng, n, m, field)
        A = DomainMatrix(rows, K)
        core_rows = to_core(A)

        def rust(op: str, *extra: Any) -> str:
            return result(lambda: rc.logic_call('domain_matrix', [Symbol(name), Symbol(op), core_rows, *extra]))

        def python(f: Any) -> str:
            return result(lambda: to_core(f()))

        assert rust('repr') == result(lambda: Symbol(repr(A))), A
        for op in ('det', 'rank', 'rref_den', 'rref', 'nullspace', 'inv_den', 'inv', 'charpoly', 'adjugate', 'transpose'):
            assert rust(op) == python(lambda: getattr(A, op)()), (op, A)
        B = DomainMatrix(random_matrix(rng, n, rng.randint(1, 2), field), K)
        assert rust('solve_den', to_core(B)) == python(lambda: A.solve_den(B)), (A, B)
        C = DomainMatrix(random_matrix(rng, m, rng.randint(1, 3), field), K)
        assert rust('mul', to_core(C)) == python(lambda: A * C), (A, C)
        D = DomainMatrix(random_matrix(rng, n, m, field), K)
        assert rust('add', to_core(D)) == python(lambda: A + D), (A, D)
        k = rng.randint(-1, 3)
        assert rust('pow', Integer(k)) == python(lambda: A**k), (A, k)
