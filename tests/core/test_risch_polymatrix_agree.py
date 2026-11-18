"""``sympy_extras.integrals.risch.polymatrix`` against its Rust port:
the arithmetic, the joins and the transposes of :class:`FieldMatrix`, and
its reduced row echelon form, nullspace and rank by Gauss–Jordan
elimination over the field of the tower (random matrices of low rank,
with entries rational in `x`, `a` and `exp(x)`)."""

from __future__ import annotations

import random
from typing import Any, Callable

import pytest

from ..conftest import load
from .test_integers_agree import outcome


def _entry(rng: random.Random) -> Any:
    from sympy_extras.core import Integer, Symbol
    from sympy_extras.functions import exp
    x, a = Symbol('x'), Symbol('a')
    atoms = [x, a, exp(x), Integer(1), Integer(2), Integer(-1), x/2, Integer(0), Integer(0)]
    e = rng.choice(atoms)
    if rng.random() < 0.3:
        e = e/(rng.choice(atoms[:3]) + rng.randint(1, 2))
    if rng.random() < 0.3:
        e = e*rng.choice(atoms)
    return e


def _matrix(rng: random.Random, m: int, n: int) -> list[list[Any]]:
    """A random ``m x n`` matrix, of low rank half of the time."""
    rows = [[_entry(rng) for _ in range(n)] for _ in range(m)]
    if m > 1 and rng.random() < 0.5:
        c = _entry(rng)
        rows[-1] = [rows[0][j]*c + rows[1 % m][j] for j in range(n)]
    return rows


def _flat(rows: list[list[Any]], n: int) -> list[Any]:
    from sympy_extras.core import Integer
    return [Integer(len(rows)), Integer(n)] + [e for r in rows for e in r]


def _python(rows: list[list[Any]], n: int) -> Any:
    from sympy_extras.integrals.risch.polymatrix import FieldMatrix
    from sympy_extras.integrals.risch.tower import RationalFunction as RF
    return FieldMatrix.from_rows([[RF.from_expr(e) for e in r] for r in rows], n)


def _shown(M: Any) -> tuple[int, int, tuple[str, ...]]:
    return (M.rows, M.cols, tuple(str(e) for e in M.flat()))


def _rust_shown(t: Any) -> tuple[int, int, tuple[str, ...]]:
    return (int(t.args[0]), int(t.args[1]), tuple(str(e) for e in t.args[2].args))


def _rust(rc: Any, op: str, *matrices: tuple[list[list[Any]], int]) -> Callable[[], Any]:
    from sympy_extras.core import Symbol

    def got() -> Any:
        args: list[Any] = [Symbol(op)]
        for rows, n in matrices:
            args += _flat(rows, n)
        return rc.logic_call('field_matrix', args)
    return got


@pytest.mark.parametrize("seed", range(16))
def test_field_matrix(rust: Any, seed: int) -> None:
    rc = load("sympy_extras_rs.core")
    rng = random.Random(3000 + seed)
    m, n = rng.randint(1, 4), rng.randint(1, 4)
    A, B, C = _matrix(rng, m, n), _matrix(rng, m, n), _matrix(rng, n, rng.randint(1, 3))
    P = _python(A, n)

    def rust_rref() -> Any:
        r = _rust(rc, 'rref', (A, n))()
        return (_rust_shown(r.args[0]), tuple(int(p) for p in r.args[1].args))
    assert outcome(rust_rref) == outcome(lambda: (lambda r: (_shown(r[0]), r[1]))(P.rref())), A
    assert outcome(lambda: tuple(_rust_shown(v) for v in _rust(rc, 'nullspace', (A, n))().args)) == \
        outcome(lambda: tuple(_shown(v) for v in P.nullspace())), A
    assert outcome(lambda: int(_rust(rc, 'rank', (A, n))().args[0])) == outcome(P.rank), A
    for op, other, fn in [('add', (B, n), lambda: P + _python(B, n)), ('sub', (B, n), lambda: P - _python(B, n)),
                          ('mul', (C, len(C[0])), lambda: P*_python(C, len(C[0]))),
                          ('row_join', (B, n), lambda: P.row_join(_python(B, n))),
                          ('col_join', (B, n), lambda: P.col_join(_python(B, n)))]:
        got = lambda: _rust_shown(_rust(rc, op, (A, n), other)().args[0])  # noqa: E731
        assert outcome(got) == outcome(lambda: _shown(fn())), (op, A)
    assert outcome(lambda: _rust_shown(_rust(rc, 'transpose', (A, n))().args[0])) == \
        outcome(lambda: _shown(P.transpose())), A
