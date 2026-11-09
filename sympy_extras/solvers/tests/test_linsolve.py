"""solve_linear_unique against SymPy's linsolve."""
from __future__ import annotations

import random


def test_solve_linear_unique_matches_sympy_linsolve() -> None:
    import sympy
    from sympy_extras.core.symbol import Symbol as NativeSymbol
    from sympy_extras.interop import expr_from_sympy, sympy_expr, to_sympy
    from sympy_extras.solvers.linsolve import solve_linear_unique
    rng = random.Random(7)
    n, m = sympy.symbols('n m')
    unknowns = list(sympy.symbols('c0:4', cls=sympy.Dummy))
    native_unknowns = []
    for u in unknowns:
        nu = expr_from_sympy(u)
        assert isinstance(nu, NativeSymbol)
        native_unknowns.append(nu)
    pool = [sympy.Integer(0), sympy.Integer(1), sympy.Integer(-2), sympy.Rational(3, 2), n, n + 1, m - n, 1/(n + 2)]
    unique = 0
    for trial in range(80):
        size = rng.choice([1, 2, 3])
        rows = rng.choice([size - 1, size, size, size + 1]) if size > 1 else size
        eqs = [sympy.Add(*[rng.choice(pool)*u for u in unknowns[:size]]) + rng.choice(pool) for _ in range(rows)]
        got = solve_linear_unique([expr_from_sympy(e) for e in eqs], native_unknowns[:size])
        want = sympy.linsolve(eqs, unknowns[:size])
        point = None
        if isinstance(want, sympy.FiniteSet) and want.args:
            first = want.args[0]
            assert isinstance(first, sympy.Tuple)
            values = list(first.args)
            if not any(v.has(*unknowns[:size]) for v in values):
                point = values
        if point is None:
            assert got is None, (eqs, got, want)
            continue
        unique += 1
        assert got is not None, (eqs, want)
        for g, w in zip(got, point):
            assert sympy.cancel(sympy_expr(to_sympy(g)) - w) == 0, (eqs, got, point)
    assert unique > 20


def test_solve_linear_unique_declines() -> None:
    from sympy_extras.core import Symbol, sqrt, pi, Integer
    from sympy_extras.functions import log
    from sympy_extras.solvers.linsolve import solve_linear_unique
    a, b, n = Symbol('a'), Symbol('b'), Symbol('n')
    assert solve_linear_unique([a*b - 1, a - b], [a, b]) is None        # not linear
    assert solve_linear_unique([a + b - 1, a + b - 2], [a, b]) is None  # inconsistent
    assert solve_linear_unique([sqrt(Integer(2))*a - 1], [a]) is None            # outside QQ(n)
    assert solve_linear_unique([a - log(n)], [a]) is None
    assert solve_linear_unique([a - pi], [a]) is None
    assert solve_linear_unique([], [a]) is None
