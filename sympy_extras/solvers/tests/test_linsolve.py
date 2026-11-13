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


def test_linear_root_and_numeric_linear_solution_match_sympy_solve() -> None:
    import sympy
    from sympy_extras.core.symbol import Symbol as NativeSymbol
    from sympy_extras.interop import expr_from_sympy, sympy_expr, to_sympy
    from sympy_extras.solvers.linsolve import linear_root, numeric_linear_solution
    rng = random.Random(11)
    i, j, k, n = sympy.symbols('i j k n')
    d = sympy.Dummy('d')
    coefficients = [sympy.Integer(1), sympy.Integer(-1), sympy.Integer(2), sympy.Integer(-3), sympy.Rational(1, 2)]
    rests = [i, 2*i + 1, i - n + sympy.Rational(1, 3), n*i, sympy.sin(n), sympy.Integer(0)]
    roots = 0
    for _ in range(60):
        v = rng.choice([k, d])
        e = rng.choice(coefficients)*v + rng.choice(rests)
        if rng.random() < 0.2:
            e += v**2
        native_v = expr_from_sympy(v)
        assert isinstance(native_v, NativeSymbol)
        got = linear_root(expr_from_sympy(e), native_v)
        if got is None:
            assert e.has(v**2), e
            continue
        roots += 1
        assert sympy.srepr(sympy_expr(to_sympy(got))) == sympy.srepr(sympy.solve(e, v)[0]), e
    assert roots >= 30
    systems = 0
    for _ in range(60):
        unknowns = [i, j, n][:rng.choice([1, 2, 3])]
        eqs = [sympy.Add(*[rng.choice(coefficients + [sympy.Integer(0)])*u for u in unknowns])
               + rng.choice([sympy.Integer(0), sympy.Integer(1), sympy.Integer(-2), sympy.Rational(1, 2)])
               for _ in range(rng.choice([len(unknowns), len(unknowns) + 1]))]
        got_dict = numeric_linear_solution([expr_from_sympy(e) for e in eqs])
        want = sympy.solve(eqs, dict=True)
        free = set().union(*[e.free_symbols for e in eqs])
        if len(want) == 1 and set(want[0]) == free:
            systems += 1
            assert got_dict is not None, eqs
            assert {sympy.srepr(to_sympy(a)): sympy.srepr(to_sympy(b)) for a, b in got_dict.items()} == \
                {sympy.srepr(a): sympy.srepr(b) for a, b in want[0].items()}, eqs
        else:
            assert got_dict is None, (eqs, got_dict, want)
    assert systems >= 20


def test_solve_one_free_of_the_unknown_matches_sympy() -> None:
    import sympy
    from sympy_extras.core.symbol import Symbol as NativeSymbol
    from sympy_extras.interop import expr_from_sympy
    from sympy_extras.solvers.linsolve import solve_one
    i, k, x = sympy.symbols('i k x')
    m = sympy.Dummy('m')
    native_m = expr_from_sympy(m)
    assert isinstance(native_m, NativeSymbol)
    for e in [sympy.Integer(0), sympy.Integer(3), 2*k + 1, i*x + x, 1 + i**-2, sympy.sin(i)]:
        assert sympy.solve(e, m) == []
        assert solve_one(expr_from_sympy(e), native_m) == [], e


def test_solve_one_drops_roots_contradicting_the_unknown() -> None:
    import sympy
    from sympy_extras.core import Symbol, Rational
    from sympy_extras.solvers.linsolve import solve_one
    j, p, i = Symbol('j', integer=True), Symbol('p', positive=True), Symbol('i')
    # 2*j - 1 with an integer j gave [1/2], where SymPy's solve gives []
    assert solve_one(2*j - 1, j) == []
    assert sympy.solve(2*sympy.Symbol('j', integer=True) - 1, sympy.Symbol('j', integer=True)) == []
    assert solve_one(p + 1, p) == []
    # kept when the assumptions hold, or cannot be decided
    assert solve_one(2*j - 4, j) == [2]
    assert solve_one(2*p - 1, p) == [Rational(1, 2)]
    assert solve_one(2*j - i, j) == [i/2]
