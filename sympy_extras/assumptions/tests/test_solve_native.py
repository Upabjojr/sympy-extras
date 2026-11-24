"""The native solve on objects of the core, checked by substitution and
against hand computations (the SymPy face is tested in test_solve.py and
the solvers' tests)."""
from __future__ import annotations

from sympy_extras.core import Basic, Eq, Expr, I, Integer, Rational, Symbol, Tuple, as_expr, sqrt
from sympy_extras.core.logic import true
from sympy_extras.functions import exp, log, sin
from sympy_extras.sets import (ConditionSet, Complexes, EmptySet, FiniteSet, ImageSet, Integers, PointSet,
    Reals, Union, finite_set, interval)
from sympy_extras.assumptions.facts import element
from sympy_extras.assumptions.solve import _refuted, _same_number, solve
from sympy_extras.sets.sets import point_set

x, y, a, b = Symbol('x'), Symbol('y'), Symbol('a'), Symbol('b')


def _vanishes(f: Expr, values: dict[Basic, Basic]) -> bool:
    e = as_expr(f.xreplace(values)).doit()
    return abs(complex(e.evalf(30))) < 1e-20


def test_rational_equations_over_the_complex_numbers() -> None:
    assert solve(x**2 - 2, x) == finite_set(-sqrt(2), sqrt(2))
    assert solve(x**2 + 1, x) == finite_set(-I, I)
    assert solve(x**2 - a, x) == finite_set(-sqrt(a), sqrt(a))
    assert solve(a*x - 1, x, a > 0) == finite_set(1/a)
    assert solve(1/x - 2, x, x > 0) == finite_set(Rational(1, 2))
    # the assumptions choose among the solutions
    assert solve(x**2 - a**2, x, (a > 0) & (x < 0)) == finite_set(-a)
    found = solve(x**2 - a, x, x > 0)
    assert isinstance(found, ConditionSet) and found.base_set == finite_set(-sqrt(a), sqrt(a))
    assert solve(x**2 - 2, x, (x > 0) & (x < 0)) == EmptySet


def test_real_unknowns_and_domains() -> None:
    assert solve(x**2 - 2, x, x > 0) == finite_set(sqrt(2))
    assert solve(x**2 + 1, x, domain=Reals) == EmptySet
    found = solve((x**2 - 2 > 0) & (x < 3), x, domain=Reals)
    assert isinstance(found, Union) and len(found.args) == 2
    assert solve(x**2 - 4, x, element(x, Integers) & (x > 0)) == finite_set(2)
    assert solve(2*x - 1, x, element(x, Integers)) == EmptySet
    assert solve(exp(x) - 2, x, x > 1) == EmptySet
    assert solve(exp(x) - 2, x, x > 0) == finite_set(log(2))
    found = solve(sin(x) > 0, x, (x > 0) & (x < 10))
    assert isinstance(found, Union) and all(isinstance(piece, type(interval(0, 1))) for piece in found.args)


def test_systems() -> None:
    found = solve([x**2 + y**2 - 1, x - y], [x, y], x > 0)
    assert isinstance(found, PointSet) and found == point_set(Tuple(sqrt(2)/2, sqrt(2)/2))
    found = solve([x**2 + y**2 - 5, x*y - 2], [x, y])
    assert isinstance(found, PointSet) and len(found) == 4
    for point in found.points:
        u, v = point.args
        assert _vanishes(x**2 + y**2 - 5, {x: u, y: v}) and _vanishes(x*y - 2, {x: u, y: v})
    found = solve([x**2 + y**2 <= 1, x + y >= 1], [x, y], domain=Reals)
    assert isinstance(found, ConditionSet) and found.symbols == (x, y)
    assert solve(Eq(3*x + 5*y, 22), [x, y], (x >= 0) & (y >= 0), domain=Integers) == point_set(Tuple(4, 2))
    image = solve(Eq(3*x + 5*y, 22), [x, y], domain=Integers)
    assert isinstance(image, ImageSet) and len(image.variables) == 1
    t = image.variables[0]
    values = image.image
    assert isinstance(values, Tuple)
    for k in range(-3, 4):
        u, v = [as_expr(c.subs(t, Integer(k))) for c in values.args]
        assert 3*u + 5*v == 22


def test_cases_of_the_parameters() -> None:
    assert str(solve(a*x - 1, x, cases=True)) == 'ConditionSet(x, Ne(a, 0), {1/a})'
    found = solve(a*x - b, x, cases=True)
    assert isinstance(found, Union) and len(found.args) == 2
    assert solve(a*x - b, x, a > 0, cases=True) == finite_set(b/a)
    assert solve(Eq(x**2, a), x, a < 0, domain=Reals, cases=True) == EmptySet
    assert solve(x**2 - 2, x, cases=True) == finite_set(-sqrt(2), sqrt(2))
    assert solve(a*x - b, x, Eq(a, 0), cases=True) == ConditionSet(x, Eq(a, 0) & Eq(b, 0), Complexes)


def test_helpers() -> None:
    # two radical forms of one number are one solution (sympy-extras#50)
    assert _same_number(Rational(15, 26) + 3*sqrt(3)/13, sqrt(45*sqrt(3)/169 + Rational(333, 676))) is True
    assert _same_number(I, -I) is False
    equations = [x**5 - x - 1 - y, y**2 - 2]
    assert _refuted(PointSet(Tuple(x, sqrt(2))), equations, [x, y]) is True
    assert _refuted(PointSet(Tuple(1/y, y)), [x*y - 1], [x, y]) is False


def test_errors() -> None:
    from sympy_extras._testing import untyped
    try:
        solve(Integer(1), [])
    except ValueError:
        pass
    else:
        raise AssertionError('no unknown')
    try:
        untyped(solve)(Reals, x)
    except TypeError:
        pass
    else:
        raise AssertionError('a set is not an equation')
    assert isinstance(solve(x - 1, x, true), FiniteSet)


def test_solutions_contradicting_the_flags_of_the_unknown_are_dropped() -> None:
    from sympy_extras.core import pi
    from sympy_extras.core.relational import eq
    k = Symbol('k', integer=True)
    p = Symbol('p', positive=True)
    # pi*k = 1 gave {1/pi} for an integer k, which 1/pi is not (SymPy's
    # solve drops it, as the native solve did for 2*k = 1)
    assert solve(eq(pi*k - 1, Integer(0)), k) == EmptySet
    assert solve(2*k - 1, k) == EmptySet
    # p**2 = 4 gave {-2, 2} for a positive p
    assert solve(p**2 - 4, p) == finite_set(2)
    # k**2 = 2 gave {-sqrt(2), sqrt(2)} for an integer k
    assert solve(k**2 - 2, k) == EmptySet
    # a solution whose flags cannot be decided is kept, the facts about
    # the parameters decide the others
    assert solve(a*k - 1, k) == finite_set(1/a)
    q = Symbol('q', positive=True)
    assert solve(a*q + 1, q, a > 0) == EmptySet
    assert solve(a*q + 1, q, a < 0) == finite_set(-1/a)
    # systems: the points are checked for every unknown
    assert solve([x + y - 1, x - y], [x, y]) == point_set(Tuple(Rational(1, 2), Rational(1, 2)))
    m = Symbol('m', integer=True)
    assert solve([x + m - 1, x - m], [x, m]) == EmptySet


def test_a_point_real_only_where_the_condition_fails() -> None:
    # I*y for a real y is real at y = 0 only, where it is 0: outside (p, q)
    # for p > 0, a case of the parameters for p < 0 < q (Q.real(I*y) is no
    # longer False for a real y, sympy-extras#25, and the zeros of
    # x**2 + y**2 on (p, q) were left undecided)
    x, y, p, q = Symbol('x'), Symbol('y'), Symbol('p'), Symbol('q')
    assert solve(x**2 + y**2, x, [x > p, x < q, p > 0, q > 0], domain=Reals) == EmptySet
    assert isinstance(solve(x**2 + y**2, x, [x > p, x < q, p < 0, q > 0], domain=Reals), ConditionSet)


def test_sympy_issue_31275_V276_system_with_transcendental_equations() -> None:
    # "the system cannot be solved" (NotImplementedError); SymPy's nonlinsolve returns {(x, 0),
    # (-exp(y) - 1/2, y)}, which its own test pins: neither satisfies the second equation
    # (sympy/sympy#22413). The solutions are (LambertW(-exp(2)/2, n)/2 - 1, 0) and, with
    # x = -exp(y) - 1/2, those of an equation in y alone
    import cmath
    first = 4*y*(2*x + 2*exp(y) + 1)*exp(2*x)
    second = 4*x*exp(2*x) + 4*y*exp(2*x + y) + 4*exp(2*x + y) + 1
    found = solve([Eq(first, 0), Eq(second, 0)], [x, y])
    assert isinstance(found, Union)
    families = [s for s in found.sets if isinstance(s, ImageSet)]
    conditions = [s for s in found.sets if isinstance(s, ConditionSet)]
    assert len(families) == 1 and len(conditions) == 1 and len(found.sets) == 2
    n = families[0].variables[0]
    image = families[0].image
    assert isinstance(image, Tuple) and image.args[1] == 0
    for k in range(-2, 3):
        values: dict[Basic, Basic] = {x: as_expr(image.args[0]).subs(n, k), y: Integer(0)}
        assert _vanishes(first, values) and _vanishes(second, values)
    # a solution with y != 0, found numerically (Newton's method on the second equation at
    # x = -exp(y) - 1/2): it satisfies the condition of the set, and the condition has the first equation
    from sympy_extras.core import Float
    point = complex(1.0407812886383698, 1.218274716694042)
    y0 = Float(point.real) + Float(point.imag)*I
    x0 = Float((-cmath.exp(point) - 0.5).real) + Float((-cmath.exp(point) - 0.5).imag)*I
    relations = conditions[0].condition.args
    assert Eq(x, -exp(y) - Rational(1, 2)) in relations and len(relations) == 2
    for relation in relations:
        assert isinstance(relation, Eq)
        residual = as_expr(relation.lhs - relation.rhs).subs(x, x0).subs(y, y0)
        assert abs(complex(residual.evalf(20))) < 1e-12
    assert abs(complex(second.subs(x, x0).subs(y, y0).evalf(20))) < 1e-12
    # over the reals the first family is empty
    real = solve([Eq(first, 0), Eq(second, 0)], [x, y], domain=Reals)
    assert isinstance(real, ConditionSet)


def test_systems_by_factors_and_substitution() -> None:
    # NotImplementedError before for every system with a transcendental equation
    found = solve([Eq(sin(x), 0), Eq(y, x)], [x, y])
    assert isinstance(found, Union) and all(isinstance(s, ImageSet) for s in found.sets)
    for s in found.sets:
        assert isinstance(s, ImageSet) and isinstance(s.image, Tuple)
        for k in (-1, 0, 2):
            px, py = [as_expr(c).subs(s.variables[0], k) for c in s.image.args]
            assert sin(px) == 0 and px == py
    assert solve([Eq(exp(x), 2), Eq(y, x - log(2))], [x, y], domain=Reals) == point_set(Tuple(log(2), Integer(0)))
    # the points which are not in the domain or fail a condition are dropped
    assert solve([Eq(exp(x), 2), Eq(y, log(x - 1))], [x, y], domain=Reals) == EmptySet
    assert solve([Eq(x*exp(y), 0), Eq(x + y, 1), x > 0], [x, y]) == EmptySet
    assert solve([Eq(exp(x) + y, 0), Eq(exp(x) + y, 1)], [x, y]) == EmptySet
    # a factor which is solved and one which is not
    mixed = solve([Eq(y*(x - exp(y)), 0), Eq(x + y, 1)], [x, y])
    assert isinstance(mixed, Union) and point_set(Tuple(Integer(1), Integer(0))) in mixed.sets
    assert any(isinstance(s, ConditionSet) for s in mixed.sets)
