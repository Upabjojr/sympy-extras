"""The native transcendental solver on objects of the core: the solutions
are checked by substitution into the equation (exact evaluation of the
elementary functions at the points), the intervals against hand
computations."""
from __future__ import annotations

from sympy_extras.core import Eq, Integer, Rational, Symbol, E, sqrt, cbrt, oo, pi
from sympy_extras.core.basic import Basic
from sympy_extras.core.expr import Expr
from sympy_extras.core.evalf import N
from sympy_extras.core.expr import as_expr
from sympy_extras.core.logic import true
from sympy_extras.core.relational import Relational
from sympy_extras.functions import Abs, abs_, cos, cosh, exp, log, sin, sinh, tan
from sympy_extras.sets import EmptySet, Integers, finite_set, interval
from sympy_extras.sets.conditionset import ConditionSet
from sympy_extras.sets.fancysets import ImageSet
from sympy_extras.sets.sets import FiniteSet, Set, Union
from sympy_extras.solvers.transcendental import Reduction, polynomialize, solve_transcendental

x = Symbol('x')
a = Symbol('a')


def _holds(equation: Relational, value: Expr) -> bool:
    """The equation at the value: exactly, or within 10**-25 at 30 digits
    (sinh(log(2 + sqrt(5))) is not written 2)."""
    if equation._xreplace_evaluated({x: value}) is true:
        return True
    difference = as_expr((equation.lhs - equation.rhs)._xreplace_evaluated({x: value}))
    return (abs_(N(difference, 30)) < Rational(1, 10**25)) is true


def _image_points(s: Set) -> list[Expr]:
    """The points of the images of the integers for k in -2..2."""
    points: list[Expr] = []
    for part in (s.args if isinstance(s, Union) else (s,)):
        assert isinstance(part, ImageSet) and part.base_sets == (Integers,), part
        image = part.image
        assert isinstance(image, Expr)
        k = part.variables[0]
        for n in range(-2, 3):
            point = image.subs(k, Integer(n))
            assert isinstance(point, Expr)
            points.append(point)
    return points


def test_polynomialize_native() -> None:
    r = polynomialize(Eq(exp(2*x) - 3*exp(x) + 2, 0), x)
    assert isinstance(r, Reduction) and r.family == 'exp' and r.argument == x
    t = r.variable
    assert r.formula == Eq(t**2 - 3*t + 2, 0) and r.kernel() == exp(x)
    r = polynomialize(Eq(2**x + 4**x, 6), x)
    assert r is not None and r.family == 'exp' and r.argument == x*log(2)
    r = polynomialize(Eq(sin(2*x), cos(x)), x)
    assert r is not None and r.family == 'trig' and r.kernel() == tan(x/2)
    r = polynomialize(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x)
    assert r is not None and r.family == 'root' and r.order == 6
    r = polynomialize(Eq(log(x)**2, 3*log(x) - 2), x)
    assert r is not None and r.family == 'log'
    assert polynomialize(Eq(exp(x) + sin(x), 1), x) is None
    assert polynomialize(Eq(x*exp(x), 1), x) is None
    assert polynomialize(Eq(exp(x) + exp(x**2), 1), x) is None


def test_exponential_equations_native() -> None:
    for equation, expected in ((Eq(exp(2*x) - 3*exp(x) + 2, 0), finite_set(0, log(2))),
                               (Eq(exp(3*x) - 6*exp(2*x) + 11*exp(x) - 6, 0), finite_set(0, log(2), log(3))),
                               (Eq(sinh(x), 2), finite_set(log(2 + sqrt(5)))),
                               (Eq(cosh(x), 1), finite_set(0)),
                               (Eq(2**x, 8), finite_set(3)),
                               (Eq(exp(x + 1), 3), finite_set(log(3) - 1)),
                               (Eq(exp(x), pi), finite_set(log(pi)))):
        found = solve_transcendental(equation, x)
        assert found == expected, (equation, found)
        assert isinstance(found, FiniteSet)
        for v in found.args:
            assert isinstance(v, Expr) and _holds(equation, v), (equation, v)
    assert solve_transcendental(Eq(exp(x), -1), x) is EmptySet
    assert solve_transcendental(exp(x) > 2, x) == interval(log(2), oo, True, True)
    assert solve_transcendental(exp(2*x) - 5*exp(x) + 6 <= 0, x) == interval(log(2), log(3))
    # the kernel with a parameter: SymPy's solveset is not needed when the
    # solver is given (here: the parameter positive)
    found = solve_transcendental(Eq(exp(x), a), x, solver=lambda f, v: finite_set(a) if f == Eq(v - a, 0) else None,
                                 decide=lambda c: True if c == (a > 0) else None)
    assert found == finite_set(log(a))
    undecided = solve_transcendental(Eq(exp(x), a), x, solver=lambda f, v: finite_set(a), decide=lambda c: None)
    assert isinstance(undecided, ConditionSet) and undecided.args[1] == (a > 0)


def test_trigonometric_equations_native() -> None:
    for equation, roots in ((Eq(sin(x) + cos(x), 1), [Integer(0), pi/2]),
                            (Eq(cos(x), -1), [pi]),
                            (Eq(tan(x), 1), [pi/4, 5*pi/4]),
                            (Eq(sin(2*x), cos(x)), [pi/6, pi/2, 5*pi/6, 3*pi/2]),
                            (Eq(sin(2*x), Rational(1, 2)), [pi/12, 5*pi/12, 13*pi/12])):
        found = solve_transcendental(equation, x)
        assert found is not None, equation
        points = _image_points(found)
        for p in points:
            assert _holds(equation, p), (equation, p)
        for r in roots:
            assert any((p - r).is_zero for p in points), (equation, r, points)
    assert solve_transcendental(Eq(sin(x), 2), x) is EmptySet


def test_logarithms_roots_and_absolute_values_native() -> None:
    assert solve_transcendental(Eq(log(x)**2, 3*log(x) - 2), x) == finite_set(E, exp(2))
    assert solve_transcendental(log(x) > 1, x) == interval(E, oo, True, True)
    assert solve_transcendental(Eq(log(x**2 - 3), 0), x) == finite_set(-2, 2)
    assert solve_transcendental(Eq(sqrt(x + 1) + cbrt(x + 1), 2), x) == finite_set(0)
    assert solve_transcendental(sqrt(x) < 2, x) == interval(0, 4, False, True)
    assert solve_transcendental(Eq(x**Rational(3, 2), 8), x) == finite_set(4)
    assert solve_transcendental(Eq(Abs(x - 1) + Abs(x + 1), 3), x) == finite_set(Rational(-3, 2), Rational(3, 2))
    assert solve_transcendental(Abs(exp(x) - 3) < 1, x) == interval(log(2), log(4), True, True)


def test_unreducible_native() -> None:
    assert solve_transcendental(Eq(log(x) + log(x + 1), 1), x) is None


def test_lambert_native() -> None:
    # the candidates come from SymPy's solve (through the face); the
    # branch -1 gives the second real root of x*exp(-x) = 1/4
    from sympy_extras.functions import LambertW
    found = solve_transcendental(Eq(x*exp(x), 1), x)
    assert found == finite_set(LambertW(Integer(1)))
    two = solve_transcendental(Eq(x*exp(-x), Rational(1, 4)), x)
    assert isinstance(two, FiniteSet) and len(two.args) == 2
    for v in two.args:
        assert isinstance(v, Basic) and v.has(LambertW)
