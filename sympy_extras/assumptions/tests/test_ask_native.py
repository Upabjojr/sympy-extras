"""The native ask, refine, simplify and resolve on objects of the core,
checked against hand computations (the SymPy faces are tested in
test_ask.py, test_refine.py and test_resolve.py)."""
from __future__ import annotations

from sympy_extras.core import Eq, Integer, Q, Rational, Symbol, mod, pi, sqrt
from sympy_extras.core.logic import BooleanTrue, and_
from sympy_extras.core.relational import Gt
from sympy_extras.functions import abs_, atan, atan2, exp, floor, log, max_, piecewise, sign, sin
from sympy_extras.sets import Complexes, Integers, Interval, Reals
from sympy_extras.assumptions.facts import Facts, element
from sympy_extras.assumptions.quantifiers import Exists, ForAll
from sympy_extras.assumptions.ask import _clearly_not_zero, _linear_ask, ask
from sympy_extras.assumptions.refine import _Refiner, refine, simplify
from sympy_extras.assumptions.resolve import resolve

x, y, n = Symbol('x'), Symbol('y'), Symbol('n')


def test_native_ask() -> None:
    assert ask(x**3 > 0, x > 0) is True and ask(x*y > 0, (x > 0) & (y < 0)) is False
    # the decomposition: (x - 1)**2 >= 0, and x > 1 under x > 2
    assert ask(x**2 - 2*x + 1 >= 0, x > 0) is True and ask(x > 1, x > 2) is True and ask(x > 3, x > 2) is None
    assert ask(x**2 >= 0) is None and ask(x**2 >= 0, domain=Reals) is True
    assert ask(element(x + 1, Integers), element(x, Integers)) is True
    assert ask(element(x, Interval(0, 1)), (x > Rational(1, 4)) & (x < Rational(1, 2))) is True
    assert ask(Q.positive(x), Eq(x, 3)) is True
    # the reality of roots and logarithms is decided by the signs of the arguments
    assert ask(element(sqrt(x - 2), Reals), x > 3) is True and ask(element(sqrt(x - 2), Reals), x < 1) is False
    assert ask(element(log(x), Reals), x < 0) is False
    # calculus, and the structure theorem
    assert ask(sin(x) > 0, (x > 0) & (x < pi)) is True and ask(exp(x) > 1 + x, x > 0) is True
    assert ask(Eq(4*atan(Rational(1, 5)) - atan(Rational(1, 239)), pi/4)) is True
    assert ask(ForAll(x, x**2 + 2*x*y + y**2 >= 0), domain=Reals) is True
    assert ask(Exists(y, y**2 < x), x > 0) is True


def test_native_ask_helpers() -> None:
    assert _linear_ask(x + y < 3, and_(x < 1, y < 1), [x, y]) is True
    assert _linear_ask(x**2 + y < 3, and_(x < 1, y < 1), [x, y]) is None
    assert _clearly_not_zero(log(x) - 1, Facts([x > 0])) is True
    assert _clearly_not_zero(log(x**2) - 2*log(x), Facts([x > 0])) is False


def test_native_refine() -> None:
    assert refine(abs_(x - 1) + sqrt(x**2), x > 2) == 2*x - 1
    assert refine(sqrt(x**2 - 2*x + 1), x > 1) == x - 1
    assert refine(sign(x*y), (x > 0) & (y < 0)) == Integer(-1)
    assert refine(max_(x, x**2), (x > 0) & (x < 1)) == x
    assert refine(floor(n + 1), element(n, Integers)) == n + 1
    assert refine(atan2(y, x - 1), x > 1) == atan(y/(x - 1))
    assert refine(mod(n**2 + n, 2), element(n, Integers)) == Integer(0)
    assert refine(piecewise((x, x > 0), (-x, True)), x < -1) == -x
    # Booleans: decided atom by atom, the rest kept
    assert refine((x > 0) & (y > 0), x > 1) == Gt(y, 0)
    assert isinstance(refine(x*y > 0, (x > 0) & (y > 0)), BooleanTrue)
    assert isinstance(refine(ForAll(y, y**2 + x > 0), x > 0), BooleanTrue)
    # the dummies carry the facts as assumptions of the core
    d = _Refiner(Facts([x > 0])).dummy_for(x)
    assert d is not None and d.is_positive is True


def test_native_simplify() -> None:
    assert simplify((x**2 - 1)/(x - 1), x > 1) == x + 1
    assert simplify(log(x) + log(y), (x > 0) & (y > 0)) == log(x*y)
    assert simplify(Eq(x**2, 1), x > 0) == Eq(x, 1)
    assert simplify(atan(x) + atan(1/x), x > 0) == pi/2
    assert simplify(x + Integer(0), ratio=1) == x


def test_native_resolve() -> None:
    b, c = Symbol('b'), Symbol('c')
    assert resolve(ForAll(x, x**2 + b*x + c > 0)) == (b**2 < 4*c)
    assert isinstance(resolve(ForAll(x, Exists(y, y > x))), BooleanTrue)
    assert resolve(Exists(y, Eq(x**2 + y**2, 1))) == and_(x >= -1, x <= 1)
    assert str(resolve(Exists(x, Eq(2*x, y)), domain=Integers)) == 'Eq(Mod(y, 2), 0)'
    assert str(resolve(Exists(x, Eq(Symbol('a')*x, 1)), domain=Complexes)) == 'Ne(a, 0)'
    assert resolve(x > 2, assumptions=x > 1) == Gt(x, 2)


def test_native_simplification_of_a_relation() -> None:
    # SymPy's Relational._eval_simplify, ported (simplify_boolean_of_core
    # sent every relation to SymPy): solved for its symbol when linear,
    # the coefficients made coprime, decided when the sides are numbers
    from sympy_extras.assumptions.refine import simplify_relational
    from sympy_extras.core.logic import false
    from sympy_extras.core.relational import Lt
    a, p = Symbol('a'), Symbol('p')
    assert simplify_relational(Lt(2*a, 1), 1.7, False, False, True) == Lt(a, Rational(1, 2))
    assert simplify_relational(Eq(2*p, 1), 1.7, False, False, True) == Eq(p, Rational(1, 2))
    assert simplify_relational(Lt(3*x + 6*y, 3), 1.7, False, False, True) == Lt(x + 2*y, 1)
    assert simplify_relational(Lt(x**2 + 1, x**2), 1.7, False, False, True) is false
    # an unchanged relation stays as it is
    assert simplify_relational(Lt(sin(x), 0), 1.7, False, False, True) == Lt(sin(x), 0)
