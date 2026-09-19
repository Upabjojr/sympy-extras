from __future__ import annotations

from sympy import (S, Q, Eq, Ne, Or, Not, Implies, Equivalent, Xor, ITE,
    Interval, FiniteSet, Symbol, Rational, sqrt, sin, pi)
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy.abc import x, y

from sympy_extras.assumptions import ask, element, ForAll, Exists


def test_ask_predicates() -> None:
    # answered through sympy's assumptions
    assert ask(x > 0, x > 0) is True
    assert ask(x < 0, x > 0) is False
    assert ask(x >= 0, x > 0) is True
    assert ask(Ne(x, 0), x > 0) is True
    assert ask(Eq(x, 0), x > 0) is False
    assert ask(x**3 > 0, x > 0) is True
    assert ask(x*y > 0, (x > 0) & (y > 0)) is True
    assert ask(x*y > 0, (x > 0) & (y < 0)) is False
    assert ask(x + y > 0, (x > 0) & (y > 0)) is True
    assert ask(x > 0, Q.positive(x)) is True
    assert ask(Q.positive(x), x > 0) is True
    assert ask(Q.real(x), x > y) is True
    assert ask(Q.real(x), Eq(x, y)) is None
    assert ask(element(x, S.Reals), x > 0) is True
    assert ask(element(x, S.Integers), element(x, S.Naturals)) is True
    assert ask(element(x + 1, S.Integers), element(x, S.Integers)) is True
    assert ask(element(x*y, S.Integers), element(x, S.Integers) & element(y, S.Integers)) is True
    assert ask(element(x/2, S.Integers), element(x, S.Integers)) is None
    assert ask(element(x, S.Naturals), element(x, S.Integers) & (x > 0)) is True
    assert ask(element(x, S.Rationals), element(x, S.Integers)) is True
    assert ask(element(x, S.Integers), element(x, S.Rationals)) is None
    assert ask(x > 0, element(x, S.Naturals)) is True
    assert ask(x >= 0, element(x, S.Naturals0)) is True
    assert ask(element(x, Interval(0, 1)), Eq(x, Rational(1, 2))) is True
    assert ask(element(x, FiniteSet(1, 2)), Eq(x, 3)) is False
    assert ask(Q.positive(x), Eq(x, 3)) is True
    assert ask(Q.integer(x), Eq(x, 3)) is True
    assert ask(Q.prime(x), Eq(x, 3)) is None
    assert ask(x > 0, Eq(x, 3)) is True
    assert ask(x > 2, Eq(x, 3)) is True
    assert ask(x > 4, Eq(x, 3)) is False


def test_ask_cad() -> None:
    # decided by the cylindrical algebraic decomposition
    assert ask(x > 1, x > 2) is True
    assert ask(x > 3, x > 2) is None
    assert ask(x < 3, x > 2) is None
    assert ask(x < 1, x > 2) is False
    assert ask(x**2 - 2*x + 1 >= 0, x > 0) is True
    assert ask(x**2 > 4, x > 2) is True
    assert ask(x**2 > 4, x > -2) is None
    assert ask(x**2 + y**2 > 0, (x > 0) | (y > 0)) is None
    assert ask(x**2 + y**2 > 0, ((x > 0) | (y > 0)) & element(x, S.Reals) & element(y, S.Reals)) is True
    assert ask(x**2 + y**2 > 0, (x > 0) | (y > 0), domain=S.Reals) is True
    assert ask(x**2 + y**2 >= 2*x*y, (x > 0) & (y > 0)) is True
    assert ask(x*y > 1, (x > 1) & (y > 1)) is True
    assert ask(x*y > 1, (x > 1) & (y > 0)) is None
    assert ask((x > 1) & (y > 0), (x > 2) & (y > 1)) is True
    assert ask((x > 1) | (y > 0), (x > 2) & (y < 1)) is True
    assert ask((x > 3) | (y < 0), (x < 2) & (y > 1)) is False
    assert ask(Implies(x > 2, x > 1), domain=S.Reals) is True
    assert ask(Implies(x > 1, x > 2), domain=S.Reals) is None
    assert ask(Equivalent(x > 1, x**3 > 1), domain=S.Reals) is True
    assert ask(Equivalent(x > 1, x**2 > 1), domain=S.Reals) is None
    assert ask(Xor(x > 0, x <= 0), domain=S.Reals) is True
    assert ask(ITE(x > 0, x > -1, x < 1), domain=S.Reals) is True
    assert ask(Not(x**2 < 0), domain=S.Reals) is True
    # a relation which only holds jointly
    assert ask((x > 0) | (x <= 0), domain=S.Reals) is True
    assert ask(Or(x**2 + y**2 < 1, x**2 + y**2 >= 1), domain=S.Reals) is True
    assert ask(x**2 + y**2 - 2*x*y >= 0, domain=S.Reals) is True
    assert ask(x**2 + x*y + y**2 > 0, domain=S.Reals) is None
    assert ask(x**2 + x*y + y**2 > 0, (x > 0) & (y > 0)) is True
    # rational coefficients
    assert ask(x/2 > Rational(1, 3), x > 1) is True
    # the assumptions have no real solution
    assert ask(x > 0, (x**2 < 0) & element(x, S.Reals)) is None


def test_ask_reals() -> None:
    # variables are not real unless assumed
    assert ask(x**2 >= 0) is None
    assert ask(x**2 >= 0, domain=S.Reals) is True
    assert ask(x**2 >= 0, element(x, S.Reals)) is True
    assert ask(x**2 >= 0, x > -1) is True
    assert ask(x**2 >= 0, Q.real(x)) is True
    assert ask(x**2 + y**2 >= 0, x > 0) is None
    assert ask(x**2 + y**2 >= 0, (x > 0) & element(y, S.Reals)) is True
    # an inequality in the query does not make the variables real
    assert ask(Or(x > 1, x < 2)) is None
    assert ask(Or(x > 1, x < 2), domain=S.Reals) is True
    # symbols with old style assumptions
    r = Symbol('r', real=True)
    assert ask(r**2 >= 0) is True
    p = Symbol('p', positive=True)
    assert ask(p**2 + p > 0) is True
    assert ask(p > 1) is None


def test_ask_undecidable() -> None:
    assert ask(sin(x) > 0, x > 0) is None
    assert ask(sqrt(x) > 0, x > 0) is True
    assert ask(sqrt(x) > 1, x > 4) is True  # decided by the sign analysis beyond polynomials
    assert ask(sqrt(x) > 3, x > 4) is None
    assert ask(element(x, S.Reals**2)) is None
    assert ask(Symbol('p')) is None
    assert ask(Or(Symbol('p'), x > 0), x > 1) is True


def test_ask_quantified() -> None:
    assert ask(ForAll(x, x**2 >= 0), domain=S.Reals) is True
    assert ask(ForAll(x, x**2 > 0), domain=S.Reals) is False
    assert ask(Exists(x, x**2 < 0), domain=S.Reals) is False
    assert ask(ForAll(x, x**2 + 2*x*y + y**2 >= 0), domain=S.Reals) is True
    assert ask(Exists(y, Eq(y**2, x)), x > 0) is True
    assert ask(Exists(y, Eq(y**2, x)), x < 0) is False
    assert ask(Exists(y, Eq(y**2, x)), domain=S.Reals) is None
    assert ask(ForAll(x, x**2 + y*x + 1 > 0), (y > -2) & (y < 2)) is True


def test_ask_inputs() -> None:
    assert ask(True) is True
    assert ask(False) is False
    assert ask(S.true) is True
    assert ask(x > 0, [x > 1, y > 0]) is True
    assert ask(x > 0, (x > 1, y > 0)) is True
    assert ask(2 > 1) is True
    assert ask(Eq(pi, 3)) is False
    assert ask(x > 0, domain=S.Integers) is None
    assert ask(x > 3, [x > 0, x < 0]) is None
    raises(ValueError, lambda: ask(x > 0, S.false))
    raises(TypeError, lambda: untyped(ask)(x > 0, domain=2))


def test_ask_linear() -> None:
    # linear formulas are decided by virtual substitution (Loos-Weispfenning)
    from sympy.abc import a, b, c, d, e
    from sympy_extras.assumptions.ask import _linear_ask
    assert ask(a + b + c + d + e < 5, (a < 1) & (b < 1) & (c < 1) & (d < 1) & (e < 1)) is True
    assert ask(2*a - b + c > 0, (a > b) & (b > c) & (c > 0)) is True
    assert ask(x + y > 3, (x < 1) & (y < 1)) is False
    assert ask(x + y > 0, (x < 1) & (y < 1)) is None
    assert ask(Eq(x + y, 2), (x < 1) & (y < 1)) is False
    assert ask(Ne(x + y, 2), (x < 1) & (y < 1)) is True
    assert ask(x - y > 0, (x > 2) & (y < 1)) is True
    assert ask(x - y > 1, (x > 2) & (y < 1)) is True
    assert ask(x - y > 2, (x > 2) & (y < 1)) is None
    assert _linear_ask(x + y < 3, (x < 1) & (y < 1), [x, y]) is True
    assert _linear_ask(x**2 + y < 3, (x < 1) & (y < 1), [x, y]) is None
    # a contradictory premise decides nothing
    assert _linear_ask(x < 3, (x < 1) & (x > 2), [x]) is None


def test_relations_are_undecided_where_a_function_is_not_real() -> None:
    # sympy-extras#61: the bounds of log(u) hold for u > 0, and the points
    # with u <= 0 were dropped by the constraint; the sign analysis found
    # 2 - x**(1/3) positive on -1 < x < 1
    from sympy import log, Rational, sqrt, exp
    assert ask(log(x) < 0, [x > -1, x < 1]) is None
    assert ask(log(x) < 0, [x > -1, x < 0]) is False
    assert ask(log(x) <= 1, [x > -1, x < 1]) is None
    assert ask(log(x) < x, [x > -1, x < 1]) is None
    assert ask(x**Rational(1, 3) < 2, [x > -1, x < 1]) is None
    assert ask(sqrt(x) < 2, [x > -1, x < 1]) is None
    # where they are real, as before
    assert ask(log(x) < 0, [x > 0, x < 1]) is True
    assert ask(log(x) > 0, x > 1) is True
    assert ask(x**Rational(1, 3) < 2, [x > 0, x < 1]) is True
    assert ask(exp(x) > 2, x > 1) is True


def test_a_complex_valued_expression_has_no_sign() -> None:
    # the sign analysis found exp(-I*x) > 0 on -1 < x < 1 (its value at 0
    # is 1 and it has no zero); found by the fuzz of sympy_extras.simplify,
    # which then split log(u*exp(-I*x)) as if exp(-I*x) were positive
    from sympy import I, exp, sin
    assert ask(exp(-I * x) > 0, [x > -1, x < 1]) is None
    assert ask(exp(I * x) + 2 > 0, [x > 0]) is None
    assert ask(exp(-x) > 0, [x > -1, x < 1]) is True
    assert ask(sin(x) + 2 > 0, [x > -1, x < 1]) is True


def test_equalities_between_elementary_functions_are_proved() -> None:
    # by the structure theorem (sympy_extras.simplify): two sides which are
    # the same function on the region; sampling and the sign analysis are
    # evidence, this is a proof
    from sympy import Rational, asin, atan, cos, exp, log, pi, sin, sqrt
    machin = 4 * atan(Rational(1, 5)) - atan(Rational(1, 239))
    assert ask(Eq(machin, pi / 4)) is True and ask(Ne(machin, pi / 4)) is False
    assert ask(Eq(4 * atan(Rational(1, 5)) - atan(Rational(1, 238)), pi / 4)) is False
    assert ask(Eq(sqrt(5 + 2 * sqrt(6)), sqrt(2) + sqrt(3))) is True
    assert ask(Eq(log(x**2), 2 * log(x)), x > 0) is True
    assert ask(Eq(log(x**2), 2 * log(x))) is None               # not identical, and equal for x > 0
    assert ask(Eq(log(x * y), log(x) + log(y)), [x > 0, y > 0]) is True
    assert ask(Eq(asin(x), atan(x / sqrt(1 - x**2))), [x > -1, x < 1]) is True
    assert ask(Eq(atan(x) + atan(1 / x), pi / 2), x > 0) is True
    assert ask(Eq(atan(x) + atan(1 / x), pi / 2), x < 0) is False or \
        ask(Eq(atan(x) + atan(1 / x), pi / 2), x < 0) is None
    # inequalities between identical sides
    assert ask(sin(x)**2 + cos(x)**2 >= 1, x > 0) is True
    assert ask(sin(x)**2 + cos(x)**2 > 1, x > 0) is False
    assert ask(exp(x + y) <= exp(x) * exp(y), [x > 0, y > 0]) is True
    assert ask(exp(x + y) < exp(x) * exp(y), [x > 0, y > 0]) is False
    # a difference which is not identically zero decides nothing by itself
    assert ask(Eq(exp(x), 1 + x)) is None
    # and what the other methods decided is as before
    assert ask(exp(x) > 0, x > 1) is True and ask(Eq(exp(x), 1), x > 0) is False


def test_the_structure_theorem_does_not_call_itself() -> None:
    # the tower asks about the signs of the arguments it meets; those
    # questions do not start a tower of their own (a depth counter), and
    # the question comes back
    from sympy import exp, log, sqrt
    import importlib
    from sympy_extras.assumptions import ask as ask_
    module = importlib.import_module('sympy_extras.assumptions.ask')     # the package exports the function ask
    assert ask_(Eq(log((exp(x) + 1) * (exp(x) + 2)), log(exp(x) + 1) + log(exp(x) + 2)), x > 0) is True
    assert ask_(Eq(sqrt((exp(x) + 1)**2), exp(x) + 1), x > 0) is True
    assert vars(module)['_structure_depth'] == 0
