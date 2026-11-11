"""The RootSum node and root_sum on the core, and the domain EX."""
from __future__ import annotations

from sympy_extras.core import I, Lambda, Rational, sqrt
from sympy_extras.core.symbol import Symbol
from sympy_extras.core.evalf import numeric_value
from sympy_extras.functions import log
from sympy_extras.polys.domains.expressiondomain import EX
from sympy_extras.polys.rootoftools import RootSum, root_sum

t, x, a = Symbol('t'), Symbol('x'), Symbol('a')


def test_unevaluated_node() -> None:
    r = RootSum(t**2 - 1, Lambda(t, t**2), t)
    # the constructor does not evaluate; doit writes the roots
    assert isinstance(r, RootSum) and r.args == (t**2 - 1, Lambda(t, t**2), t)
    assert r.doit() == 2
    assert r.free_symbols == set() and r.bound_symbols == (t,)


def test_evaluation() -> None:
    L = Lambda(t, t*log(x + t))
    assert root_sum(t - 2, L, t) == 2*log(x + 2)
    assert root_sum(t**3 - 1, L, t, quadratic=True) == log(x + 1) + \
        (-Rational(1, 2) - sqrt(3)*I/2)*log(x - Rational(1, 2) - sqrt(3)*I/2) + \
        (-Rational(1, 2) + sqrt(3)*I/2)*log(x - Rational(1, 2) + sqrt(3)*I/2)
    # the quadratic factor stays without quadratic=True
    r = root_sum(4*t**3 - 2*t - 2, L, t)
    assert r == log(x + 1) + RootSum(2*t**2 + 2*t + 1, L, t)
    # constants taken out
    assert root_sum(t**3 - t - 1, Lambda(t, 2*t*log(x + t) + 3), t) == \
        2*RootSum(t**3 - t - 1, L, t) + 9
    # a function free of the variable: the degree times it
    assert root_sum(t**3 - t - 1, Lambda(t, x), t) == 3*x
    # a rational function: the trace (sum of the roots, of their squares,
    # of 1/(x + r))
    assert root_sum(t**3 - t - 1, Lambda(t, t), t) == 0
    assert root_sum(t**3 - t - 1, Lambda(t, t**2), t) == 2
    assert root_sum(t**3 - t - 1, Lambda(t, 1/(x + t)), t) == (3*x**2 - 1)/(x**3 - x + 1)
    # parameters: the roots scaled (preprocess_roots)
    assert root_sum(t**2 + 1/(4*a**2), L, t, quadratic=True) == (-I*log(x - I/(2*a))/2 + I*log(x + I/(2*a))/2)/a


def test_derivative() -> None:
    r = root_sum(t**3 - t - 1, Lambda(t, t*log(x + t)), t)
    # SymPy's form, factored
    assert str(r.diff(x)) == '-(2*x - 3)/(x**3 - x + 1)'
    # SymPy 1.14 gives 1/x: RootSum.new returns a function free of the
    # variable without the degree
    assert RootSum(t**3 - t - 1, Lambda(t, log(t*x)), t).diff(x) == 3/x


def test_substitution_and_numerical_value() -> None:
    r = RootSum(t**3 + t + 1, Lambda(t, t*log(x - t)), t)
    s = r.subs(x, 2)
    assert isinstance(s, RootSum) and s.free_symbols == set()
    # the bound variable is left alone
    assert r.subs(t, 5) == r
    value = numeric_value(s, 80)
    explicit = numeric_value(s.doit(), 80)
    assert value is not None and explicit is not None and abs(value - explicit) < 1e-20


def test_interop_round_trip() -> None:
    import sympy
    from sympy_extras.interop import from_sympy, to_sympy
    T, X = sympy.symbols('t x')
    s = sympy.RootSum(T**3 - T - 1, sympy.Lambda(T, T*sympy.log(X + T)))
    native = from_sympy(s)
    assert isinstance(native, RootSum)
    assert to_sympy(native) == s
    X_ = from_sympy(X)
    assert isinstance(X_, Symbol)
    assert to_sympy(native.diff(X_)) == s.diff(X)


def test_expression_domain() -> None:
    # SymPy's EX: cancel, then expand over the products
    assert EX.add(1/a, a) == a + 1/a
    assert EX.exquo(a**2 - 1, a + 1) == a - 1
    assert EX.is_zero(EX.sub(sqrt(2)*sqrt(3), sqrt(6)))
    assert EX.gcd(2*a, 4*a) == 1
    assert EX.is_negative(-a) and not EX.is_negative(a)
