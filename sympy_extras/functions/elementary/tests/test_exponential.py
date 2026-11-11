"""The typed exp and log against SymPy's (the oracle): evaluated forms,
derivatives, numerical values and facts; and the conversions."""
from __future__ import annotations

import random
from typing import Callable, Optional

import sympy

from sympy_extras._typing import as_expr as sympy_expr
from sympy_extras.core import E, I, Integer, Rational, Symbol, pi
from sympy_extras.interop import to_sympy, from_sympy
from sympy_extras.core.expr import Expr, as_expr
from sympy_extras.functions.elementary.exponential import Exp, Log, exp, log

x, y = Symbol('x'), Symbol('y')
r = Symbol('r', real=True)
p = Symbol('p', positive=True)

#: SymPy's answer for each fact compared
FACTS: dict[str, Callable[[sympy.Expr], Optional[bool]]] = {
    'real': lambda s: s.is_real, 'positive': lambda s: s.is_positive, 'negative': lambda s: s.is_negative,
    'zero': lambda s: s.is_zero, 'finite': lambda s: s.is_finite, 'rational': lambda s: s.is_rational,
    'algebraic': lambda s: s.is_algebraic, 'integer': lambda s: s.is_integer, 'complex': lambda s: s.is_complex,
    'extended_real': lambda s: s.is_extended_real, 'nonnegative': lambda s: s.is_nonnegative,
}


def _arguments() -> list[Expr]:
    rng = random.Random(1)
    values: list[Expr] = [Integer(0), Integer(1), Integer(-1), Integer(2), Integer(-3), Integer(15), Rational(1, 7),
                          Rational(-2, 3), E, I, -I, 3*I, Rational(-1, 2)*I, pi*I, 2*pi*I, 3*pi*I/2, pi*I/2,
                          Rational(7, 3)*pi*I, Rational(-5, 4)*pi*I, x, r, p, x + 1, 2*x, x + y, log(x), 2*log(x),
                          x + log(y), log(x) + log(y), p - 1, -r, r**2]
    for _ in range(25):
        a, b = rng.randint(-5, 5), rng.randint(1, 4)
        values.append(Rational(a, b)*pi*I)
    return values


def test_evaluation_matches_sympy() -> None:
    for a in _arguments():
        sa = to_sympy(a)
        # without re-evaluation by SymPy, which would hide a difference
        assert str(to_sympy(exp(a), evaluate=False)) == str(sympy.exp(sa)), a
        if a != 0:
            assert str(to_sympy(log(a), evaluate=False)) == str(sympy.log(sa)), a


def test_exp_of_infinity_times_a_number() -> None:
    # exp(oo*sign(log(5/2) + I*pi)) stayed unevaluated, where SymPy's exp
    # gives zoo; the core's limit of (-5/2)**x at oo was that expression
    # instead of zoo (found by the port of assumptions.limits)
    from sympy_extras.core import oo
    from sympy_extras.core.mul import Mul
    from sympy_extras.functions import sign
    c = sign(log(Rational(5, 2)) + I*pi)
    for a in (Mul(oo, c), Mul(-oo, c), Mul(oo, I), Mul(oo, -c), Mul(oo, 1 + I), Mul(oo, -1 + I)):
        sa = to_sympy(a, evaluate=False)
        assert str(to_sympy(exp(a), evaluate=False)) == str(sympy.exp(sa)), a
    assert str(exp(Mul(oo, c))) == 'zoo' and exp(Mul(-oo, c)) == 0 and str(exp(Mul(oo, I))) == 'nan'


def test_log_with_a_base() -> None:
    cases: list[tuple[Expr, Expr]] = [(Integer(24), Integer(2)), (Integer(8), Integer(2)), (Rational(1, 8), Integer(2)),
                                      (Rational(3, 4), Integer(2)), (Integer(81), Integer(3)), (Integer(10), Integer(10)),
                                      (Integer(7), E)]
    for value, base in cases:
        assert str(to_sympy(log(value, base), evaluate=False)) == str(sympy.log(to_sympy(value), to_sympy(base))), \
            (value, base)
    assert str(log(5, 1)) == 'zoo' and str(log(1, 1)) == 'nan'


def test_derivatives() -> None:
    sx = sympy.Symbol('x')
    for f in [exp(x), log(x), exp(x**2), log(x**2 + 1), exp(log(x) + x), log(exp(x) + 1), x*exp(2*x)*log(x)]:
        assert sympy.simplify(sympy_expr(to_sympy(f.diff(x))) - sympy_expr(to_sympy(f)).diff(sx)) == 0, f
    # E**f is exp(f), and its derivative goes through the chain rule
    assert (E**(x**2)).diff(x) == 2*x*exp(x**2)
    assert (2**x).diff(x) == 2**x*log(2)


def test_numerical_values() -> None:
    for a in [Integer(1), Rational(1, 3), I, 1 + I, pi, Integer(-2)]:
        for f in (exp, log):
            value = f(a).evalf(30)
            expected = sympy.N(sympy_function(f)(to_sympy(a)), 30)
            assert abs(complex(sympy_expr(to_sympy(value))) - complex(expected)) < 1e-25, (f, a)


def sympy_function(f: object) -> type[sympy.Function]:
    return sympy.exp if f is exp else sympy.log


def test_facts_agree_with_sympy() -> None:
    for a in _arguments():
        for node in (exp(a), log(a) if a != 0 else exp(a)):
            s = sympy_expr(to_sympy(node))
            for name, sympy_fact in FACTS.items():
                got = node._ask(name)
                expected = sympy_fact(s)
                if got is not None:
                    assert got == expected, (node, name, got, expected)


def test_nodes_and_conversions() -> None:
    assert Exp(0) != Integer(1) and exp(0) == 1 and Exp(x).func is Exp
    assert from_sympy(sympy.exp(sympy.Symbol('x'))) == Exp(x)
    assert type(from_sympy(sympy.log(sympy.Symbol('x')))) is Log
    assert to_sympy(Exp(x)) == sympy.exp(sympy.Symbol('x'))
    assert as_expr(from_sympy(sympy.exp(1))) == E
    assert E**x == exp(x) and exp(x).as_base_exp() == (E, x)
    assert exp(-x).as_numer_denom() == (1, exp(x))
    assert str(exp(x)*exp(y)) == 'exp(x)*exp(y)'
    assert (exp(x + y)).expand() == exp(x)*exp(y)
