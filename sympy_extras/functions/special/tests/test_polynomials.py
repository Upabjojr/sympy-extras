"""The typed orthogonal polynomials against SymPy (the oracle), compared
without letting SymPy re-evaluate the results."""
from __future__ import annotations

import itertools
from typing import Callable

import mpmath
import pytest
import sympy

from sympy_extras._typing_sympy import as_expr as sympy_expr
from sympy_extras.core import I, Integer, Rational, Symbol, oo, zoo
from sympy_extras.interop import to_sympy
from sympy_extras.core.expr import Expr
from sympy_extras.functions import assoc_laguerre, chebyshevt, chebyshevu, conjugate, hermite, laguerre, legendre

x, y, a = Symbol('x'), Symbol('y'), Symbol('a')
n = Symbol('n', integer=True)
m = Symbol('m', integer=True, positive=True)
s = Symbol('s')

FUNCTIONS: list[tuple[Callable[[Expr, Expr], Expr], Callable[[sympy.Expr, sympy.Expr], sympy.Expr]]] = [
    (legendre, sympy.legendre), (chebyshevt, sympy.chebyshevt), (chebyshevu, sympy.chebyshevu),
    (hermite, sympy.hermite), (laguerre, sympy.laguerre)]


def sympy_value(e: Expr) -> sympy.Expr:
    return sympy_expr(to_sympy(e))


def same(ours: Expr, theirs: sympy.Basic) -> bool:
    return str(to_sympy(ours, evaluate=False)) == str(theirs)


def kept_at_infinity(ours: Expr, theirs: sympy.Basic) -> bool:
    """sympy/sympy#31275 (V108): SymPy gives ``f(n, oo) = oo`` (``(-1)**n*oo`` for ``laguerre``) for any
    symbolic order, the order 0 for which the polynomial is 1 included; the node is kept here unless the
    order is known to be positive."""
    text, expected = str(ours), str(theirs)
    return (', oo)' in text or ', -oo)' in text) and 'oo' in expected and 'oo)' not in expected


def reflected_by_sympy_only(ours: Expr, theirs: sympy.Basic) -> bool:
    """SymPy reflects ``f(s, -x)`` to ``(-1)**s*f(s, x)`` for the symbol ``s`` of no assumption:
    the functions of a noninteger degree have no parity, the node is kept here
    (test_sympy_issues_second_pass.test_reflection_of_the_polynomials_for_an_integer_degree_only)."""
    return '(-1)**s' in str(theirs) and '(-1)**s' not in str(ours) and ('(s, ' in str(ours))


def test_values_match_sympy() -> None:
    orders: list[Expr] = [Integer(k) for k in range(-4, 8)] + [n, -n, m, -m, -n - 1, -m - 2, s]
    points: list[Expr] = [x, -x, x + 1, 2*y, Integer(0), Integer(1), Integer(-1), Rational(1, 2), oo, -oo, I]
    different = []
    for order, z in itertools.product(orders, points):
        so, sz = sympy_value(order), sympy_value(z)
        for f, g in FUNCTIONS:
            try:
                theirs = g(so, sz)
            except ValueError:
                with pytest.raises(ValueError):
                    f(order, z)
                continue
            if z in (oo, -oo) and kept_at_infinity(f(order, z), theirs):
                assert order.is_positive is not True, (f.__name__, order, z)
                continue
            if f is laguerre and z == -oo and str(theirs) == 'nan' and ', -oo)' in str(f(order, z)):
                continue  # exp(x)*laguerre(-n - 1, -x) at -oo is 0*oo for SymPy; the node is kept here
            if reflected_by_sympy_only(f(order, z), theirs):
                continue
            if not same(f(order, z), theirs):
                different.append((f.__name__, str(order), str(z), str(f(order, z)), str(theirs)))
    for k in range(5):
        for alpha in [a, Integer(0), Integer(2), Rational(1, 2)]:
            for z in [x, Integer(0), x + 1]:
                ours, theirs = assoc_laguerre(k, alpha, z), sympy.assoc_laguerre(k, sympy_value(alpha), sympy_value(z))
                if not same(ours, theirs):
                    different.append(('assoc_laguerre', str(k), str((alpha, z)), str(ours), str(theirs)))
    for z in [Integer(0), oo, -oo, x]:
        ours, theirs = assoc_laguerre(m, a, z), sympy.assoc_laguerre(sympy_value(m), sympy_value(a), sympy_value(z))
        if not same(ours, theirs):
            different.append(('assoc_laguerre', 'm', str(z), str(ours), str(theirs)))
    assert not different, different[:20]


def test_symbolic_order_at_infinity() -> None:
    # sympy/sympy#31275 (V108): oo for a positive order only; the polynomials of order 0 are 1
    from sympy_extras.functions import hermite_prob
    positive, k = Symbol('p', positive=True), Symbol('k', integer=True, positive=True)
    for f in (legendre, chebyshevt, chebyshevu, hermite, hermite_prob):
        assert f(positive, oo) == oo and f(k, oo) == oo, f
        assert ', oo)' in str(f(s, oo)) and ', oo)' in str(f(n, oo)), f
        assert f(Integer(0), oo) == 1, f
    assert laguerre(k, oo) == (-1)**k*oo and laguerre(k, -oo) == oo and laguerre(positive, -oo) == oo
    assert ', oo)' in str(laguerre(positive, oo)) and ', oo)' in str(laguerre(s, oo))
    assert ', -oo)' in str(laguerre(s, -oo)) and laguerre(Integer(0), oo) == 1


def test_gegenbauer_at_minus_one() -> None:
    # sympy/sympy#31275 (V107): the pole at x = -1 for re(a) > 1/2 is that of the noninteger orders;
    # C_n^a(-1) = (-1)**n*gamma(2*a + n)/(gamma(2*a)*n!) for the polynomials
    from sympy_extras.functions import Gegenbauer, gegenbauer
    k, q = Symbol('k', integer=True, nonnegative=True), Symbol('q', noninteger=True)
    three_quarters = Rational(3, 4)
    assert gegenbauer(s, three_quarters, Integer(-1)) == Gegenbauer(s, three_quarters, Integer(-1))
    assert gegenbauer(q, three_quarters, Integer(-1)) == zoo
    value = gegenbauer(k, three_quarters, Integer(-1))
    for order, expected in ((2, Rational(15, 8)), (3, Rational(-35, 16))):
        assert gegenbauer(Integer(order), three_quarters, Integer(-1)) == expected
        assert abs(complex(value.subs(k, Integer(order)).evalf(20)) - complex(expected.evalf(20))) < 1e-15
        assert abs(mpmath.gegenbauer(order, 0.75, -1) - float(expected.p)/expected.q) < 1e-12


def test_gegenbauer_where_twice_the_parameter_is_a_nonpositive_integer() -> None:
    # gegenbauer(n, -1, x) was 0 for every n (SymPy's rule): the generating function (1 - 2*x*t + t**2)**(-a) is
    # 1 - 2*x*t + t**2 at a = -1, so C_0 = 1, C_1 = -2*x, C_2 = 1 and 0 from the degree 3 on. With a symbolic n
    # the closed forms at x = 1, -1, 0 were 0 or nan for 2*a = 0, -1, -2, ... (gamma functions at their poles)
    from sympy_extras.core import expand
    from sympy_extras.functions import Gegenbauer, gegenbauer
    t = Symbol('t')
    for parameter in (Integer(-1), Integer(-2), Integer(0), Integer(-3)):
        generating = expand((1 - 2*x*t + t**2)**(-parameter))
        for order in range(8):
            assert expand(gegenbauer(Integer(order), parameter, x) - generating.coeff(t, order)) == 0, (parameter, order)
    assert gegenbauer(Integer(1), Integer(-1), x) == -2*x and gegenbauer(Integer(2), Integer(-1), x) == 1
    k = Symbol('k', integer=True, nonnegative=True)
    degenerate: list[Rational] = [Integer(-1), Integer(-2), Integer(0), Rational(-1, 2), Rational(-3, 2)]
    for twice in degenerate:
        for point in (x, Integer(1), Integer(0)):
            assert gegenbauer(k, twice, point) == Gegenbauer(k, twice, point), (twice, point)
            assert point == 0 or gegenbauer(k, twice, -point) == (-1)**k*Gegenbauer(k, twice, point), (twice, point)
            assert gegenbauer(s, twice, point) == Gegenbauer(s, twice, point), (twice, point)
    # the polynomial is 0 where the degree is known to be above -2*a (an integer a)
    big = Symbol('big', integer=True, positive=True)
    assert gegenbauer(big + 2, Integer(-1), x) == 0 and gegenbauer(big, Integer(0), x) == 0
    assert gegenbauer(big + 1, Integer(-1), x) == Gegenbauer(big + 1, Integer(-1), x)
    # the numeric value of the node is the polynomial's (mpmath.gegenbauer(1, -1, 0.3) is 0)
    assert abs(complex(Gegenbauer(Integer(1), Integer(-1), Rational(3, 10)).evalf(20)) + 0.6) < 1e-15
    assert abs(complex(Gegenbauer(Integer(2), Integer(-2), Rational(3, 10)).evalf(20)) - 2.36) < 1e-15
    assert abs(complex(Gegenbauer(Integer(4), Rational(3, 4), Rational(3, 10)).evalf(20)) - complex(mpmath.gegenbauer(4, 0.75, 0.3))) < 1e-14
    # a negative degree which is not an integer is not a zero polynomial (it was 0)
    assert gegenbauer(Rational(-1, 2), a, x) == Gegenbauer(Rational(-1, 2), a, x) and gegenbauer(Integer(-2), a, x) == 0
    assert abs(mpmath.gegenbauer(-0.5, 0.7, 0.3)) > 0.5


def test_hermite_prob_at_zero() -> None:
    # sympy/sympy#31275 (V110): He_n(0) = 2**(n/2)*sqrt(pi)/gamma((1 - n)/2); SymPy has no 2**(n/2)
    from sympy_extras.functions import hermite_prob
    value = hermite_prob(s, Integer(0))
    for order in range(7):
        assert abs(complex(value.subs(s, Integer(order)).evalf(20)) - complex(hermite_prob(Integer(order), Integer(0)))) < 1e-15
    at = complex(value.subs(s, Rational(7, 10)).evalf(20))
    assert abs(at - complex(mpmath.hermite(0.7, 0)*mpmath.mpf(2)**-0.35)) < 1e-12


def test_derivatives_conjugates_numbers() -> None:
    sx, sn = sympy.Symbol('x'), sympy.Symbol('n', integer=True)
    for f, g in FUNCTIONS:
        assert same(f(n, x**2).diff(x), g(sn, sx**2).diff(sx)), f
        assert same(conjugate(f(n, x)), sympy.conjugate(g(sn, sx))), f
        ours = complex(sympy_value(f(n, x).subs(n, Integer(5)).subs(x, Rational(1, 3)).evalf(20)))
        assert abs(ours - complex(sympy.N(g(sympy.Integer(5), sympy.Rational(1, 3)), 20))) < 1e-14, f
    sa = sympy.Symbol('a')
    assert same(assoc_laguerre(n, a, x).diff(x), sympy.assoc_laguerre(sn, sa, sx).diff(sx))


def test_jacobi_gegenbauer_assoc_legendre_hermite_prob_match_sympy() -> None:
    from sympy_extras.functions import assoc_legendre, gegenbauer, hermite_prob, jacobi
    b = Symbol('b')
    p = Symbol('p', positive=True)
    parameters: list[Expr] = [Integer(0), Integer(1), Integer(-1), Rational(1, 2), Rational(-1, 2), a, -a, b, Integer(3)]
    points: list[Expr] = [x, -x, Integer(0), Integer(1), Integer(-1), oo, x + 1]
    orders: list[Expr] = [Integer(k) for k in range(0, 4)] + [n, m, -m, s]
    different = []

    def check(name: str, ours: Callable[[], Expr], theirs: Callable[[], sympy.Basic]) -> None:
        try:
            expected = theirs()
        except ValueError:
            with pytest.raises(ValueError):
                ours()
            return
        value = ours()
        if kept_at_infinity(value, expected):
            return
        if 'zoo' in str(expected) and ('gegenbauer(' in str(value) or '(-1)**' in str(value)) and value != zoo:
            return  # sympy/sympy#31275 (V107), see test_gegenbauer_at_minus_one
        if name == 'hermite_prob' and 'sqrt(pi)/gamma' in str(expected) and '2**' in str(value):
            return  # sympy/sympy#31275 (V110), see test_hermite_prob_at_zero
        if reflected_by_sympy_only(value, expected):
            return
        if not same(value, expected):
            different.append((name, str(value), str(expected)))

    for order, alpha, z in itertools.product(orders, parameters, points):
        so, sa, sz = sympy_value(order), sympy_value(alpha), sympy_value(z)
        degenerate = alpha in (Integer(0), Integer(-1), Rational(-1, 2)) and (alpha == -1 or not order.is_Number)
        if not degenerate:
            # (2*a a nonpositive integer: SymPy's 0 for a = -1 and its quotients of gamma functions at their poles
            # are not the polynomials, see test_gegenbauer_where_twice_the_parameter_is_a_nonpositive_integer)
            check('gegenbauer', lambda: gegenbauer(order, alpha, z), lambda: sympy.gegenbauer(so, sa, sz))
        for beta in [alpha, -alpha, b, Integer(2)]:
            sbeta = sympy_value(beta)
            if beta == alpha and alpha in (Integer(-1), Rational(-1, 2)) and not order.is_Number:
                continue  # the gegenbauer polynomials of the parameter -1/2 or 0, as above
            check('jacobi', lambda: jacobi(order, alpha, beta, z), lambda: sympy.jacobi(so, sa, sbeta, sz))
    for degree, order2, z in itertools.product(orders, [Integer(k) for k in range(-3, 4)] + [m, -m, p], points):
        sd, so2, sz = sympy_value(degree), sympy_value(order2), sympy_value(z)
        check('assoc_legendre', lambda: assoc_legendre(degree, order2, z), lambda: sympy.assoc_legendre(sd, so2, sz))
    for order, z in itertools.product(orders + [Integer(-2)], points):
        so, sz = sympy_value(order), sympy_value(z)
        check('hermite_prob', lambda: hermite_prob(order, z), lambda: sympy.hermite_prob(so, sz))
    assert not different, different[:20]
    sx, sn, sa, sb = sympy.Symbol('x'), sympy.Symbol('n', integer=True), sympy.Symbol('a'), sympy.Symbol('b')
    for ours, theirs in [(jacobi(n, a, b, x), sympy.jacobi(sn, sa, sb, sx)), (gegenbauer(n, a, x), sympy.gegenbauer(sn, sa, sx)),
                         (assoc_legendre(n, m, x), sympy.assoc_legendre(sn, sympy_value(m), sx)),
                         (hermite_prob(n, x), sympy.hermite_prob(sn, sx))]:
        assert same(ours.diff(x), theirs.diff(sx)), ours
        assert same(conjugate(ours), sympy.conjugate(theirs)), ours
