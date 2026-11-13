"""The ported rewrite rules of the orthogonal polynomials (their sums,
``hermite`` and ``hermite_prob`` in terms of each other): equal to the
polynomial at integer degrees once evaluated."""
from __future__ import annotations

from sympy.testing.pytest import raises

from sympy_extras._numeric import reliable_complex
from sympy_extras.concrete.expr_with_limits import Limits, Product, Sum
from sympy_extras.core.add import add
from sympy_extras.core.expr import as_expr
from sympy_extras.core.mul import mul
from sympy_extras.core import Integer, Rational, Symbol
from sympy_extras.core.expr import Expr
from sympy_extras.functions.special import polynomials as p

x, a, b = Symbol('x'), Symbol('a'), Symbol('b')
n = Symbol('n', integer=True, nonnegative=True)
POINT: dict[Symbol, Expr] = {x: Rational(-2, 5), a: Rational(1, 2), b: Rational(1, 3)}


def _value(e: Expr) -> complex:
    v = reliable_complex(e, 15, POINT)
    assert v is not None, e
    return v


def _explicit(e: Expr) -> Expr:
    """``e`` with its sums and products of integer bounds written out term
    by term (``doit`` would evaluate the symbolic inner sums first, see
    the euler test)."""
    if isinstance(e, (Sum, Product)):
        return _unrolled(e.function, list(e.limits), isinstance(e, Sum))
    if not e.args:
        return e
    args = [_explicit(a) for a in e.args if isinstance(a, Expr)]
    if len(args) != len(e.args):
        return e
    return as_expr(e._evaluated(tuple(args)))


def _unrolled(f: Expr, limits: list[Limits], is_sum: bool) -> Expr:
    if not limits:
        return _explicit(f)
    v, a, b = limits[-1]
    assert isinstance(a, Integer) and isinstance(b, Integer), (a, b)
    terms: list[Expr] = []
    for i in range(a.p, b.p + 1):
        value = Integer(i)
        inner: list[Limits] = [(w, lo.subs(v, value), hi.subs(v, value)) for w, lo, hi in limits[:-1]]
        terms.append(_unrolled(f.subs(v, value), inner, is_sum))
    return add(*terms) if is_sum else mul(*terms)


def test_orthogonal_polynomials_as_sums() -> None:
    nodes: list[Expr] = [p.Legendre(n, x), p.ChebyshevT(n, x), p.ChebyshevU(n, x), p.Hermite(n, x),
                         p.HermiteProb(n, x), p.Laguerre(n, x), p.AssocLaguerre(n, a, x), p.Jacobi(n, a, b, x),
                         p.Gegenbauer(n, a, x),
                         # SymPy's sum lacks the phase (-1)**m: assoc_legendre(1, 1, x) is
                         # -sqrt(1 - x**2), SymPy's Sum at n = 1 sqrt(1 - x**2)
                         p.AssocLegendre(n, Integer(1), x), p.AssocLegendre(n + 2, Integer(2), x)]
    for node in nodes:
        for target in ('Sum', 'polynomial'):
            rewritten = node.rewrite(target)
            assert rewritten != node, node
            for degree in (1, 2, 4):
                lhs = node.subs(n, Integer(degree)).doit()
                rhs = _explicit(rewritten.subs(n, Integer(degree)))
                assert abs(_value(lhs) - _value(rhs)) < 1e-9, (node, degree)
    # a negative degree of laguerre: exp(x) times the sum at -n - 1
    rewritten = p.Laguerre(Integer(-3), x).rewrite('Sum')
    assert abs(_value(_explicit(rewritten)) - _value(p.laguerre(Integer(-3), x))) < 1e-9
    # SymPy raises on a degree which is not a nonnegative integer
    raises(ValueError, lambda: p.Jacobi(Rational(1, 2), a, b, x).rewrite('Sum'))
    raises(ValueError, lambda: p.Laguerre(Rational(1, 2), x).rewrite('Sum'))


def test_hermite_and_hermite_prob_in_terms_of_each_other() -> None:
    for degree in (0, 3, 5):
        d = Integer(degree)
        h = p.Hermite(d, x).rewrite(p.HermiteProb)
        assert abs(_value(h) - _value(p.hermite(d, x))) < 1e-9
        g = p.HermiteProb(d, x).rewrite(p.Hermite)
        assert abs(_value(g) - _value(p.hermite_prob(d, x))) < 1e-9
