"""The native PiSigma-fields (no SymPy): the shift, the decompositions and
Karr's solver, checked by substituting the solutions back and against
hand computations."""
from __future__ import annotations

from fractions import Fraction

from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras.core import Dummy, Integer, Rational, Symbol
from sympy_extras.functions import binomial, factorial, harmonic
from sympy_extras.polys.domains.sparsefractionfield import SparseFracElement

from sympy_extras.concrete.pisigma import (PiSigmaElement, PiSigmaField, ParametricConstants, RationalConstants,
                                           parametric_pisigma_field, rational_pisigma_field)

k, n = Symbol('k'), Symbol('n')


def _relation_holds(F: PiSigmaField[SparseFracElement[Fraction]], a: PiSigmaElement,
                    c: list[SparseFracElement[Fraction]], fs: list[PiSigmaElement], g: PiSigmaElement) -> bool:
    rhs = F.zero
    for ci, fi in zip(c, fs):
        rhs = rhs + F.from_constant(ci)*fi
    return F.sigma(g) - a*g == rhs


def test_shift_and_decompositions() -> None:
    F = parametric_pisigma_field(k, [n])
    kk = F.gens[0]
    assert F.level == 0 and F.params == (n,)
    assert F.sigma(kk) == kk + 1 and F.sigma(kk, -1) == kk - 1 and F.sigma(kk, 2) == kk + 2
    assert F.sigma(F.from_expr(1/(k + n))) == F.from_expr(1/(k + n + 1))
    F, h = F.add_sigma(1/(k + 1), harmonic(k))
    t = F.from_expr(h)
    kk = F.gens[0]
    assert F.level == 1 and F.level_of(t) == 1 and F.level_of(kk) == 0 and F.level_of(F.from_expr(n)) == -1
    assert F.sigma(t) == t + 1/(kk + 1)
    assert F.sigma(F.sigma(t), -1) == t and F.sigma(t, -1) == t - 1/kk
    assert F.as_poly_in(t**2*kk + 1, 1) == {2: kk, 0: F.one}
    assert F.as_poly_in(1/t, 1) is None
    assert F.to_constant(F.from_expr(n/2)) == F.C.from_expr(n/2)
    raises(ValueError, lambda: F.to_constant(kk))
    F, p = F.add_pi(2, Integer(2)**k)
    u = F.from_expr(p)
    kk, t = F.gens[0], F.from_expr(h)
    assert F.as_poly_in(kk/u, 2, laurent=True) == {-1: kk}
    assert F.as_poly_in(kk/u, 2) is None
    assert F.sigma(u, -1) == u/2 and F.sigma(u*t, 3) == 8*u*F.sigma(t, 3)
    # the conversions
    assert F.to_expr(F.from_expr(h*p/(k + n)), substitute=False) == h*p/(k + n)
    assert F.to_expr(F.from_expr(2*h*p + 2*p)) == 2*Integer(2)**k*(harmonic(k) + 1)
    raises(ValueError, lambda: F.from_expr(Symbol('x')))
    raises(ValueError, lambda: F.from_expr(k**Rational(1, 2)))
    raises(ValueError, lambda: F.from_expr(1/(k - k)))
    raises(TypeError, lambda: untyped(rational_pisigma_field)(2))
    raises(ValueError, lambda: F.add_pi(0, Integer(1)))


def test_fields_are_immutable() -> None:
    F = rational_pisigma_field(k)
    G, h = F.add_sigma(1/(k + 1), harmonic(k))
    assert F.level == 0 and G.level == 1 and isinstance(h, Dummy)
    # elements of different fields do not mix
    raises(ValueError, lambda: F.gens[0] + G.gens[0])
    assert G.to_expr(G.from_expr(h)) == harmonic(k)


def test_generators_keep_their_symbols() -> None:
    # the ring identifies the variables by position: a dummy generator
    # named like the index, or a parameter with assumptions, comes back
    m = Symbol('m', positive=True)
    F = parametric_pisigma_field(k, [m])
    F, t = F.add_sigma(1/(k + 1), harmonic(k), name='k')
    e = F.to_expr(F.from_expr(m*t + k), substitute=False)
    assert e == m*t + k and t in e.free_symbols and m in e.free_symbols
    assert F.to_expr(F.from_expr(m*t + k)) == k + m*harmonic(k)


def test_telescoping() -> None:
    F = rational_pisigma_field(k)
    kk = F.gens[0]
    # sum k = k(k - 1)/2
    g = F.telescope(kk)
    assert g is not None and F.sigma(g) - g == kk
    assert F.to_expr(g) == k*(k - 1)/2
    # 1/k has no rational antidifference
    assert F.telescope(1/kk) is None
    F, h = F.add_sigma(1/(k + 1), harmonic(k))
    t, kk = F.from_expr(h), F.gens[0]
    g = F.telescope(t)
    assert g is not None and F.to_expr(g) == k*(harmonic(k) - 1)
    # H_k/k needs the second order harmonic numbers: not in the field
    assert F.telescope(t/kk) is None
    F, h2 = F.add_sigma(1/(k + 1)**2, harmonic(k, 2))
    t, kk = F.from_expr(h), F.gens[0]
    g = F.telescope(t/kk)
    assert g is not None and F.sigma(g) - g == t/kk
    # sum_{j < k} H_j/j = (H_{k-1}**2 + H_{k-1}^{(2)})/2, up to a constant
    expected = F.from_expr((h**2 - 2*h/k + h2)/2)
    assert F.is_constant(g - expected)
    F, f = F.add_pi(k + 1, factorial(k))
    g = F.telescope(F.from_expr(k*f))
    assert g is not None and F.to_expr(g) == factorial(k)
    assert F.telescope(F.from_expr(f/k)) is None


def test_constants_and_homogeneous_equations() -> None:
    F = rational_pisigma_field(k)
    assert isinstance(F.constants, RationalConstants)
    # sigma(g) - 2 g = c: g = -c
    sols = F.solve(F.from_expr(Integer(2)), [F.one], level=-1)
    assert sols == [([Fraction(1)], -F.one)]
    # sigma(g) - g = c*1 in the constants: c = 0 and g constant
    sols = F.solve(F.one, [F.one], level=-1)
    assert sols is not None and [(c, g) for c, g in sols] == [([Fraction(0)], F.one)]
    assert F.solve(F.from_expr(k + 1), []) == []
    hom = F.solve(F.from_expr((k + 1)/k), [])
    assert hom is not None
    [(c, g)] = hom
    assert c == [] and F.sigma(g)*F.gens[0] == (F.gens[0] + 1)*g


def test_parametric_constants() -> None:
    F = parametric_pisigma_field(k, [n])
    assert isinstance(F.constants, ParametricConstants)
    F, t = F.add_pi((n - k)/(k + 1), binomial(n, k))
    # sigma(g) = (n + 1 - k)/(k + 1) g is solved by binomial(n + 1, k) =
    # (n + 1)/(n + 1 - k) binomial(n, k)
    a = F.from_expr((n + 1 - k)/(k + 1))
    sols = F.solve(a, [])
    assert sols is not None and len(sols) == 1
    [(c, g)] = sols
    assert c == [] and g and F.sigma(g) == a*g
    assert F.level_of(g) == 1 and F.as_poly_in(g, 1, laurent=True) is not None
    # creative telescoping of binomial(n, k): S(n + 1) = 2 S(n)
    fs = [F.from_expr(t), F.from_expr((n + 1)/(n + 1 - k)*t)]
    sols = F.solve(F.one, fs)
    assert sols is not None
    relations = [(c, g) for c, g in sols if any(not F.C.is_zero(ci) for ci in c)]
    assert len(relations) == 1
    [(c, g)] = relations
    assert _relation_holds(F, F.one, c, fs, g)
    assert F.C.exquo(c[1], c[0]) == F.C.from_expr(Rational(-1, 2))
    # the parameterized telescoping of the shifts binomial(n + i, k)**2*H_k
    # (the constants stay of degree 3 in n)
    F, h = F.add_sigma(1/(k + 1), harmonic(k))
    fs = [F.from_expr(t**2*h), F.from_expr((n + 1)**2/(n + 1 - k)**2*t**2*h),
          F.from_expr((n + 1)**2*(n + 2)**2/((n + 1 - k)**2*(n + 2 - k)**2)*t**2*h)]
    sols = F.solve(F.one, fs)
    assert sols is not None
    relations = [(c, g) for c, g in sols if any(not F.C.is_zero(ci) for ci in c)]
    assert len(relations) == 1
    [(c, g)] = relations
    assert _relation_holds(F, F.one, c, fs, g)
    for ci in c:
        assert ci.numer.degree_in(0) <= 3 and ci.denom.degree_in(0) <= 3


def test_dispersion_with_parameters() -> None:
    # the universal denominator: sum 1/((k + n)(k + n + 2)) telescopes to
    # -(2k + 2n + 1)/(2 (k + n)(k + n + 1))
    F = parametric_pisigma_field(k, [n])
    f = F.from_expr(1/((k + n)*(k + n + 2)))
    g = F.telescope(f)
    assert g is not None and F.sigma(g) - g == f
    assert F.is_constant(g - F.from_expr(-(2*k + 2*n + 1)/(2*(k + n)*(k + n + 1))))
    # a shift depending on the parameter is no integer dispersion
    assert F.telescope(F.from_expr(1/((k + n)*(k + 2*n)))) is None
