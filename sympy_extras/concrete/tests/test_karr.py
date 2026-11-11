from __future__ import annotations

from typing import Iterable, Optional

import sympy
from sympy.testing.pytest import raises

from sympy_extras._testing import untyped
from sympy_extras._typing_sympy import as_expr, as_symbol
from sympy_extras.core import Eq, Expr, Integer, Rational, Symbol
from sympy_extras.functions import Binomial, harmonic, factorial, binomial, RisingFactorial, sin, sqrt
from sympy_extras.polys.polytools import Poly, cancel
from sympy_extras.concrete.expr_with_limits import Sum, Product
from sympy_extras.interop import to_sympy

from sympy_extras.concrete.karr import karr_term, karr_sum, summation, build_pisigma_field
from sympy_extras.concrete.pisigma import parametric_pisigma_field
from sympy_extras.concrete.creative import definite_sum
from sympy_extras.concrete.karr_sympy import with_sum_special_values
from sympy_extras.concrete.tests.cases import check_antidifference_cases, check_sum_cases

a, k, n, j, x = Symbol('a'), Symbol('k'), Symbol('n'), Symbol('j'), Symbol('x')
ZERO = Integer(0)


def _check(f: Expr, closed: Optional[Expr], lower: int = 1, values: Iterable[int] = range(1, 7)) -> None:
    """The closed form agrees with the direct sum for several n."""
    assert closed is not None
    for m in values:
        direct = sum((f.subs({k: Integer(i), n: Integer(m)}).doit() for i in range(lower, m + 1)), ZERO)
        assert cancel(closed.subs(n, Integer(m)).doit() - direct) == ZERO, (f, m)


def _sympy(e: Expr) -> sympy.Expr:
    return as_expr(to_sympy(e))


def test_harmonic_sums() -> None:
    s = karr_sum(harmonic(k), (k, 1, n))
    assert s == n*harmonic(n) - n + harmonic(n)
    _check(harmonic(k), s)
    s = karr_sum(k*harmonic(k), (k, 1, n))
    _check(k*harmonic(k), s)
    s = karr_sum(harmonic(k)**2, (k, 1, n))
    assert s == n*harmonic(n)**2 - 2*n*harmonic(n) + 2*n + harmonic(n)**2 - harmonic(n)
    _check(harmonic(k)**2, s)
    s = karr_sum(k**2*harmonic(k), (k, 1, n))
    _check(k**2*harmonic(k), s)
    s = karr_sum(harmonic(k, 2), (k, 1, n))
    _check(harmonic(k, 2), s)
    s = karr_sum(harmonic(k)*harmonic(k, 2), (k, 1, n))
    _check(harmonic(k)*harmonic(k, 2), s)
    # sum(H_k**2/k) needs the nested sum of H_j**(2)/j, which is not a
    # harmonic number: correctly no closed form in harmonic numbers
    assert karr_sum(harmonic(k)**2/k, (k, 1, n)) is None
    # shifted arguments
    s = karr_sum(harmonic(k + 1), (k, 0, n))
    _check(harmonic(k + 1), s, lower=0)
    s = karr_sum(harmonic(k - 1)/k, (k, 2, n))
    _check(harmonic(k - 1)/k, s, lower=2, values=range(2, 8))
    s = karr_sum(harmonic(k)/(k + 1), (k, 1, n))
    assert s is not None
    assert cancel(s - (harmonic(n + 1)**2 - harmonic(n + 1, 2))/2).subs(n, Integer(5)).doit() == ZERO
    _check(harmonic(k)/(k + 1), s)


def test_no_closed_form() -> None:
    # provably no closed form in the field built from the summand
    assert karr_sum(harmonic(k)/k, (k, 1, n), auto=False) is None
    assert karr_term(harmonic(k, 2), k, auto=False) is None
    assert karr_sum(2**k/k, (k, 1, n)) is None
    assert karr_sum(factorial(k)/k, (k, 1, n)) is None
    assert karr_sum(binomial(n, k), (k, 0, n)) is None
    assert karr_sum(harmonic(k)*2**k, (k, 1, n)) is None
    # sum 1/k has no closed form in the rational functions, and is
    # harmonic(n) once the harmonic extension is adjoined (automatic
    # since sympy-extras#34)
    assert karr_sum(1/k, (k, 1, n), auto=False) is None
    assert karr_sum(1/k, (k, 1, n)) == harmonic(n)
    # 1/k**2 has no antidifference in the rational functions; with the
    # harmonic extension adjoined automatically (sympy-extras#34) it has
    # one, which must telescope to the summand
    assert karr_term(1/k**2, k, auto=False) is None
    g = karr_term(1/k**2, k)
    assert g is not None
    # harmonic(k + 1, 2) - harmonic(k, 2) is 1/(k + 1)**2, which simplify
    # cannot see symbolically but which is exact at integers
    assert all(g.subs(k, Integer(i + 1)) - g.subs(k, Integer(i)) - Rational(1, i**2) == ZERO for i in range(1, 7))
    # with the right extension there is one
    s = karr_sum(harmonic(k)/k, (k, 1, n), extensions=[harmonic(k, 2)])
    assert s is not None
    assert s.expand() == \
        harmonic(n)**2/2 + harmonic(n, 2)/2


def test_hypergeometric_and_rational() -> None:
    assert karr_sum(k*factorial(k), (k, 1, n)) == n*factorial(n) + factorial(n) - 1
    s2 = karr_sum(k**2*2**k, (k, 0, n))
    assert s2 is not None
    assert s2.expand() == 2*2**n*n**2 - 4*2**n*n + 6*2**n - 6
    assert karr_sum(1/(k*(k + 1)), (k, 1, n)) == n/(n + 1)
    assert karr_sum(k**2, (k, 1, n)) == n*(2*n**2 + 3*n + 1)/6
    assert karr_sum(binomial(2*k, k)/4**k, (k, 0, n)) == (2*n + 1)*binomial(2*n, n)/4**n
    s = karr_sum(k*binomial(n, k)/(n - k + 1), (k, 0, n))
    if s is not None:
        _check(k*binomial(n, k)/(n - k + 1), s, lower=0)
    assert karr_sum(RisingFactorial(x, k), (k, 0, n)) is None
    s = karr_sum(RisingFactorial(x, k)/factorial(k), (k, 0, n))
    _check(RisingFactorial(x, k)/factorial(k), s, lower=0)
    assert karr_sum(2**(-k), (k, 0, n)) == (2*2**n - 1)/2**n
    s = karr_sum(k*3**k, (k, 1, n))
    _check(k*3**k, s)
    # mixed products and sums
    s = karr_sum(harmonic(k)/2**k, (k, 1, n))
    assert s is None
    # sum(k*k!*H_k) differs from a telescoping term by sum(k!), which has
    # no closed form
    assert karr_sum(harmonic(k)*k*factorial(k), (k, 1, n)) is None
    s = karr_sum(harmonic(k)*k*factorial(k) + factorial(k), (k, 1, n))
    _check(harmonic(k)*k*factorial(k) + factorial(k), s)
    s = karr_sum(2**k*(harmonic(k) - 2/(k + 1)), (k, 1, n))
    if s is not None:
        _check(2**k*(harmonic(k) - 2/(k + 1)), s)
    # parameters
    s = karr_sum(n*k + 1/(k*(k + 1)), (k, 1, n))
    _check(n*k + 1/(k*(k + 1)), s)
    s = karr_sum(x**k, (k, 0, n))
    assert s is not None
    assert all(s.subs({x: Integer(2), n: Integer(m)}) == Integer(2**(m + 1) - 1) for m in range(0, 5))
    # the value at x = 1, where the closed form is 0/0, as a case
    assert s.subs(x, 1) == n + 1


def test_isolated_values_of_the_parameters() -> None:
    # the bug: the sum of y**k was (y*y**n - 1)/(y - 1), which is 0/0 at
    # y = 1, where the sum is n + 1
    y = Symbol('y')
    for s in (karr_sum(y**k, (k, 0, n)), summation(y**k, (k, 0, n))):
        assert s is not None
        assert s.subs({y: Integer(1), n: Integer(5)}) == Integer(6)
        assert s.subs({y: Integer(2), n: Integer(5)}) == Integer(63)
    creative = definite_sum(y**k, (k, 0, n))
    assert creative is not None
    assert creative.subs({y: Integer(1), n: Integer(5)}) == Integer(6)
    assert creative.subs({y: Integer(2), n: Integer(5)}) == Integer(63)
    s = summation(k*y**k, (k, 0, n))
    assert s.subs({y: Integer(1), n: Integer(4)}) == Integer(10)
    # a point where the sum itself is undefined (the term 1/0 at k = 0 for
    # a = 0) gets no case
    assert not summation(1/((k + a)*(k + a + 1)), (k, 0, n)).has(Eq)


def test_isolated_values_of_indefinite_sums() -> None:
    # the bug: the antidifference of y**k was y**k/(y - 1), undefined at
    # y = 1, where it is k
    y = Symbol('y')
    for value in (karr_term(y**k, k), karr_sum(y**k, k)):
        assert value is not None
        assert check_antidifference_cases(_sympy(value), _sympy(y**k), as_symbol(to_sympy(k))) == 1
        assert value.subs(y, Integer(1)) == k
    assert karr_term(y**k, k, special_values=False) == y**k/(y - 1)
    assert karr_sum(y**k, (k, 0, n), special_values=False) == (y*y**n - 1)/(y - 1)


def test_zero_powers_at_a_point_are_summed() -> None:
    # the bug: the sum of binomial(n, k)*y**k/(k + 1), ((y + 1)**(n + 1) -
    # 1)/(y*(n + 1)), got no case at y = 0 (the summand 0**k*... at the
    # point, which the algorithms do not take), where the sum is 1
    y = Symbol('y')
    term = binomial(n, k)*y**k/(k + 1)
    value = definite_sum(term, (k, 0, n))
    assert value is not None and value.subs(y, Integer(0)) == Integer(1)
    ns = as_symbol(to_sympy(n))
    assert check_sum_cases(_sympy(value), _sympy(term), as_symbol(to_sympy(k)), 0, ns, ns) == 1


def test_wrong_cases_are_left_out() -> None:
    # a case which disagrees with the sum computed term by term (here a
    # wrong value given by the callback) is not listed
    y, ks, ns = sympy.symbols('y k n')
    generic = (y**(ns + 1) - 1)/(y - 1)
    assert with_sum_special_values(generic, y**ks, [ks], [sympy.S.Zero, ns], lambda point, at: ns + 2) == generic
    assert with_sum_special_values(y**ks/(y - 1), y**ks, [ks], [], lambda point, at: 2*ks) == y**ks/(y - 1)
    right = with_sum_special_values(generic, y**ks, [ks], [sympy.S.Zero, ns], lambda point, at: ns + 1)
    assert check_sum_cases(right, y**ks, ks, 0, ns, ns) == 1


def test_nested_sums_and_products() -> None:
    inner = Sum(1/j**2, (j, 1, k))
    s = karr_sum(inner, (k, 1, n))
    _check(inner, s)
    s = karr_sum(Sum(harmonic(j), (j, 1, k)), (k, 1, n))
    _check(Sum(harmonic(j), (j, 1, k)), s)
    # a nested sum which is a known sequence is recognised
    F, f = build_pisigma_field(Sum(1/j, (j, 1, k)) + harmonic(k), k)
    assert len(F.extensions) == 1 and F.to_expr(f) == 2*harmonic(k) or F.to_expr(f) == 2*Sum(1/j, (j, 1, k))
    p = Product(2*j, (j, 1, k))
    s = karr_sum((2*k + 1)*p, (k, 1, n))
    _check((2*k + 1)*p, s)
    assert karr_sum(k*p, (k, 1, n)) is None


def test_karr_term_and_field() -> None:
    assert karr_term(harmonic(k), k) == k*(harmonic(k) - 1)
    assert karr_term(k*factorial(k), k) == factorial(k)
    g = karr_term(k, k)
    assert g is not None
    assert (g.subs(k, k + 1) - g).expand() == k
    assert karr_sum(harmonic(k), k) == karr_term(harmonic(k), k)
    F, f = build_pisigma_field(harmonic(k + 1)*factorial(k), k)
    assert [e.kind for e in F.extensions] == ['pi', 'sigma']
    assert F.to_expr(f) == (k*harmonic(k) + harmonic(k) + 1)*factorial(k)/(k + 1)
    # factorial(k + 1) is recognised as (k + 1)*factorial(k)
    F, f = build_pisigma_field(factorial(k + 1) + factorial(k), k)
    assert len(F.extensions) == 1
    assert F.to_expr(f) == (k + 2)*factorial(k)
    F, f = build_pisigma_field(harmonic(k + 2) - harmonic(k), k)
    assert len(F.extensions) == 1 and F.to_expr(f) == (2*k + 3)/(k**2 + 3*k + 2)
    F, f = build_pisigma_field(2**k + 4**k, k)
    assert len(F.extensions) == 1 and F.to_expr(f) == 2**k*(2**k + 1)
    raises(ValueError, lambda: build_pisigma_field(sin(k), k))
    raises(ValueError, lambda: build_pisigma_field(sqrt(k), k))
    raises(TypeError, lambda: untyped(build_pisigma_field)(k, 2))


def test_pisigma_field() -> None:
    # the native field with a parameter, as creative telescoping uses it
    k, n = Symbol('k'), Symbol('n')
    F = parametric_pisigma_field(k, [n])
    assert F.level == 0 and F.params == (n,)
    kk = F.gens[0]
    assert F.sigma(kk) == kk + 1 and F.sigma(kk, -1) == kk - 1 and F.sigma(kk, 2) == kk + 2
    F, h = F.add_sigma(1/(k + 1), harmonic(k))
    t = F.from_expr(h)
    kk = F.gens[0]
    assert F.level == 1 and F.level_of(t) == 1 and F.level_of(kk) == 0 and F.level_of(F.from_expr(n)) == -1
    assert F.sigma(F.sigma(t), -1) == t
    assert F.as_poly_in(t**2*kk + 1, 1) == {2: kk, 0: F.one}
    assert F.as_poly_in(1/t, 1) is None
    assert F.constant_to_expr(F.to_constant(F.from_expr(n/2))) == n/2
    raises(ValueError, lambda: F.to_constant(kk))
    F, p = F.add_pi(2, 2**k)
    u = F.from_expr(p)
    kk, t = F.gens[0], F.from_expr(h)
    assert F.as_poly_in(kk/u, 2, laurent=True) == {-1: kk}
    assert F.as_poly_in(kk/u, 2) is None
    assert F.sigma(u, -1) == u/2
    # the parameterized equation: sigma(g) - g = c1*H + c2*(k*H) has a two
    # dimensional solution space plus the constants
    sols = F.solve_embedded(F.one, [t, kk*t])
    assert sols is not None
    assert len(sols) == 3
    for c, g in sols:
        assert F.sigma(g) - g == c[0]*t + c[1]*kk*t
    # homogeneous equations
    assert F.solve_embedded(F.from_expr(k + 1), []) == []
    hom = F.solve_embedded(F.from_expr((k + 1)/k), [])
    assert hom is not None
    [(c, g)] = hom
    assert c == [] and F.sigma(g)*kk == (kk + 1)*g
    raises(ValueError, lambda: F.add_pi(0, Integer(1)))
    raises(TypeError, lambda: untyped(parametric_pisigma_field)(2, [n]))
    assert repr(F.extensions[0]) == "Extension(sigma, harmonic(k), 1/(k + 1))"
    # the numerator and the denominator, and the sign of the denominator
    # SymPy's fraction field gives
    q = F.from_expr(k/(n + 1 - k))
    assert cancel(F.numer_expr(q)/F.denom_expr(q) - k/(n + 1 - k)) == ZERO
    assert str(F.to_expr(q, positive_denominator=True)) == "-k/(k - n - 1)"
    assert str(F.to_expr(q)) == "k/(-k + n + 1)"


def test_summation() -> None:
    assert summation(k**2, (k, 1, n)) == n**3/3 + n**2/2 + n/6
    assert summation(harmonic(k), (k, 1, n)) == n*harmonic(n) - n + harmonic(n)
    assert summation(harmonic(k)/k, (k, 1, n)).expand() == harmonic(n)**2/2 + harmonic(n, 2)/2
    assert summation(2**k/k, (k, 1, n)) == Sum(2**k/k, (k, 1, n))
    assert summation(sin(k), (k, 1, n)).has(Sum)
    assert summation(harmonic(k), (k, 1, 3)) == Rational(13, 3)
    assert summation(Sum(1/j**2, (j, 1, k)), (k, 1, n)) == n*harmonic(n, 2) - harmonic(n) + harmonic(n, 2)


def test_harmonic_sum_gets_its_extension_automatically() -> None:
    # sympy-extras#34: the harmonic extension only fired when the summand
    # already carried a harmonic number, so sum 1/k came back None.
    n, k = Symbol('n', integer=True), Symbol('k', integer=True)
    result = karr_sum(1/k, (k, 1, n))
    assert result is not None
    for N in range(1, 7):
        assert result.subs(n, Integer(N)) == harmonic(Integer(N)), N


def test_closed_form_is_not_asserted_at_its_own_poles() -> None:
    # sympy-extras#35: the antidifference has a denominator in n whose
    # integer roots are exactly where the returned closed form 0 is false:
    # the sum is -1 at n = 1 and 2 at n = 2.
    n, k = Symbol('n', integer=True), Symbol('k', integer=True)
    term = (-1)**k*k**2*binomial(n, k)
    result = karr_sum(term, (k, 1, n))
    assert result is not None
    for N in range(1, 7):
        assert result.subs(n, Integer(N)) == Sum(term.subs(n, Integer(N)), (k, 1, N)).doit(), N


def test_homogeneous_pi_solutions_of_a_single_exponent_are_found() -> None:
    # sigma(g) = a*g in a Pi-extension t is solved by c*t**j for one
    # exponent j; the recursion over the exponents stopped at the first
    # exponent without a solution and returned [], so binomial(n + 1, k)
    # was not recognised as binomial(n, k)*(n + 1)/(n + 1 - k) and became
    # a second, algebraically dependent Pi-extension.
    k, n = Symbol('k'), Symbol('n')
    F, _ = parametric_pisigma_field(k, [n]).add_pi((n - k)/(k + 1), binomial(n, k))
    sols = F.solve_embedded(F.from_expr((n + 1 - k)/(k + 1)), [])
    assert sols is not None and len(sols) == 1
    [(c, g)] = sols
    assert c == [] and g != F.zero
    assert F.sigma(g) == F.from_expr((n + 1 - k)/(k + 1))*g
    assert F.level_of(g) == 1 and F.as_poly_in(g, 1, laurent=True) is not None
    # the same through the native field of build_pisigma_field
    k_, n_ = Symbol('k'), Symbol('n')
    field, element = build_pisigma_field(Binomial(n_ + 1, k_)*Binomial(n_, k_), k_)
    assert len(field.extensions) == 1
    assert field.to_expr(element).has(*[e.symbol for e in field.extensions]) is False


def test_parameterized_solutions_have_small_constants() -> None:
    # the bases of the successive steps of the recursion were composed
    # without normalisation: the parameterized telescoping of the shifts
    # binomial(n + i, k)**2*harmonic(k) gave constants of degree 530 in n
    # (and took two minutes) for a relation whose coefficients have degree 3.
    k, n = Symbol('k'), Symbol('n')
    F, t = parametric_pisigma_field(k, [n]).add_pi((n - k)/(k + 1), binomial(n, k))
    F, h = F.add_sigma(1/(k + 1), harmonic(k))
    fs = [F.from_expr(t**2*h), F.from_expr((n + 1)**2/(n + 1 - k)**2*t**2*h),
          F.from_expr((n + 1)**2*(n + 2)**2/((n + 1 - k)**2*(n + 2 - k)**2)*t**2*h)]
    sols = F.solve_embedded(F.one, fs)
    assert sols is not None
    relations = [(c, g) for c, g in sols if any(c)]
    assert len(relations) == 1
    [(c, g)] = relations
    # the numerator and the denominator of an element over Q(n) are not
    # normalised: the difference is compared with zero
    assert F.sigma(g) - g - sum((ci*fi for ci, fi in zip(c, fs)), F.zero) == F.zero
    for ci in c:
        assert F.level_of(ci) == -1
        numerator, denominator = F.numer_expr(ci), F.denom_expr(ci)
        assert Poly(numerator, n).degree() <= 3 and Poly(denominator, n).degree() <= 3
