"""Series involving polygamma functions and harmonic numbers (Euler
sums), summed in closed form through zeta values.

The polygamma functions at integers are harmonic numbers,

.. math::

    \\psi(n + 1) = H_n - \\gamma, \\qquad
    \\psi^{(m)}(n + 1) = (-1)^{m+1} m! \\left(\\zeta(m + 1) - H_n^{(m+1)}\\right),

with `H_n^{(p)} = \\sum_{k \\le n} k^{-p}`, so that a series
`\\sum_n \\psi^{(m)}(n + a)\\, n^{-q}` is a combination of zeta values and
of the **Euler sums** `S(p, q) = \\sum_{n \\ge 1} H_n^{(p)} n^{-q}`
(convergent for `q \\ge 2`). The Euler sums are evaluated by

* Euler's formula for `p = 1`:
  `S(1, q) = (1 + q/2)\\,\\zeta(q + 1) - \\tfrac12 \\sum_{k=1}^{q-2} \\zeta(k + 1)\\zeta(q - k)`;
* the symmetry `S(p, q) + S(q, p) = \\zeta(p)\\zeta(q) + \\zeta(p + q)`, which
  gives `S(p, p)`;
* the formula of Borwein, Borwein and Girgensohn (in the form of
  Flajolet and Salvy) for an odd weight `p + q`;
* the known evaluations of weight six, `S(2, 4) = \\zeta(3)^2 - \\zeta(6)/3`.

Sums of even weight at least eight with `p \\ne q`, `p, q \\ge 2`, are not
all expressible through zeta values and are left alone (``None``).

The **integral representation** `\\psi(n) = -\\gamma + \\int_0^1 (1 - t^{n-1})/(1 - t)\\, dt`
turns `\\sum_n \\psi(n) n^{-s}` for a symbolic `s` into
`-\\gamma \\zeta(s) + \\int_0^1 (\\zeta(s) - \\mathrm{Li}_s(t)/t)/(1 - t)\\, dt`
(the way Mathematica's notes describe these series); this form is
returned by :func:`polygamma_integral_representation`.

References
==========

.. [Flajolet] P. Flajolet, B. Salvy, Euler sums and contour integral
   representations, Experimental Mathematics 7 (1998), Theorems 2.2 and
   3.1.
.. [Borwein] D. Borwein, J. M. Borwein, R. Girgensohn, Explicit
   evaluation of Euler sums, Proceedings of the Edinburgh Mathematical
   Society 38 (1995).
"""
from __future__ import annotations

from typing import Optional

from sympy_extras.core.add import Add, add
from sympy_extras.core.expr import Expr, ExprLike, to_expr
from sympy_extras.core.mul import Mul, mul
from sympy_extras.core.numbers import EulerGamma, Integer, Rational
from sympy_extras.core.power import Pow
from sympy_extras.core.symbol import Dummy, Symbol
from sympy_extras.functions.combinatorial.factorials import binomial, factorial
from sympy_extras.functions.combinatorial.numbers import Harmonic, harmonic
from sympy_extras.functions.special.gamma_functions import PolyGamma
from sympy_extras.functions.special.zeta_functions import polylog, zeta
from sympy_extras.polys.partfrac import apart

__all__ = ['euler_sum', 'polygamma_series', 'polygamma_integral_representation']

ZERO, ONE = Integer(0), Integer(1)


def _zeta(k: int) -> Expr:
    return zeta(Integer(k))


def euler_sum(p: int, q: int) -> Optional[Expr]:
    """The Euler sum ``S(p, q) = Sum(harmonic(n, p)/n**q, (n, 1, oo))`` in
    zeta values, or ``None`` when no evaluation is known.

    Examples
    ========

    >>> from sympy_extras.concrete.eulersums import euler_sum
    >>> euler_sum(1, 2)
    2*zeta(3)
    >>> euler_sum(1, 3)
    pi**4/72
    >>> euler_sum(2, 4)
    -pi**6/2835 + zeta(3)**2
    >>> euler_sum(2, 3)
    -9*zeta(5)/2 + pi**2*zeta(3)/2
    >>> euler_sum(2, 6) is None
    True
    """
    if p < 1 or q < 2:
        return None
    if p == 1:
        total = (1 + Rational(q, 2)) * _zeta(q + 1)
        total = total - Rational(1, 2) * add(*[_zeta(k + 1) * _zeta(q - k) for k in range(1, q - 1)])
        return total.expand()
    if p == q:
        return ((_zeta(p)**2 + _zeta(2 * p)) / 2).expand()
    w = p + q
    if w % 2 == 1:
        return _odd_weight(p, q)
    if q == 1:
        return None
    if (p, q) == (2, 4):
        return (_zeta(3)**2 - _zeta(6) / 3).expand()
    if (p, q) == (4, 2):
        # from the symmetry with S(2, 4)
        return (_zeta(2) * _zeta(4) + _zeta(6) - (_zeta(3)**2 - _zeta(6) / 3)).expand()
    return None


def _odd_weight(p: int, q: int) -> Expr:
    """Flajolet–Salvy, Theorem 3.1: for an odd weight ``w = p + q``,
    ``S(p, q)`` is a rational combination of ``zeta(w)`` and of the
    products ``zeta(2k)*zeta(w - 2k)`` (and ``zeta(p)*zeta(q)`` for an odd
    ``p``)."""
    w = p + q
    sign = (-1)**p
    total: Expr = _zeta(w) * (Rational(1, 2) - Rational(sign, 2) * binomial(Integer(w - 1), Integer(p))
                              - Rational(sign, 2) * binomial(Integer(w - 1), Integer(q)))
    if p % 2 == 1 and p > 1:
        total = total + _zeta(p) * _zeta(q)
    for k in range(1, q // 2 + 1):
        total = total + sign * binomial(Integer(w - 2 * k - 1), Integer(p - 1)) * _zeta(2 * k) * _zeta(w - 2 * k)
    for k in range(1, p // 2 + 1):
        total = total + sign * binomial(Integer(w - 2 * k - 1), Integer(q - 1)) * _zeta(2 * k) * _zeta(w - 2 * k)
    return total.expand()


def _integer(e: Expr) -> Optional[int]:
    return int(e) if isinstance(e, Integer) else None


def _arg(f: Expr, i: int) -> Expr:
    a = f.args[i]
    assert isinstance(a, Expr)
    return a


def _split(term: Expr, n: Symbol) -> Optional[tuple[Expr, Expr, int]]:
    """``(c, f, q)`` with ``term == c * f * n**(-q)``, ``c`` free of ``n``
    and ``f`` a single harmonic number or polygamma factor."""
    factors = [a for a in term.args if isinstance(a, Expr)] if isinstance(term, Mul) else [term]
    constant: list[Expr] = []
    special: list[Expr] = []
    q = 0
    for f in factors:
        if not f.has(n):
            constant.append(f)
        elif isinstance(f, Pow) and f.base == n and isinstance(f.exp, Integer):
            q -= int(f.exp)
        elif f == n:
            q -= 1
        else:
            special.append(f)
    if len(special) != 1:
        return None
    return mul(*constant), special[0], q


def _harmonic_data(f: Expr, n: Symbol) -> Optional[tuple[int, int, Expr]]:
    """``(p, shift, extra)`` with ``f == harmonic(n + shift, p) + extra``
    for a harmonic number, and ``polygamma(m, n + a)`` written as
    ``(-1)**(m+1) m! (zeta(m+1) - harmonic(n + a - 1, m + 1))``."""
    if isinstance(f, Harmonic):
        argument = _arg(f, 0)
        p = _integer(_arg(f, 1)) if len(f.args) > 1 else 1
        shift = _integer(argument - n)
        if p is None or shift is None:
            return None
        return p, shift, ZERO
    if isinstance(f, PolyGamma):
        m = _integer(_arg(f, 0))
        shift = _integer(_arg(f, 1) - n)
        if m is None or shift is None or m < 0:
            return None
        return m + 1, shift - 1, ZERO
    return None


def _to_harmonic(f: Expr, n: Symbol) -> Optional[tuple[Expr, int, int, Expr]]:
    """``f`` as ``scale * harmonic(n + shift, p) + constant``."""
    data = _harmonic_data(f, n)
    if data is None:
        return None
    p, shift, _ = data
    if isinstance(f, PolyGamma):
        m = p - 1
        if m == 0:
            # psi(n + a) = H_{n + a - 1} - gamma
            return ONE, shift, 1, -EulerGamma
        scale = Integer(-1)**m * factorial(Integer(m))
        return scale, shift, p, -scale * zeta(Integer(p))
    return ONE, shift, p, ZERO


def _rational_tail(expr: Expr, k: Symbol, lower: int) -> Optional[Expr]:
    """``Sum(expr, (k, lower, oo))`` for a rational function whose poles
    are at integers, through partial fractions: ``Sum(1/(k + a)**m)`` is
    ``zeta(m) - harmonic(lower + a - 1, m)`` and the simple poles, whose
    coefficients add up to zero, give ``-Sum(c_a harmonic(lower + a - 1))``
    (SymPy's ``summation`` gets some of these sums wrong)."""
    decomposed = apart(expr, k)
    terms = [a for a in decomposed.args if isinstance(a, Expr)] if isinstance(decomposed, Add) else [decomposed]
    total: Expr = ZERO
    simple: Expr = ZERO
    for term in terms:
        c, rest = term.as_independent(k)
        if not rest.has(k):
            return None
        if isinstance(rest, Pow) and isinstance(rest.exp, Integer) and int(rest.exp) < 0:
            base, m = rest.base, -int(rest.exp)
        else:
            base, m = rest, 1
            inverse = 1 / rest
            if inverse.is_polynomial(k) is True:
                base = inverse
            else:
                return None
        a = _integer(base - k)
        if a is None or lower + a < 1:
            return None
        if m == 1:
            simple = simple + c
            total = total - c * harmonic(Integer(lower + a - 1))
        else:
            total = total + c * (zeta(Integer(m)) - harmonic(Integer(lower + a - 1), Integer(m)))
    if not (simple.is_Number and simple == 0):
        return None
    return total.expand()


def polygamma_series(term: ExprLike, n: Symbol, lower: ExprLike = 1) -> Optional[Expr]:
    """``Sum(term, (n, lower, oo))`` for a summand ``c * f(n) / n**q`` with
    ``f`` a harmonic number ``harmonic(n + a, p)`` or a polygamma function
    ``polygamma(m, n + a)`` (integer ``a``), in zeta values; ``None`` when
    the summand is not of this form or the Euler sum is not known.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import harmonic, polygamma
    >>> from sympy_extras.concrete.eulersums import polygamma_series
    >>> n = Symbol('n')
    >>> polygamma_series(harmonic(n)/n**2, n)
    2*zeta(3)
    >>> polygamma_series(polygamma(0, n)/n**2, n)
    -EulerGamma*pi**2/6 + zeta(3)
    >>> polygamma_series(polygamma(1, n)/n**2, n)
    7*pi**4/360
    >>> polygamma_series(harmonic(n, 2)/n**2, n)
    7*pi**4/360
    >>> polygamma_series(harmonic(n)/n**2, n, 2)
    -1 + 2*zeta(3)
    """
    term_ = to_expr(term)
    lower_ = to_expr(lower)
    if not isinstance(lower_, Integer) or int(lower_) < 1:
        return None
    L = int(lower_)
    split = _split(term_, n)
    if split is None:
        return None
    c, f, q = split
    converted = _to_harmonic(f, n)
    if converted is None or q < 2:
        return None
    scale, shift, p, constant = converted
    if L + shift < 0:
        # a harmonic number with a negative index (a pole of polygamma)
        return None
    base = euler_sum(p, q)
    if base is None:
        return None
    k = Dummy('k')
    # sum_{n >= L} H_n^{(p)} / n**q
    total: Expr = base - add(*[harmonic(Integer(j), Integer(p)) / Integer(j)**q for j in range(1, L)])
    # H_{n + shift} = H_n + sum_{j=1}^{shift} 1/(n + j)**p, or
    # H_n - sum_{j=0}^{-shift-1} 1/(n - j)**p
    for j in (range(1, shift + 1) if shift > 0 else range(0, -shift)):
        offset = j if shift > 0 else -j
        extra = _rational_tail(1 / ((k + offset)**p * k**q), k, L)
        if extra is None:
            return None
        total = total + extra if shift > 0 else total - extra
    total = scale * total + constant * (zeta(Integer(q)) - add(*[Integer(j)**(-q) for j in range(1, L)]))
    return (c * total).expand()


def polygamma_integral_representation(term: ExprLike, n: Symbol) -> Optional[Expr]:
    """``Sum(polygamma(0, n)/n**s, (n, 1, oo))`` as an integral through
    ``psi(n) = -EulerGamma + Integral((1 - t**(n - 1))/(1 - t), (t, 0, 1))``,
    for a symbolic exponent ``s`` (the integral is SymPy's ``Integral``,
    wrapped: it has no class here yet).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.functions import polygamma
    >>> from sympy_extras.concrete.eulersums import polygamma_integral_representation
    >>> n, s = Symbol('n'), Symbol('s')
    >>> polygamma_integral_representation(polygamma(0, n)/n**s, n)
    Integral((zeta(s) - polylog(s, t)/t)/(1 - t), (t, 0, 1)) - EulerGamma*zeta(s)

    (the wrapped SymPy node is printed first: SymPy orders it after)
    """
    import sympy
    from sympy_extras.interop import from_sympy, sympy_expr
    from sympy_extras.interop import to_sympy
    term_ = to_expr(term)
    factors = [a for a in term_.args if isinstance(a, Expr)] if isinstance(term_, Mul) else [term_]
    exponent: Optional[Expr] = None
    found = False
    constant: list[Expr] = []
    for f in factors:
        if isinstance(f, PolyGamma) and _arg(f, 0) == 0 and _arg(f, 1) == n:
            found = True
        elif isinstance(f, Pow) and f.base == n and not f.exp.has(n):
            exponent = -f.exp
        elif not f.has(n):
            constant.append(f)
        else:
            return None
    if not found or exponent is None:
        return None
    t = Symbol('t')
    s = exponent
    integrand = (zeta(s) - polylog(s, t) / t) / (1 - t)
    integral = from_sympy(sympy.Integral(sympy_expr(to_sympy(integrand)), (sympy_expr(to_sympy(t)), 0, 1)))
    assert isinstance(integral, Expr)
    return mul(*constant) * (-EulerGamma * zeta(s) + integral)
