"""The modulo operation: the class ``Mod`` and the function ``mod``.

``Mod(p, q)`` is an unevaluated node; ``mod(p, q)`` applies SymPy's rules
[1]_: numbers are reduced (with the sign of ``q``, as Python's ``%``),
``Mod(p, p)`` and ``Mod(0, q)`` are ``0``, an integer modulo ``1`` is
``0``, a common factor of ``p`` and ``q`` is taken out
(``Mod(6*x, 3)`` is ``3*Mod(2*x, 1)``, ``Mod(2*n, 2)`` is ``0``), the
integer and rational terms of ``p`` are reduced modulo an integer or
rational ``q`` (``Mod(x + 3, 3)`` is ``Mod(x, 3)``), a negative integer
coefficient of an integer ``p`` is reduced modulo an integer ``q``, and
nested moduli collapse when one modulus divides the other. The operator
``%`` of ``Expr`` calls ``mod``.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Mod*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.mod.Mod
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

from .add import Add, add
from .basic import Basic
from .expr import Expr, ExprLike, factors_of, terms_of, to_expr
from .function import Application
from .mul import mul
from .numbers import Float, Integer, Number, Rational, nan, number_from_fraction
from .parameters import global_parameters
from .power import pow
from .sorting import ClassKey

__all__ = ['Mod', 'mod']


class Mod(Application):
    """The unevaluated modulo ``Mod(p, q)``.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Mod
    >>> x = Symbol('x')
    >>> Mod(7, 3), Mod(7, 3).doit(), Mod(x, 3), x % 3, (2*x) % 2
    (Mod(7, 3), 1, Mod(x, 3), Mod(x, 3), 2*(Mod(x, 1)))
    """

    __slots__ = ()

    def __init__(self, p: ExprLike, q: ExprLike) -> None:
        Application.__init__(self, p, q)

    @property
    def name(self) -> str:
        return 'Mod'

    @classmethod
    def class_key(cls) -> ClassKey:
        return (4, 10000, 'Mod')

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        p, q = args
        if not (isinstance(p, Expr) and isinstance(q, Expr)):
            raise TypeError("Mod takes two expressions")
        return mod(p, q)

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import facts_of
        p, q = facts_of(self.args[0]), facts_of(self.args[1])
        facts: dict[str, bool] = {}
        if p.get('commutative') is True and q.get('commutative') is True:
            facts['commutative'] = True
        if p.get('integer') is True and q.get('integer') is True:
            facts['integer'] = True
        if q.get('real') is True:
            facts['real'] = True
        if q.get('positive') is True:
            facts['nonnegative'] = True
        if q.get('negative') is True:
            facts['nonpositive'] = True
        if p.get('finite') is True and q.get('finite') is True:
            facts['finite'] = True
        return facts


def mod(p: ExprLike, q: ExprLike) -> Expr:
    """The evaluated ``p`` modulo ``q``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, mod, Rational
    >>> x = symbols('x')
    >>> n = symbols('n', integer=True)
    >>> mod(7, 3), mod(-7, 3), mod(7, -3), mod(Rational(7, 2), 2)
    (1, 2, -2, 3/2)
    >>> mod(x, x), mod(n, 1), mod(2*n, 2), mod(x + 3, 3), mod(6*x, 3), mod(-n, 3)
    (0, 0, 0, Mod(x, 3), 3*(Mod(2*x, 1)), Mod(2*n, 3))
    """
    a, b = to_expr(p), to_expr(q)
    if not global_parameters.evaluate:
        return Mod(a, b)
    if isinstance(b, Number) and b.exact_value() == 0:
        raise ZeroDivisionError("Modulo by zero")
    if isinstance(a, Number) and isinstance(b, Number):
        return _number_mod(a, b)
    if (isinstance(a, Number) and a.exact_value() is None) or (isinstance(b, Number) and b.exact_value() is None):
        return nan
    if a == b:
        return Integer(0)
    if isinstance(a, Number) and a.exact_value() == 0:
        return Integer(0)
    if isinstance(b, Rational) and b.p == 1 and b.q == 1 and a.is_integer is True:
        return Integer(0)
    if isinstance(a, Mod):
        inner_p, inner_q = a.args
        if isinstance(inner_q, Rational) and isinstance(b, Rational):
            ratio = inner_q.as_fraction() / b.as_fraction()
            if ratio.denominator == 1:
                return mod(inner_p, b)
            if (1 / ratio).denominator == 1:
                return a
    common = _common_factor(a, b)
    if common is not None:
        return mul(common, mod(_divide(a, common), _divide(b, common)))
    reduced = _reduce_terms(a, b)
    if reduced is not None:
        return reduced
    if isinstance(b, Rational) and b.p == 2 and b.q == 1 and _has_consecutive_factors(a):
        return Integer(0)
    return Mod(a, b)


def _has_consecutive_factors(a: Expr) -> bool:
    """Whether ``a`` is a product of integers with two factors which differ
    by an odd integer (``n*(n + 1)`` is even), the rule SymPy applies for
    the modulus ``2``."""
    from .add import sub
    if not a.is_Mul:
        return False
    factors = factors_of(a)
    if any(f.is_integer is not True for f in factors):
        return False
    for i, f in enumerate(factors):
        for g in factors[i + 1:]:
            d = sub(g, f)
            if isinstance(d, Rational) and d.q == 1 and d.p % 2 == 1:
                return True
    return False


def _number_mod(a: Number, b: Number) -> Expr:
    va, vb = a.exact_value(), b.exact_value()
    if va is None or vb is None:
        return nan
    value = va - vb * (va / vb).__floor__()
    if isinstance(a, Float) or isinstance(b, Float):
        prec = max(a.precision if isinstance(a, Float) else 0, b.precision if isinstance(b, Float) else 0)
        return Float(number_from_fraction(value), precision=prec)
    return number_from_fraction(value)


def _coefficient(e: Expr) -> Fraction:
    c, _ = e.as_coeff_Mul()
    value = c.exact_value()
    return value if value is not None and not c.is_Float else Fraction(1)


def _common_factor(a: Expr, b: Expr) -> Optional[Expr]:
    """A common factor of ``a`` and ``b`` (a rational content and common
    symbolic factors), ``None`` when it is ``1``."""
    from math import gcd
    terms = terms_of(a)
    ca = _coefficient(a) if not isinstance(a, Add) else _content(terms)
    cb = _coefficient(b) if not isinstance(b, Add) else _content(terms_of(b))
    content = Fraction(gcd(ca.numerator, cb.numerator)) if ca.denominator == 1 and cb.denominator == 1 else Fraction(1)
    factors: list[Expr] = []
    if not isinstance(a, Add):
        for f in factors_of(a):
            if isinstance(f, Number):
                continue
            base, exponent = f.as_base_exp()
            if not (isinstance(exponent, Rational) and exponent.q == 1 and exponent.p > 0):
                continue
            other = _exponent_of(base, b)
            if other > 0:
                factors.append(pow(base, Integer(min(exponent.p, other))))
    if content == 1 and not factors:
        return None
    if content == 1:
        return mul(*factors)
    return mul(number_from_fraction(content), *factors)


def _content(terms: tuple[Expr, ...]) -> Fraction:
    from math import gcd, lcm
    result = Fraction(0)
    for t in terms:
        c = _coefficient(t)
        if c == 0:
            return Fraction(1)
        result = Fraction(gcd(result.numerator, c.numerator), lcm(result.denominator, c.denominator)) \
            if result else abs(c)
    return result if result else Fraction(1)


def _exponent_of(base: Expr, e: Expr) -> int:
    """The positive integer power of ``base`` dividing ``e`` (a product)."""
    if isinstance(e, Add):
        return 0
    for f in factors_of(e):
        b, exponent = f.as_base_exp()
        if b == base and isinstance(exponent, Rational) and exponent.q == 1 and exponent.p > 0:
            return exponent.p
    return 0


def _divide(e: Expr, factor: Expr) -> Expr:
    from .mul import div
    if isinstance(e, Add):
        return add(*[div(t, factor) for t in e.args])
    return div(e, factor)


def _is_integer_multiple(t: Expr, b: Expr) -> bool:
    from .mul import div
    return div(t, b).is_integer is True


def _reduce_terms(a: Expr, b: Expr) -> Optional[Expr]:
    """``a`` with its rational terms reduced modulo a rational ``b``, and a
    negative integer coefficient of an integer ``a`` reduced modulo an
    integer ``b``; ``None`` when nothing changes."""
    if not isinstance(b, Rational):
        if isinstance(a, Add):
            # the terms which are integer multiples of the modulus vanish
            kept = [t for t in a.args if not _is_integer_multiple(t, b)]
            if len(kept) < len(a.args):
                return mod(add(*kept), b)
        return None
    vb = b.as_fraction()
    if isinstance(a, Add):
        changed = False
        new_terms: list[Expr] = []
        for t in a.args:
            if isinstance(t, Rational):
                reduced = t.as_fraction() - vb * (t.as_fraction() / vb).__floor__()
                if reduced != t.as_fraction():
                    changed = True
                new_terms.append(number_from_fraction(reduced))
            elif b.q == 1 and t.is_integer is True and _coefficient(t).denominator == 1 \
                    and _coefficient(t) % vb == 0:
                changed = True
            else:
                new_terms.append(t)
        if changed:
            return mod(add(*new_terms), b)
        return None
    if b.q == 1 and a.is_integer is True:
        c, rest = a.as_coeff_Mul()
        if isinstance(c, Rational) and c.q == 1 and c.p < 0:
            return mod(mul(number_from_fraction(c.as_fraction() % vb), rest), b)
    return None
