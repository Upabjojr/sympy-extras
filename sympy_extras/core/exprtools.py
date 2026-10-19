"""Tools on expressions which belong to SymPy's ``exprtools``:
``factor_terms`` and ``gcd_terms`` take the common factors out of the
terms of a sum, ``factor_nc`` does it for noncommutative sums.

The common factors of a sum are its rational content (``2*x + 4*y`` is
``2*(x + 2*y)``), the sign when every term is negative (``-x - y`` is
``-(x + y)``, for ``factor_terms`` only), and the factors shared by all
the terms, with their smallest integer exponent (``x**2*y + x*y**2`` is
``x*y*(x + y)``, ``1/x + 1/x**2`` is ``(1 + 1/x)/x``); with
``fraction=True`` (the default of ``gcd_terms``) the terms are first put
over a common denominator (``1/x + 1/y`` is ``(x + y)/(x*y)``). Both
functions work on the arguments of every node (``f(2*x + 4*y)`` becomes
``f(2*(x + 2*y))``), as SymPy's do [1]_. The combination of radicals
(``radical=True``) and ``factor_nc`` are SymPy's, through the conversion.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: exprtools*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.exprtools
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence, Union, overload

import sympy

from .add import Add, add
from .basic import Basic
from .expr import Expr, ExprLike, factors_of, terms_of, to_expr
from .mul import mul
from .numbers import Integer, Number, Rational
from .parameters import distribute
from .power import pow

__all__ = ['factor_terms', 'gcd_terms', 'factor_nc']

#: what the functions accept: a node, or a list of them (factored one by one)
Terms = Union[ExprLike, Basic, Sequence[ExprLike]]


@overload
def factor_terms(expr: ExprLike, radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> Expr: ...
@overload
def factor_terms(expr: Basic, radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> Basic: ...
@overload
def factor_terms(expr: Sequence[ExprLike], radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> list[Basic]: ...
def factor_terms(expr: Terms, radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> Union[Basic, list[Basic]]:
    """The common factors of the terms taken out, in every node (SymPy's
    ``factor_terms``). With ``clear=False`` (the default) a rational
    content is not taken out of a sum in which a term has an integer
    coefficient (``x/2 + y`` stays).

    Examples
    ========

    >>> from sympy_extras.core import symbols, factor_terms, Eq, Function
    >>> x, y, z = symbols('x y z')
    >>> factor_terms(2*x + 4*y), factor_terms(x*y + x), factor_terms(-x - y), factor_terms(x**2*y + x*y**2)
    (2*(x + 2*y), x*(y + 1), -(x + y), x*y*(x + y))
    >>> factor_terms(x*(x + 1) + 2*x), factor_terms(1/x + 1/x**2), factor_terms(1/x + 1/y, fraction=True)
    (x*(x + 3), (1 + 1/x)/x, (x + y)/(x*y))
    >>> f = Function('f')
    >>> factor_terms(f(2*x + 4*y)*(3*x + 6)), factor_terms(Eq(2*x + 4, 2*y)), factor_terms([2*x + 4, x])
    (3*(x + 2)*f(2*(x + 2*y)), Eq(2*(x + 2), 2*y), [2*(x + 2), x])
    """
    if isinstance(expr, Sequence) and not isinstance(expr, Basic):
        return [factor_terms(_basic(e), radical, clear, fraction, sign) for e in expr]
    if radical:
        from .interop import sympy_expr, through_sympy
        return through_sympy(_basic(expr), lambda s: sympy.factor_terms(sympy_expr(s), radical=True, clear=clear,
                                                                        fraction=fraction, sign=sign))
    return _factor(_basic(expr), fraction, 'always' if sign else 'with_factor', True, clear)


@overload
def gcd_terms(terms: ExprLike, isprimitive: bool = False, clear: bool = True, fraction: bool = True) -> Expr: ...
@overload
def gcd_terms(terms: Basic, isprimitive: bool = False, clear: bool = True, fraction: bool = True) -> Basic: ...
@overload
def gcd_terms(terms: Sequence[ExprLike], isprimitive: bool = False, clear: bool = True,
              fraction: bool = True) -> Expr: ...
def gcd_terms(terms: Terms, isprimitive: bool = False, clear: bool = True, fraction: bool = True) -> Basic:
    """The terms put over a common denominator with their content and
    their common factors taken out (SymPy's ``gcd_terms``); a list of
    terms is summed first. The sign is not taken out (``-x - y`` stays),
    and with ``isprimitive`` the content is not.

    Examples
    ========

    >>> from sympy_extras.core import symbols, gcd_terms
    >>> x, y = symbols('x y')
    >>> gcd_terms(x/2 + y/4), gcd_terms(2/x + 4/x**2), gcd_terms(-2*x - 4*y), gcd_terms([2*x + 4*y, 6*x])
    ((2*x + y)/4, 2*(x + 2)/x**2, 2*(-x - 2*y), 4*(2*x + y))
    """
    if isinstance(terms, Sequence) and not isinstance(terms, Basic):
        return _factor(add(*[to_expr(t) for t in terms]), fraction, 'never', not isprimitive, clear)
    return _factor(_basic(terms), fraction, 'never', not isprimitive, clear)


def factor_nc(expr: ExprLike) -> Expr:
    """The common factors of a noncommutative sum taken out (SymPy's ``factor_nc``).

    Examples
    ========

    >>> from sympy_extras.core import Symbol, factor_nc
    >>> x, A, B = Symbol('x'), Symbol('A', commutative=False), Symbol('B', commutative=False)
    >>> factor_nc(x*A + x*B)
    x*(A + B)
    """
    from .interop import expr_through_sympy
    return expr_through_sympy(to_expr(expr), sympy.factor_nc)


def _basic(e: Union[ExprLike, Basic]) -> Basic:
    return e if isinstance(e, Basic) else to_expr(e)


#: when the sign of a sum of negative terms is taken out: always
#: (``factor_terms``), only with another factor (``factor_terms`` with
#: ``sign=False``: ``-2*x - 2`` gives ``-2*(x + 1)`` but ``-x - y`` stays), never (``gcd_terms``)
Sign = Literal['always', 'with_factor', 'never']


def _factor(e: Basic, fraction: bool, sign: Sign, content: bool, clear: bool) -> Basic:
    """``factor_terms`` of every node of ``e``, bottom up."""
    from .interop import SymPyExprNode, sympy_expr, through_sympy
    if isinstance(e, SymPyExprNode):
        # SymPy's handling of its own objects (the content of an Integral is taken out)
        return through_sympy(e, lambda s: sympy.factor_terms(sympy_expr(s), clear=clear, fraction=fraction,
                                                             sign=sign == 'always'))
    if not e.args:
        return e
    rebuilt = e._evaluated(tuple(_factor(a, fraction, sign, content, clear) for a in e.args))
    if isinstance(rebuilt, Add):
        return _factor_sum(rebuilt, fraction, sign, content, clear)
    return rebuilt


def _factor_sum(e: Add, fraction: bool, sign: Sign, content: bool, clear: bool) -> Expr:
    """The common factors of the terms of the sum ``e`` taken out."""
    if fraction:
        combined = _over_common_denominator(e, clear)
        if combined is not None:
            numerator, denominator = combined
            factored = _factor(numerator, False, sign, content, clear)
            with distribute(False):
                return mul(factored if isinstance(factored, Expr) else numerator, pow(denominator, Integer(-1)))
    coefficient: Expr = Integer(1)
    rest: Expr = e
    if content:
        c, primitive = e.as_content_primitive(clear=clear)
        coefficient, rest = c, primitive
    terms = list(terms_of(rest))
    common = _common_factors(terms)
    negative = len(terms) > 1 and all(_coefficient_sign(t) < 0 for t in terms)
    extracted = common or not (coefficient.is_Number and coefficient == 1)
    if negative and (sign == 'always' or (sign == 'with_factor' and extracted)):
        coefficient = mul(coefficient, Integer(-1))
        terms = [mul(Integer(-1), t) for t in terms]
        rest = add(*terms)
    if common:
        factor = mul(*[pow(b, Integer(n)) for b, n in common.items()])
        inverse = mul(*[pow(b, Integer(-n)) for b, n in common.items()])
        rest = add(*[mul(t, inverse) for t in terms])
        coefficient = mul(coefficient, factor)
    with distribute(False):  # 2*(x + 2*y), not 2*x + 4*y
        return mul(coefficient, rest)


def _coefficient_sign(t: Expr) -> int:
    c = t.as_coeff_Mul()[0]
    return -1 if c.is_negative is True else 1


def _integer_powers(t: Expr) -> dict[Expr, int]:
    """The factors of ``t`` as ``base -> integer exponent`` (a factor
    with another exponent, as ``sqrt(x)`` or ``x**n``, is a base with the
    exponent ``1``; ``exp(2*x)`` is ``exp(x)**2``), the rational
    coefficient left out."""
    powers: dict[Expr, int] = {}
    for f in factors_of(t):
        if isinstance(f, Rational):
            continue
        b, e = f.as_base_exp()
        if isinstance(e, Rational) and e.q == 1:
            base, n = b, e.p
        else:
            c, inner = e.as_coeff_Mul()
            if isinstance(c, Rational) and c.q == 1 and c.p != 1:
                base, n = pow(b, inner), c.p
            else:
                base, n = f, 1
        powers[base] = powers.get(base, 0) + n
    return powers


def _common_factors(terms: list[Expr]) -> dict[Expr, int]:
    """The bases common to every term with the exponent to take out:
    the smallest when all are positive, the largest when all are
    negative (``1/x`` from ``1/x + 1/x**2``)."""
    if len(terms) < 2:
        return {}
    powers = [_integer_powers(t) for t in terms]
    common: dict[Expr, int] = {}
    for base in powers[0]:
        if not all(base in p for p in powers):
            continue
        exponents = [p[base] for p in powers]
        if all(n > 0 for n in exponents):
            common[base] = min(exponents)
        elif all(n < 0 for n in exponents):
            common[base] = max(exponents)
    return common


def _over_common_denominator(e: Add, clear: bool) -> Optional[tuple[Expr, Expr]]:
    """``(numerator, denominator)`` of the sum put over the least common
    denominator of its terms, ``None`` when no term has a denominator;
    the integer denominators are left in the numerator unless ``clear``."""
    parts = [t.as_numer_denom() for t in terms_of(e)]
    if all(isinstance(d, Number) and d == 1 for _, d in parts):
        return None
    lcm: dict[Expr, int] = {}
    for _, d in parts:
        for b, n in _integer_powers(d).items():
            lcm[b] = max(lcm.get(b, 0), n)
    from .intfunc import ilcm
    constant = 1
    for _, d in parts:
        c = d.as_coeff_Mul()[0]
        if clear and isinstance(c, Rational) and c.q == 1:
            constant = ilcm(constant, abs(c.p))
    denominator = mul(Integer(constant), *[pow(b, Integer(n)) for b, n in lcm.items()])
    numerator = add(*[mul(numer, denominator, pow(denom, Integer(-1))) for numer, denom in parts])
    return (numerator, denominator)
