"""Tools on expressions which belong to SymPy's ``exprtools``:
``factor_terms`` and ``gcd_terms`` take the common factors out of the
terms of a sum, ``factor_nc`` does it for noncommutative sums.

Ported from SymPy's ``sympy.core.exprtools``: a term is a rational
coefficient times a numerator and a denominator, products of bases with
positive integer exponents (``decompose_power``: ``x**(3/2)`` is
``sqrt(x)**3``, ``exp(2*x)`` is ``exp(x)**2``, ``x`` and ``sqrt(x)`` are
different bases); the gcd of the terms takes the smallest exponents of
the common bases and the gcd of the coefficients, and with
``fraction=True`` (the default of ``gcd_terms``) the terms are put over
the lcm of their denominators. ``factor_terms`` works on the arguments of
every node (``f(2*x + 4*y)`` becomes ``f(2*(x + 2*y))``), takes the sign
out of a sum of negative terms and, with ``radical=True``, the common
radicals of integers (``sqrt(2)*x + sqrt(6)`` is ``sqrt(2)*(x +
sqrt(3))``). ``factor_nc`` and the sums and integrals of SymPy's objects
go through SymPy.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: exprtools*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.exprtools
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/core/exprtools.py (decompose_power, Factors, Term,
# _gcd_terms, gcd_terms, factor_terms), sympy/core/mul.py (_keep_coeff) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: typed; Factors with integer exponents only (the only ones
# Term creates); the gcd of non-rational numbers is 1.
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Callable, Final, Optional, Sequence, Union, overload

import sympy

from .add import Add, add
from .basic import Basic
from .containers import Tuple
from .expr import Expr, ExprLike, factors_of, to_expr
from .function import Application
from .mul import Mul, _keep_coeff, mul
from .numbers import ImaginaryUnit, Integer, Number, Rational
from .power import Pow, pow
from .symbol import Dummy

__all__ = ['factor_terms', 'gcd_terms', 'factor_nc', 'decompose_power', 'Factors', 'Term']

#: what the functions accept: a node, or a list of them (factored one by one)
Terms = Union[ExprLike, Basic, Sequence[ExprLike]]


# PORTED FROM SYMPY: sympy/core/exprtools.py:decompose_power
def decompose_power(e: Expr) -> tuple[Expr, int]:
    """``(base, n)`` with ``e = base**n``, ``n`` an integer (``x**(3/2)`` is
    ``(sqrt(x), 3)``, ``x**(2*y)`` is ``(x**y, 2)``).

    >>> from sympy_extras.core import Symbol, Rational
    >>> x, y = Symbol('x'), Symbol('y')
    >>> decompose_power(x**Rational(3, 2)), decompose_power(x**(2*y)), decompose_power(1/x)
    ((sqrt(x), 3), (x**y, 2), (x, -1))
    """
    base, exponent = e.as_base_exp()
    if isinstance(exponent, Number):
        if isinstance(exponent, Rational):
            if exponent.q != 1:
                base = pow(base, Rational(1, exponent.q))
            return (base, exponent.p)
        return (e, 1)
    c, tail = exponent.as_coeff_Mul(rational=True)
    if isinstance(c, Rational):
        if c == Integer(-1):
            return (pow(base, tail), -1)
        if c != Integer(1):
            return (pow(base, _keep_coeff(Rational(1, c.q), tail)), c.p)
    return (e, 1)


def _number_gcd(a: Number, b: Number) -> Number:
    """SymPy's ``Rational.gcd`` (gcd of the numerators over the lcm of the
    denominators); ``1`` for other numbers."""
    if isinstance(a, Rational) and isinstance(b, Rational):
        from math import gcd, lcm
        return Rational(gcd(a.p, b.p), lcm(a.q, b.q))
    return Integer(1)


def _number_lcm(a: Number, b: Number) -> Number:
    if isinstance(a, Rational) and isinstance(b, Rational):
        from math import gcd, lcm
        return Rational(lcm(a.p, b.p), gcd(a.q, b.q))
    product = mul(a, b)
    return product if isinstance(product, Number) else Integer(1)


# PORTED FROM SYMPY: sympy/core/exprtools.py:Factors
class Factors:
    """A product of bases with nonzero integer exponents, ``{base: n}``
    (``I**2`` is folded into ``-1``)."""

    __slots__ = ('factors',)

    def __init__(self, factors: Optional[dict[Expr, int]] = None) -> None:
        result: dict[Expr, int] = {}
        for base, n in (factors or {}).items():
            if n:
                result[base] = n
        unit = result.pop(ImaginaryUnit(), 0) % 4
        if unit in (2, 3):
            result[Integer(-1)] = result.get(Integer(-1), 0) + 1
        if unit in (1, 3):
            result[ImaginaryUnit()] = 1
        if result.get(Integer(-1), 0) % 2 == 0:
            result.pop(Integer(-1), None)
        elif Integer(-1) in result:
            result[Integer(-1)] = 1
        self.factors: Final[dict[Expr, int]] = result

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Factors) and other.factors == self.factors

    def __hash__(self) -> int:
        return hash(frozenset(self.factors.items()))

    def __repr__(self) -> str:
        return 'Factors(%s)' % self.factors

    @property
    def is_one(self) -> bool:
        return not self.factors

    def as_expr(self) -> Expr:
        args: list[Expr] = []
        for factor, n in self.factors.items():
            if n != 1:
                b, e = factor.as_base_exp()
                args.append(pow(b, _keep_coeff(Integer(n), e)))
            else:
                args.append(factor)
        return mul(*args)

    def mul(self, other: Factors) -> Factors:
        factors = dict(self.factors)
        for factor, n in other.factors.items():
            factors[factor] = factors.get(factor, 0) + n
        return Factors(factors)

    def normal(self, other: Factors) -> tuple[Factors, Factors]:
        """``self`` and ``other`` with their common factors cancelled."""
        mine, theirs = dict(self.factors), dict(other.factors)
        for factor, n in self.factors.items():
            if factor not in other.factors:
                continue
            d = n - other.factors[factor]
            if d > 0:
                mine[factor] = d
                del theirs[factor]
            else:
                del mine[factor]
                if d:
                    theirs[factor] = -d
                else:
                    del theirs[factor]
        return Factors(mine), Factors(theirs)

    def quo(self, other: Factors) -> Factors:
        quotient = dict(self.factors)
        for factor, n in other.factors.items():
            if factor in quotient:
                d = quotient[factor] - n
                if d <= 0:
                    del quotient[factor]
                if d > 0:
                    quotient[factor] = d
        return Factors(quotient)

    def gcd(self, other: Factors) -> Factors:
        return Factors({f: min(n, other.factors[f]) for f, n in self.factors.items() if f in other.factors})

    def lcm(self, other: Factors) -> Factors:
        factors = dict(self.factors)
        for factor, n in other.factors.items():
            factors[factor] = max(n, factors[factor]) if factor in factors else n
        return Factors(factors)

    def pow(self, n: int) -> Factors:
        return Factors({f: e * n for f, e in self.factors.items()})


# PORTED FROM SYMPY: sympy/core/exprtools.py:Term
class Term:
    """A commutative term as ``coeff * numer / denom``."""

    __slots__ = ('coeff', 'numer', 'denom')

    def __init__(self, coeff: Number, numer: Factors, denom: Factors) -> None:
        self.coeff: Final[Number] = coeff
        self.numer: Final[Factors] = numer
        self.denom: Final[Factors] = denom

    @classmethod
    def of(cls, term: Expr) -> Term:
        if term.is_commutative is False:
            raise ValueError('commutative expression expected')
        coeff, factors = term.as_coeff_mul()
        numer: dict[Expr, int] = {}
        denom: dict[Expr, int] = {}
        for factor in factors:
            base, n = decompose_power(factor)
            if isinstance(base, Add):
                content, base = base.primitive()
                scaled = mul(coeff, pow(content, Integer(n)))
                if isinstance(scaled, Number):
                    coeff = scaled
            if n > 0:
                numer[base] = numer.get(base, 0) + n
            else:
                denom[base] = denom.get(base, 0) - n
        return cls(coeff, Factors(numer), Factors(denom))

    def as_expr(self) -> Expr:
        return mul(self.coeff, self.numer.as_expr(), pow(self.denom.as_expr(), Integer(-1)))

    def mul(self, other: Term) -> Term:
        coeff = mul(self.coeff, other.coeff)
        numer, denom = self.numer.mul(other.numer).normal(self.denom.mul(other.denom))
        return Term(coeff if isinstance(coeff, Number) else Integer(1), numer, denom)

    def inv(self) -> Term:
        inverse = pow(self.coeff, Integer(-1))
        return Term(inverse if isinstance(inverse, Number) else Integer(1), self.denom, self.numer)

    def quo(self, other: Term) -> Term:
        return self.mul(other.inv())

    def gcd(self, other: Term) -> Term:
        return Term(_number_gcd(self.coeff, other.coeff), self.numer.gcd(other.numer), self.denom.gcd(other.denom))

    def lcm(self, other: Term) -> Term:
        return Term(_number_lcm(self.coeff, other.coeff), self.numer.lcm(other.numer), self.denom.lcm(other.denom))


# PORTED FROM SYMPY: sympy/core/exprtools.py:_gcd_terms
def _gcd_terms(terms: Sequence[Expr], isprimitive: bool, fraction: bool) -> tuple[Expr, Expr, Expr]:
    """``(content, numerator, denominator)`` of a sum of terms."""
    parsed = [Term.of(t) for t in terms if not (isinstance(t, Number) and t == 0)]
    if not parsed:
        return (Integer(0), Integer(0), Integer(1))
    if len(parsed) == 1:
        content: Expr = parsed[0].coeff
        numer = parsed[0].numer.as_expr()
        denom = parsed[0].denom.as_expr()
    else:
        common = parsed[0]
        for term in parsed[1:]:
            common = common.gcd(term)
        parsed = [term.quo(common) for term in parsed]
        if fraction:
            lcm_denom = parsed[0].denom
            for term in parsed[1:]:
                lcm_denom = lcm_denom.lcm(term.denom)
            numers = [mul(term.coeff, term.numer.mul(lcm_denom.quo(term.denom)).as_expr()) for term in parsed]
            denom = lcm_denom.as_expr()
        else:
            numers = [term.as_expr() for term in parsed]
            denom = Integer(1)
        content = common.as_expr()
        numer = add(*numers)
    if not isprimitive and isinstance(numer, Add):
        c, numer = numer.primitive()
        content = mul(content, c)
    return (content, numer, denom)


def _mask(terms: Sequence[Expr]) -> tuple[list[Expr], dict[Basic, Basic]]:
    """The noncommutative parts of the terms replaced by dummies."""
    masked: list[Expr] = []
    replacements: dict[Basic, Basic] = {}
    for t in terms:
        if t.is_commutative is not False:
            masked.append(t)
            continue
        parts = factors_of(t)
        c = [f for f in parts if f.is_commutative is not False]
        nc = [f for f in parts if f.is_commutative is False]
        d = Dummy()
        replacements[d] = mul(*nc)
        masked.append(mul(*c, d))
    return masked, replacements


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError('an expression is expected, got %s' % (b,))
    return b


# PORTED FROM SYMPY: sympy/core/exprtools.py:gcd_terms (a sum)
def _gcd_terms_sum(terms: Sequence[Expr], isprimitive: bool, clear: bool, fraction: bool) -> Expr:
    masked, replacements = _mask(terms)
    content, numer, denom = _gcd_terms(masked, isprimitive, fraction)
    numer = _as_expr(numer.xreplace(replacements)) if replacements else numer
    coeff, factors = content.as_coeff_Mul()
    if not clear:
        c, rest = coeff.as_coeff_Mul()
        if not isinstance(c, Integer) and isinstance(numer, Add):
            n, d = c.as_numer_denom()
            divided = mul(numer, pow(d, Integer(-1)))
            if isinstance(divided, Add) and any(isinstance(a.as_coeff_Mul()[0], Integer) for a in divided.args):
                numer = divided
                coeff = _as_number(mul(n, rest))
    return _keep_coeff(coeff, mul(factors, numer, pow(denom, Integer(-1))), clear=clear)


def _as_number(e: Expr) -> Number:
    return e if isinstance(e, Number) else Integer(1)


@overload
def gcd_terms(terms: ExprLike, isprimitive: bool = False, clear: bool = True, fraction: bool = True) -> Expr: ...
@overload
def gcd_terms(terms: Basic, isprimitive: bool = False, clear: bool = True, fraction: bool = True) -> Basic: ...
@overload
def gcd_terms(terms: Sequence[ExprLike], isprimitive: bool = False, clear: bool = True,
              fraction: bool = True) -> Expr: ...
# PORTED FROM SYMPY: sympy/core/exprtools.py:gcd_terms
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
        return _gcd_terms_sum([to_expr(t) for t in terms], isprimitive, clear, fraction)
    e = _basic(terms)
    if isinstance(e, Add):
        return _gcd_terms_sum(list(e.args), isprimitive, clear, fraction)
    if not e.args:
        return e
    if isinstance(e, Mul):
        c, factors = e.as_coeff_mul()
        return _keep_coeff(c, mul(*[_as_expr(gcd_terms(f, isprimitive, clear, fraction)) for f in factors]),
                           clear=clear)
    return e._evaluated(tuple(gcd_terms(a, isprimitive, clear, fraction) if isinstance(a, Expr)
                              else _rebuild_args(a, lambda b: gcd_terms(b, isprimitive, clear, fraction))
                              for a in e.args))


def _rebuild_args(e: Basic, f: Callable[[Expr], Basic]) -> Basic:
    if not e.args:
        return e
    return e._evaluated(tuple(f(a) if isinstance(a, Expr) else _rebuild_args(a, f) for a in e.args))


@overload
def factor_terms(expr: ExprLike, radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> Expr: ...
@overload
def factor_terms(expr: Basic, radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> Basic: ...
@overload
def factor_terms(expr: Sequence[ExprLike], radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> list[Basic]: ...
# PORTED FROM SYMPY: sympy/core/exprtools.py:factor_terms
def factor_terms(expr: Terms, radical: bool = False, clear: bool = False, fraction: bool = False,
                 sign: bool = True) -> Union[Basic, list[Basic]]:
    """The common factors of the terms taken out, in every node (SymPy's
    ``factor_terms``). With ``clear=False`` (the default) a rational
    content is not taken out of a sum in which a term has an integer
    coefficient (``x/2 + y`` stays).

    Examples
    ========

    >>> from sympy_extras.core import symbols, factor_terms, Eq, Function, sqrt, Rational
    >>> x, y, z = symbols('x y z')
    >>> factor_terms(2*x + 4*y), factor_terms(x*y + x), factor_terms(-x - y), factor_terms(x**2*y + x*y**2)
    (2*(x + 2*y), x*(y + 1), -(x + y), x*y*(x + y))
    >>> factor_terms(x*(x + 1) + 2*x), factor_terms(1/x + 1/x**2), factor_terms(1/x + 1/y, fraction=True)
    (x*(x + 3), (1 + 1/x)/x, (x + y)/(x*y))
    >>> f = Function('f')
    >>> factor_terms(f(2*x + 4*y)*(3*x + 6)), factor_terms(Eq(2*x + 4, 2*y)), factor_terms([2*x + 4, x])
    (3*(x + 2)*f(2*(x + 2*y)), Eq(2*(x + 2), 2*y), [2*(x + 2), x])
    >>> factor_terms(4*x**Rational(3, 2) + 3*sqrt(x)), factor_terms(sqrt(2)*x + sqrt(6), radical=True)
    (sqrt(x)*(4*x + 3), sqrt(2)*(x + sqrt(3)))
    """
    if isinstance(expr, Sequence) and not isinstance(expr, Basic):
        return [factor_terms(_basic(e), radical, clear, fraction, sign) for e in expr]
    return _FactorTerms(radical, clear, fraction, sign).do(_basic(expr))


class _FactorTerms:
    """SymPy's ``factor_terms.do`` with its options."""

    __slots__ = ('radical', 'clear', 'fraction', 'sign')

    def __init__(self, radical: bool, clear: bool, fraction: bool, sign: bool) -> None:
        self.radical: Final[bool] = radical
        self.clear: Final[bool] = clear
        self.fraction: Final[bool] = fraction
        self.sign: Final[bool] = sign

    def do(self, e: Basic) -> Basic:
        from .interop import SymPyExprNode, sympy_expr, through_sympy
        if not e.args:
            return e
        if isinstance(e, SymPyExprNode):
            # SymPy's handling of its own objects (the content of an Integral is taken out)
            return through_sympy(e, lambda s: sympy.factor_terms(sympy_expr(s), radical=self.radical, clear=self.clear,
                                                                 fraction=self.fraction, sign=self.sign))
        if isinstance(e, (Pow, Application, Tuple)) or not isinstance(e, Expr):
            new = tuple(self.do(a) for a in e.args)
            return e if new == e.args else e._evaluated(new)
        content, p = e.as_content_primitive(radical=self.radical, clear=self.clear)
        if isinstance(p, Add):
            terms = [_as_expr(self.do(a)) for a in p.args]
            if all(t.as_coeff_Mul()[0].extract_multiplicatively(Integer(-1)) is not None for t in terms):
                content = _as_number(mul(content, Integer(-1)))
                terms = [mul(Integer(-1), t) for t in terms]
            special: dict[Basic, Basic] = {}
            for i, t in enumerate(terms):
                _, exponent = t.as_base_exp()
                if isinstance(exponent, Mul) and exponent != mul(*exponent.args):
                    d = Dummy()
                    special[d] = t
                    terms[i] = d
            factored = _as_expr(gcd_terms(Add(*terms) if len(terms) > 1 else terms[0], isprimitive=True,
                                          clear=self.clear, fraction=self.fraction))
            p = _as_expr(factored.xreplace(special)) if special else factored
        elif p.args:
            p = _as_expr(p._evaluated(tuple(self.do(a) for a in p.args)))
        return _keep_coeff(content, p, clear=self.clear, sign=self.sign)


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
