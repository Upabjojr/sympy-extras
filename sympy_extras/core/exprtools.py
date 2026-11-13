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

from fractions import Fraction
from typing import Callable, Final, Optional, Sequence, Union, overload

from .add import Add, add
from .basic import Basic
from .containers import Tuple
from .expr import Expr, ExprLike, factors_of, to_expr
from .function import Application
from .mul import Mul, _keep_coeff, mul
from .numbers import ImaginaryUnit, Integer, Number, Rational, number_from_fraction
from .power import Pow, pow
from .symbol import Dummy, Symbol

__all__ = ['factor_terms', 'gcd_terms', 'factor_nc', 'decompose_power', 'Factors', 'ExprFactors', 'Term']

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
            return (pow(base, _keep_coeff(number_from_fraction(Fraction(1, c.q)), tail)), c.p)
    return (e, 1)


def _number_gcd(a: Number, b: Number) -> Number:
    """SymPy's ``Rational.gcd`` (gcd of the numerators over the lcm of the
    denominators); ``1`` for other numbers."""
    if isinstance(a, Rational) and isinstance(b, Rational):
        from math import gcd, lcm
        return number_from_fraction(Fraction(gcd(a.p, b.p), lcm(a.q, b.q)))
    return Integer(1)


def _number_lcm(a: Number, b: Number) -> Number:
    if isinstance(a, Rational) and isinstance(b, Rational):
        from math import gcd, lcm
        return number_from_fraction(Fraction(lcm(a.p, b.p), gcd(a.q, b.q)))
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


# PORTED FROM SYMPY: sympy/core/exprtools.py:Factors (any exponents)
class ExprFactors:
    """SymPy's ``Factors`` of an expression: ``{base: exponent}`` with any
    exponents (numbers or symbolic), ``I`` counted, rationals split into
    numerator and denominator. ``normal`` cancels the common factors of
    two products, ``gcd`` and ``lcm`` take the smaller and the larger
    exponents where their difference has a known sign."""

    __slots__ = ('factors',)

    def __init__(self, factors: dict[Expr, Expr]) -> None:
        self.factors: Final[dict[Expr, Expr]] = factors

    @classmethod
    def of(cls, e: Expr) -> ExprFactors:
        from .numbers import I as unit
        if isinstance(e, Number):
            return cls._of_number(e)
        if not e.args:
            return cls({e: Integer(1)})
        commutative = [f for f in factors_of(e) if f.is_commutative is not False]
        noncommutative = [f for f in factors_of(e) if f.is_commutative is False]
        units = sum(1 for f in commutative if f == unit)
        rest = [f for f in commutative if f != unit]
        powers: dict[Expr, Expr] = dict(Mul(*rest).as_powers_dict()) if len(rest) > 1 else (
            dict(rest[0].as_powers_dict()) if rest else {})
        for f in list(powers):
            if isinstance(f, Rational) and f.q != 1:
                p, q = Integer(f.p), Integer(f.q)
                powers[p] = add(powers.get(p, Integer(0)), powers[f])
                powers[q] = add(powers.get(q, Integer(0)), mul(Integer(-1), powers[f]))
                powers.pop(f)
        if units:
            powers[unit] = add(powers.get(unit, Integer(0)), Integer(units))
        if noncommutative:
            powers[Mul(*noncommutative)] = Integer(1)
        return cls(powers)

    @classmethod
    def _of_number(cls, n: Number) -> ExprFactors:
        if n == 0:
            return cls({Integer(0): Integer(1)})
        if n == 1:
            return cls({})
        result: dict[Expr, Expr] = {}
        value: Number = n
        if value.is_negative is True:
            result[Integer(-1)] = Integer(1)
            negated = mul(Integer(-1), value)
            value = negated if isinstance(negated, Number) else value
        if not (isinstance(value, Integer) and value.p == 1):
            if isinstance(value, Rational) and value.q != 1:
                if value.p != 1:
                    result[Integer(value.p)] = Integer(1)
                result[Integer(value.q)] = Integer(-1)
            else:
                result[value] = Integer(1)
        return cls(result)

    @property
    def is_zero(self) -> bool:
        return len(self.factors) == 1 and Integer(0) in self.factors

    @property
    def is_one(self) -> bool:
        return not self.factors

    def as_expr(self) -> Expr:
        args: list[Expr] = []
        for factor, exponent in self.factors.items():
            if exponent != Integer(1):
                if isinstance(exponent, Integer):
                    b, e = factor.as_base_exp()
                    args.append(pow(b, _keep_coeff(exponent, e)))
                else:
                    args.append(pow(factor, exponent))
            else:
                args.append(factor)
        return mul(*args)

    def normal(self, other: ExprFactors) -> tuple[ExprFactors, ExprFactors]:
        """``self`` and ``other`` with the common factors cancelled."""
        mine, theirs = dict(self.factors), dict(other.factors)
        for factor, self_exp in self.factors.items():
            if factor not in other.factors:
                continue
            other_exp = other.factors[factor]
            exp = add(self_exp, mul(Integer(-1), other_exp))
            if isinstance(exp, Number) and exp == 0:
                del mine[factor]
                del theirs[factor]
            elif isinstance(exp, Number):
                if exp.is_positive is True:
                    mine[factor] = exp
                    del theirs[factor]
                else:
                    del mine[factor]
                    theirs[factor] = mul(Integer(-1), exp)
            else:
                r = self_exp.extract_additively(other_exp)
                if r is not None:
                    if not (isinstance(r, Number) and r == 0):
                        mine[factor] = r
                        del theirs[factor]
                    else:
                        del mine[factor]
                        del theirs[factor]
                else:
                    sc, sa = self_exp.as_coeff_Add()
                    if not (isinstance(sc, Number) and sc == 0):
                        oc, oa = other_exp.as_coeff_Add()
                        diff = add(sc, mul(Integer(-1), oc))
                        if diff.is_positive is True:
                            mine[factor] = add(mine[factor], mul(Integer(-1), oc))
                            other_exp = oa
                        elif diff.is_negative is True:
                            mine[factor] = add(mine[factor], mul(Integer(-1), sc))
                            theirs[factor] = add(theirs[factor], mul(Integer(-1), sc))
                            other_exp = add(oa, mul(Integer(-1), diff))
                        else:
                            mine[factor] = sa
                            other_exp = oa
                    if not (isinstance(other_exp, Number) and other_exp == 0):
                        theirs[factor] = other_exp
                    else:
                        del theirs[factor]
        return ExprFactors(mine), ExprFactors(theirs)

    def quo(self, other: ExprFactors) -> ExprFactors:
        """SymPy's ``Factors.quo`` for the numerical exponents it is used
        with (the factor ``-1``)."""
        quotient = dict(self.factors)
        for factor, exp in other.factors.items():
            if factor in quotient:
                d = add(quotient[factor], mul(Integer(-1), exp))
                if isinstance(d, Number):
                    if d.is_positive is not True:
                        del quotient[factor]
                    if d.is_negative is not True and not (isinstance(d, Number) and d == 0):
                        quotient[factor] = d
                else:
                    r = quotient[factor].extract_additively(exp)
                    if r is not None:
                        if isinstance(r, Number) and r == 0:
                            del quotient[factor]
                        else:
                            quotient[factor] = r
        return ExprFactors(quotient)

    def gcd(self, other: ExprFactors) -> ExprFactors:
        if other.is_zero:
            return ExprFactors(dict(self.factors))
        result: dict[Expr, Expr] = {}
        for factor, exp in self.factors.items():
            if factor in other.factors:
                lt = add(exp, mul(Integer(-1), other.factors[factor])).is_negative
                if lt is True:
                    result[factor] = exp
                elif lt is False:
                    result[factor] = other.factors[factor]
        return ExprFactors(result)


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

    # PORTED FROM SYMPY: sympy/core/exprtools.py:_factor_sum_int
    def _factor_sum(self, e: Expr) -> Optional[Expr]:
        """The factors free of the variables taken out of the sum, or
        ``None`` when ``e`` is not a sum."""
        from sympy_extras.concrete.expr_with_limits import Sum, sum_
        if not isinstance(e, Sum):
            return None
        result = e.function
        if result == 0:
            return Integer(0)
        wrt = e.variables
        f = factor_terms(result, radical=self.radical, clear=self.clear, fraction=self.fraction, sign=self.sign)
        i, d = f.as_independent(*wrt)
        limits = list(e.limits)
        if isinstance(f, Add):
            return i * sum_(1, *limits) + sum_(d, *limits)
        return i * sum_(d, *limits)

    def do(self, e: Basic) -> Basic:
        if not e.args:
            return e
        own = e._eval_factor_terms(self.radical, self.clear, self.fraction, self.sign)
        if own is not None:
            # a node with a rule of its own (a wrapped foreign object)
            return own
        if isinstance(e, (Pow, Application, Tuple)) or not isinstance(e, Expr):
            new = tuple(self.do(a) for a in e.args)
            return e if new == e.args else e._evaluated(new)
        factored_sum = self._factor_sum(e)
        if factored_sum is not None:
            return factored_sum
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


# PORTED FROM SYMPY: sympy/core/exprtools.py:_mask_nc
def _mask_nc(eq: Expr, name: str = 'mask') -> tuple[Expr, dict[Basic, Basic], list[Symbol]]:
    """``eq`` with its noncommutative objects replaced by dummies: a single
    noncommutative symbol or object by a commutative dummy (it then
    factors as usual), the other objects by noncommutative dummies;
    with the mapping back and the noncommutative symbols."""
    from .sorting import default_sort_key
    from .symbol import Dummy
    counter = [0]

    def dummy(**assumptions: bool) -> Dummy:
        d = Dummy(name + str(counter[0]), **assumptions)
        counter[0] += 1
        return d

    if eq.is_commutative is not False:
        return eq, {}, []
    nc_obj: set[Basic] = set()
    nc_syms: set[Symbol] = set()

    def visit(a: Basic) -> None:
        # SymPy's preorder_traversal(keys=default_sort_key) with skip
        if isinstance(a, Expr) and a.is_commutative is False:
            if isinstance(a, Symbol):
                nc_syms.add(a)
                return
            if not (a.is_Add or a.is_Mul or a.is_Pow):
                nc_obj.add(a)
                return
        for child in sorted(a.args, key=default_sort_key):
            visit(child)

    visit(eq)
    rep: list[tuple[Basic, Symbol]] = []
    # a single noncommutative symbol or object factors as usual (polys
    # would complain): a commutative dummy
    if len(nc_obj) == 1 and not nc_syms:
        rep.append((nc_obj.pop(), dummy()))
    elif len(nc_syms) == 1 and not nc_obj:
        rep.append((nc_syms.pop(), dummy()))
    # the other noncommutative objects: noncommutative dummies
    for n in sorted(nc_obj, key=default_sort_key):
        nc = dummy(commutative=False)
        rep.append((n, nc))
        nc_syms.add(nc)
    masked = eq.subs(rep)
    return masked, {v: k for k, v in rep}, sorted(nc_syms, key=default_sort_key)


# PORTED FROM SYMPY: sympy/core/exprtools.py:factor_nc
def factor_nc(expr: ExprLike) -> Expr:
    """The common factors of a noncommutative sum taken out: the commutative
    gcd, the common noncommutative prefix and suffix (or common powers),
    the rest factored as a commutative polynomial whose factors are put
    back in an order which gives the sum.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, factor_nc, symbols
    >>> x, A, B = Symbol('x'), Symbol('A', commutative=False), Symbol('B', commutative=False)
    >>> factor_nc(x*A + x*B), factor_nc(A**2 + 2*A*B + B**2), factor_nc(A*B*x + A*x)
    (x*(A + B), (A + B)**2, x*A*(1 + B))
    >>> factor_nc(A**2 - A*B - B*A + B**2), factor_nc(A*x**2 + A*x)
    ((A - B)**2, x*(x + 1)*A)
    """
    from itertools import permutations
    from .add import Add, add
    from .function import expand, expand_power_exp
    from .mul import Mul, _keep_coeff, mul
    from .numbers import Integer
    from .power import Pow, pow
    from .sorting import default_sort_key
    from .symbol import Dummy
    from sympy_extras.polys.polytools import factor, gcd
    from sympy_extras.simplify.powsimp import powsimp
    e = to_expr(expr)
    if not e.args:
        return e
    if not isinstance(e, Add):
        rebuilt = e._evaluated(tuple(factor_nc(a) if isinstance(a, Expr) else a for a in e.args))
        return _as_expr(rebuilt)
    e = add(*[expand_power_exp(a) for a in e.args if isinstance(a, Expr)])
    masked, rep, nc_symbols = _mask_nc(e)
    if rep:
        return _as_expr(factor(masked).subs(rep))

    def cnc(t: Expr) -> list[list[Expr]]:
        # SymPy's args_cnc (the commutative prefix, the rest)
        fs = [f for f in (t.args if isinstance(t, Mul) else (t,)) if isinstance(f, Expr)]
        k = next((i for i, f in enumerate(fs) if f.is_commutative is not True), len(fs))
        c_part = fs[:k]
        if c_part and isinstance(c_part[0], Number) and c_part[0].is_extended_negative is True \
                and not (isinstance(c_part[0], Integer) and c_part[0].p == -1):
            c_part[:1] = [Integer(-1), -c_part[0]]
        return [c_part, fs[k:]]

    def raw_mul(fs: list[Expr]) -> Expr:
        # Mul._from_args
        if not fs:
            return Integer(1)
        return fs[0] if len(fs) == 1 else Mul(*fs)

    args = [cnc(a) for a in (masked.args if isinstance(masked, Add) else (masked,)) if isinstance(a, Expr)]
    c: Expr = Integer(1)
    g: Expr = Integer(1)
    lft: Expr = Integer(1)
    rgt: Expr = Integer(1)
    hit = False
    # a commutative gcd
    for i, a in enumerate(args):
        if i == 0:
            c = raw_mul(a[0])
        elif a[0]:
            c = gcd(c, raw_mul(a[0]))
        else:
            c = Integer(1)
    if not (isinstance(c, Integer) and c.p == 1):
        hit = True
        c, g = c.as_coeff_Mul()
        if not (isinstance(g, Integer) and g.p == 1):
            for i, (cc, _) in enumerate(args):
                q = raw_mul(list(cc)) / g
                args[i][0] = list(q.args) if isinstance(q, Mul) else [q]
        for i, (cc, _) in enumerate(args):
            if cc:
                cc[0] = cc[0] / c
            else:
                cc = [1 / c]
            args[i][0] = cc
    # a common noncommutative prefix
    n: list[Expr] = []
    found_prefix = True
    for i, a in enumerate(args):
        if i == 0:
            n = a[1][:]
        else:
            n = _common_prefix(n, a[1])
        if not n:
            found_prefix = False
            # a power which can be extracted
            if not args[0][1]:
                break
            b, ex = args[0][1][0].as_base_exp()
            ok = False
            if isinstance(ex, Integer):
                least = ex.p
                for t in args:
                    if not t[1]:
                        break
                    bt, et = t[1][0].as_base_exp()
                    if isinstance(et, Integer) and bt == b:
                        least = min(least, et.p)
                    else:
                        break
                else:
                    ok = hit = True
                    lft = pow(b, Integer(least))
                    il = pow(b, Integer(-least))
                    for t in args:
                        t[1][0] = il * t[1][0]
            break
    if found_prefix and args:
        hit = True
        lenn = len(n)
        lft = mul(*n)
        for t in args:
            t[1] = t[1][lenn:]
    # a common noncommutative suffix
    found_suffix = True
    for i, a in enumerate(args):
        if i == 0:
            n = a[1][:]
        else:
            n = _common_suffix(n, a[1])
        if not n:
            found_suffix = False
            if not args[0][1]:
                break
            b, ex = args[0][1][-1].as_base_exp()
            if isinstance(ex, Integer):
                least = ex.p
                for t in args:
                    if not t[1]:
                        break
                    bt, et = t[1][-1].as_base_exp()
                    if isinstance(et, Integer) and bt == b:
                        least = min(least, et.p)
                    else:
                        break
                else:
                    hit = True
                    rgt = pow(b, Integer(least))
                    il = pow(b, Integer(-least))
                    for t in args:
                        t[1][-1] = t[1][-1] * il
            break
    if found_suffix and args:
        hit = True
        lenn = len(n)
        rgt = mul(*n)
        for t in args:
            t[1] = t[1][:len(t[1]) - lenn]
    mid = add(*[mul(*cc) * mul(*nc) for cc, nc in args]) if hit else masked
    # the symbols sorted so that the dummies come in the order of the
    # symbols (no factor -1 introduced: A**2 - B**2)
    rep1: list[tuple[Basic, Basic]] = [(s, Dummy()) for s in sorted(nc_symbols, key=default_sort_key)]
    unrep1 = [(v, k) for k, v in rep1]
    unrep1.reverse()
    new_mid, r2, _ = _mask_nc(mid.subs(rep1))
    new_mid = powsimp(factor(new_mid))
    new_mid = new_mid.subs(r2).subs(unrep1)
    if isinstance(new_mid, Pow):
        return _keep_coeff(c, g * lft * new_mid * rgt)
    if isinstance(new_mid, Mul):

        def _pemexpand(t: Expr) -> Expr:
            # the minimal hints to check the result
            return expand(t, deep=True, mul=True, power_exp=True, power_base=False, multinomial=True, log=False)

        cfac: list[Expr] = []
        ncfac: list[Expr] = []
        for f in new_mid.args:
            if not isinstance(f, Expr):
                continue
            if f.is_commutative is not False:
                cfac.append(f)
            else:
                b, ex = f.as_base_exp()
                if isinstance(ex, Integer) and ex.p > 0:
                    ncfac.extend([b] * ex.p)
                else:
                    ncfac.append(f)
        pre_mid = g * mul(*cfac) * lft
        target = _pemexpand(e / c)
        for perm in permutations(ncfac, len(ncfac)):
            ok_expr = pre_mid * mul(*perm) * rgt
            if _pemexpand(ok_expr) == target:
                return _keep_coeff(c, ok_expr)
    # mid was a sum which did not factor
    return _keep_coeff(c, g * lft * mid * rgt)


def _common_prefix(a: list[Expr], b: list[Expr]) -> list[Expr]:
    """SymPy's ``common_prefix`` of two sequences."""
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return a[:k]


def _common_suffix(a: list[Expr], b: list[Expr]) -> list[Expr]:
    """SymPy's ``common_suffix`` of two sequences."""
    k = 0
    while k < len(a) and k < len(b) and a[len(a) - 1 - k] == b[len(b) - 1 - k]:
        k += 1
    return a[len(a) - k:] if k else []


def _basic(e: Union[ExprLike, Basic]) -> Basic:
    return e if isinstance(e, Basic) else to_expr(e)
