"""Products: the class ``Mul`` and the evaluating function ``mul``.

``Mul(*factors)`` is a node with the given factors in the given order,
not simplified. The function ``mul(*factors)`` returns SymPy's canonical
product: nested products are flattened, the numbers are multiplied into
one coefficient, the powers of the same base whose exponents have the
same non-numerical part are collected (``mul(x, x)`` is ``x**2``,
``mul(x**y, x**(2*y))`` is ``x**(3*y)``, while ``x**y*x**z`` is kept),
numerical bases with the same exponent are multiplied (``sqrt(2)*sqrt(3)``
is ``sqrt(6)``), a zero coefficient gives zero, a coefficient alone in
front of a sum is distributed (``mul(2, x + y)`` is ``2*x + 2*y``), the
commutative factors are sorted with the coefficient first, and the
noncommutative ones keep their order (adjacent powers of the same base
are merged). The operators of ``Expr`` call ``mul``; ``Mul(...).doit()``
is ``mul(...)``.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Mul*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.mul.Mul
"""
from __future__ import annotations

from fractions import Fraction
from typing import Callable, TYPE_CHECKING, Iterable, Optional

from .add import Add, add
from .basic import Basic
from .expr import Expr, ExprLike, factors_of, to_expr
from .numbers import (ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity, Number, Rational, nan,
    number_from_fraction)
from .operations import AssocOp
from .parameters import global_parameters
from .sorting import ClassKey, compare_key

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Mul', 'mul', 'div', 'neg', 'prod']

#: the passes of ``mul`` on a product whose rebuilt factors need
#: flattening again (each pass simplifies strictly; the bound is a guard)
_MAX_PASSES = 8


class Mul(AssocOp):
    """An unevaluated product.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Mul
    >>> x, y = symbols('x y')
    >>> p = Mul(x, 2, x)
    >>> p
    x*2*x
    >>> p.args
    (x, 2, x)
    >>> p.doit()
    2*x**2
    """

    __slots__ = ()

    is_Mul = True
    identity = Integer(1)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (3, 0, 'Mul')

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import mul_facts
        return mul_facts(self._args)

    def as_content_primitive(self, radical: bool = False, clear: bool = True) -> tuple[Number, Expr]:
        coefficient, rest = self.as_coeff_Mul()
        content = Fraction(1)
        primitives: list[Expr] = []
        if isinstance(coefficient, Rational):
            content = abs(coefficient.as_fraction())
            if coefficient.p < 0:
                primitives.append(Integer(-1))
        else:
            primitives.append(coefficient)
        # a fractional coefficient takes the content of the sums with it even
        # with clear=False: (x + 2/3)/(2*n) gives (3*x + 2)/(6*n) in SymPy
        inner_clear = clear or (isinstance(coefficient, Rational) and coefficient.q != 1)
        for f in factors_of(rest):
            c, p = f.as_content_primitive(radical=radical, clear=inner_clear)
            value = c.exact_value()
            content *= value if value is not None else Fraction(1)
            primitives.append(p)
        return (number_from_fraction(content), mul(*primitives))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return mul(*[_as_expr(a) for a in args])

    def _sympystr(self, printer: StrPrinter) -> str:
        return printer.print_mul(self._args)

    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        if not (isinstance(old, Mul) and isinstance(new, Expr)):
            return None
        remaining = list(self._args)
        for factor in old.args:
            if factor not in remaining:
                return None
            remaining.remove(factor)
        return mul(new, *[_as_expr(f._subs(old, new)) for f in remaining])

    def as_coeff_Mul(self, rational: bool = False) -> tuple[Number, Expr]:
        number: Number = Integer(1)
        rest: list[Expr] = []
        for a in self._args:
            if isinstance(a, Number):
                number = number.number_mul(a)
            else:
                rest.append(a)
        if isinstance(number, ComplexInfinity):
            return (Integer(1), self)  # as SymPy: zoo*z + 3*z stays
        return (number, _raw(rest))

    def as_two_terms(self) -> tuple[Expr, Expr]:
        return (self._args[0], _raw(list(self._args[1:])))

    def as_base_exp(self) -> tuple[Expr, Expr]:
        # x**2*y**2 is (x*y)**2, as in SymPy, when every factor has the
        # same integer exponent
        pairs = [f.as_base_exp() for f in self._args]
        exponents = {e for _, e in pairs}
        if len(exponents) == 1:
            exponent = pairs[0][1]
            if isinstance(exponent, Rational) and exponent.q == 1 and exponent.p != 1:
                return (mul(*[b for b, _ in pairs]), exponent)
        return (self, Integer(1))

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        parts = [a.as_numer_denom() for a in self._args]
        return (mul(*[n for n, _ in parts]), mul(*[d for _, d in parts]))

    def could_extract_minus_sign(self) -> bool:
        return self.as_coeff_Mul()[0].is_negative is True

    def primitive(self) -> tuple[Number, Expr]:
        # SymPy: the absolute rational coefficient and the rest with the sign
        c, rest = self.as_coeff_Mul()
        if not isinstance(c, Rational) or c.p == 0:
            return (Integer(1), self)
        if c.p < 0:
            return (Rational(-c.p, c.q), mul(Integer(-1), rest))
        return (c, rest)


def _raw(factors: list[Expr]) -> Expr:
    if not factors:
        return Integer(1)
    if len(factors) == 1:
        return factors[0]
    return Mul(*factors)


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


#: the commutative factors during flattening, keyed by base and by the
#: non-numerical part of the exponent: the numerical coefficients of the
#: exponents, and the factor itself when it is the only one of its key
_Collected = dict[tuple[Expr, Expr], tuple[list[Number], Expr]]


def mul(*args: ExprLike) -> Expr:
    """The evaluated product of the arguments.

    The factors are taken as already canonical, as SymPy's ``Mul`` does:
    ``mul(Pow(x, 0))`` is ``x**0``, while ``Mul(Pow(x, 0)).doit()``
    evaluates bottom up.

    Examples
    ========

    >>> from sympy_extras.core import symbols, mul, sqrt, Rational
    >>> x, y = symbols('x y')
    >>> mul(x, x)
    x**2
    >>> mul(2, x, 3, y, x)
    6*x**2*y
    >>> mul(2, x + y)
    2*x + 2*y
    >>> mul(sqrt(2), sqrt(6))
    2*sqrt(3)
    >>> mul(x, 0), mul()
    (0, 1)
    """
    factors = [to_expr(a) for a in args]
    if not global_parameters.evaluate:
        return _raw(_flattened(factors))
    if any(f.is_Order for f in _flattened(factors)):
        return _with_orders(factors)
    for _ in range(_MAX_PASSES):
        result, again = _mul_pass(factors)
        if again is None:
            return result
        factors = again
    return _raw(factors)


def _is_one(n: Number) -> bool:
    return isinstance(n, Rational) and n.p == 1 and n.q == 1


def _mul_pass(args: list[Expr]) -> tuple[Expr, Optional[list[Expr]]]:
    """One pass of flattening and collecting; the second component is the
    list of factors to flatten again when a rebuilt factor is a product
    or a number."""
    pow = _pow
    coefficient: Number = Integer(1)
    collected: _Collected = {}
    noncommutative: list[Expr] = []
    for a in _flattened(args):
        if isinstance(a, Number):
            coefficient = coefficient.number_mul(a)
            continue
        if a.is_commutative is False:
            noncommutative.append(a)
            continue
        base, exponent = a.as_base_exp()
        c, rest = exponent.as_coeff_Mul()
        key = (base, rest)
        if key in collected:
            collected[key][0].append(c)
        else:
            collected[key] = ([c], a)
    if isinstance(coefficient, NaN):
        return (nan, None)
    # the commutative factors rebuilt: positive numerical bases with the
    # same exponent are multiplied together
    rebuilt: list[Expr] = []
    numerical: dict[Expr, list[Rational]] = {}
    for (base, rest), (cs, factor) in collected.items():
        total: Number = cs[0]
        for c in cs[1:]:
            total = total.number_add(c)
        exponent = rest if _is_one(total) else mul(total, rest)
        if isinstance(base, Rational) and _groupable(base, exponent, factor):
            if base.p > 0 and isinstance(exponent, Rational) and exponent.p < 0:
                base, exponent = Rational(base.q, base.p), Rational(-exponent.p, exponent.q)
            numerical.setdefault(exponent, []).append(base)
            continue
        if len(cs) == 1:
            rebuilt.append(factor)
            continue
        rebuilt.append(pow(base, exponent))
    for exponent, bases in numerical.items():
        if len(bases) == 1:
            rebuilt.append(_single_power(collected, bases[0], exponent))
            continue
        # the bases with the same exponent multiplied: (-5)**(2/3)*2**(2/3)
        # is (-10)**(2/3), (-2)**(1/3)*(-3)**(1/3) is (-1)**(2/3)*6**(1/3)
        negatives = sum(1 for b in bases if b.p < 0)
        product: Number = Integer(1)
        for b in bases:
            product = product.number_mul(Rational(abs(b.p), b.q))
        if negatives == 0:
            rebuilt.append(pow(product, exponent))
        elif negatives == 1:
            rebuilt.append(pow(product.number_mul(Integer(-1)), exponent))
        else:
            from .power import minus_one_pow
            if not isinstance(exponent, Rational):
                raise TypeError("a rational exponent is expected")
            rebuilt.append(minus_one_pow(negatives * exponent.as_fraction()))
            rebuilt.append(pow(product, exponent))
    # adjacent noncommutative powers of the same base are merged
    merged: list[Expr] = []
    for f in noncommutative:
        if merged:
            b1, e1 = merged[-1].as_base_exp()
            b2, e2 = f.as_base_exp()
            if b1 == b2 and e1.as_coeff_Mul()[1] == e2.as_coeff_Mul()[1]:
                merged[-1] = pow(b1, add(e1, e2))
                continue
        merged.append(f)
    again = [f for f in rebuilt + merged if f.is_Mul or isinstance(f, Number)]
    if again:
        return (Integer(0), [coefficient] + rebuilt + merged)
    if isinstance(coefficient, (Infinity, NegativeInfinity)):
        # oo absorbs the factors of known sign
        kept: list[Expr] = []
        for f in rebuilt:
            if f.is_positive is True:
                continue
            if f.is_negative is True:
                coefficient = coefficient.number_mul(Integer(-1))
                continue
            kept.append(f)
        rebuilt = kept
    elif isinstance(coefficient, ComplexInfinity):
        # zoo absorbs the nonzero factors known real or known non-real
        rebuilt = [f for f in rebuilt if not (f.is_zero is False and f.is_extended_real is not None)]
    rebuilt.sort(key=compare_key)
    if not rebuilt and not merged:
        return (coefficient, None)
    zero = coefficient.exact_value()
    if zero is not None and zero == 0:
        return (Integer(0), None)
    one = _is_one(coefficient)
    if one and len(rebuilt) + len(merged) == 1:
        return ((rebuilt + merged)[0], None)
    if not one and len(rebuilt) == 1 and not merged and coefficient.is_finite_number \
            and global_parameters.distribute:
        only = rebuilt[0]
        if isinstance(only, Add):
            return (add(*[mul(coefficient, t) for t in only.args]), None)
    factors = ([] if one else [coefficient]) + rebuilt + merged
    return (Mul(*factors), None)


def _with_orders(factors: list[Expr]) -> Expr:
    """A product with ``Order`` factors, evaluated by SymPy."""
    import sympy
    from .interop import expr_from_sympy, sympy_expr as _sympy_expr, to_sympy
    return expr_from_sympy(sympy.Mul(*[_sympy_expr(to_sympy(f)) for f in factors]))


def _flattened(args: list[Expr]) -> list[Expr]:
    """The factors with the nested products opened, in order."""
    result: list[Expr] = []
    for a in args:
        if isinstance(a, Mul):
            result.extend(_flattened(list(a.args)))
        else:
            result.append(a)
    return result


def _groupable(base: Rational, exponent: Expr, factor: Expr) -> bool:
    """Whether the numerical ``base**exponent`` may be multiplied with the
    other numerical bases of the same exponent: a positive base with a
    non-integer rational or a symbolic exponent (``sqrt(2)*sqrt(3)``,
    ``2**x*3**x``), a negative base under a rational exponent of
    denominator at least 3 (``I`` and ``sqrt(2)`` stay apart)."""
    from .power import Pow
    if base.p > 0:
        if isinstance(exponent, Rational):
            return exponent.q > 1
        return not isinstance(exponent, Number)
    if base.p < 0 and isinstance(factor, Pow):
        return isinstance(exponent, Rational) and exponent.q > 2 and exponent.p > 0
    return False


def _single_power(collected: _Collected, base: Rational, exponent: Expr) -> Expr:
    """``base**exponent`` for a numerical base alone in its exponent group:
    the factor as given when its exponent did not change."""
    from .power import Pow, pow
    single = collected.get((base, exponent.as_coeff_Mul()[1]))
    if single is not None and len(single[0]) == 1:
        factor = single[1]
        if isinstance(factor, Pow) and factor.base == base and factor.exp == exponent:
            return factor
    return pow(base, exponent)


def div(a: ExprLike, b: ExprLike) -> Expr:
    """The evaluated quotient ``a / b``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, div, Integer
    >>> x, y = symbols('x y')
    >>> div(x, y), div(Integer(1), 2), div(x, x)
    (x/y, 1/2, 1)
    """
    from .power import pow
    numerator = to_expr(a)
    if isinstance(numerator, Rational) and numerator.p == 1 and numerator.q == 1:
        return pow(b, Integer(-1))
    return mul(numerator, pow(b, Integer(-1)))


def prod(a: Iterable[ExprLike], start: ExprLike = 1) -> Expr:
    """The product of the elements of ``a`` times ``start``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, prod
    >>> x, y = symbols('x y')
    >>> prod([x, y, 2]), prod([], 3)
    (2*x*y, 3)
    """
    return mul(start, *a)


def neg(a: ExprLike) -> Expr:
    """The evaluated opposite ``-a``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, neg
    >>> x, y = symbols('x y')
    >>> neg(x), neg(x + y), neg(-2*x)
    (-x, -x - y, 2*x)
    """
    return mul(Integer(-1), a)

_POW: Optional[Callable[[ExprLike, ExprLike], Expr]] = None


def _pow(b: ExprLike, e: ExprLike) -> Expr:
    """``pow`` of the sibling module, bound at the first call (it imports this one)."""
    global _POW
    if _POW is None:
        from .power import pow
        _POW = pow
    return _POW(b, e)
