"""The domain ``EX`` of arbitrary expressions of the typed core, whose
elements are expressions (:class:`~sympy_extras.core.expr.Expr`).

Ported from SymPy's ``sympy.polys.domains.expressiondomain``: an element
is kept as it is given, and the result of an arithmetic operation is put
in the normal form ``cancel(e)`` expanded over products only
(``expand(deep=False, mul=True, multinomial=False, power_base=False,
power_exp=False, log=False)``), so that a rational function of the atoms
of the coefficients (symbols, ``sqrt(2)``, ``pi``, ``exp(a)``) has one
form; an element is zero when its ``is_zero`` is ``True``. The gcd of two
elements is ``1``, as in SymPy, so that the content of a polynomial over
``EX`` is ``1`` and its primitive part itself; an element is negative
when ``could_extract_minus_sign`` says so.

The domain is what the algorithms on polynomials with symbolic or
irrational coefficients use where SymPy's ``Poly`` would take the
domain ``EX`` (the rational integrator
:mod:`sympy_extras.integrals.risch.rationaltools`).

Examples
========

>>> from sympy_extras.core import symbols, sqrt
>>> from sympy_extras.polys.domains.expressiondomain import EX
>>> a = symbols('a')
>>> EX.add(1/a, a), EX.mul(sqrt(2), sqrt(2)/a), EX.exquo(a**2 - 1, a + 1)
(a + 1/a, 2/a, a - 1)
>>> EX.is_zero(EX.sub(sqrt(2)*sqrt(3), sqrt(6)))
True
"""
# ---------------------------------------------------------------------
# PORTED FROM SYMPY: sympy/polys/domains/expressiondomain.py (ExpressionDomain) (SymPy 1.14)
# Copyright (c) 2006-2023 SymPy Development Team. BSD 3-clause license,
# see sympy_extras/LICENSE-SymPy.
# Changes: a Domain[Expr] of the typed core (the elements are the
# expressions themselves, no wrapper class); typed.
# ---------------------------------------------------------------------
from __future__ import annotations

from sympy_extras.core.expr import Expr
from sympy_extras.core.function import expand
from sympy_extras.core.mul import mul
from sympy_extras.core.add import add
from sympy_extras.core.numbers import Integer, Number
from sympy_extras.core.power import pow

from .domain import Domain, U

__all__ = ['ExpressionDomain', 'RationalizedExpressionDomain', 'EX', 'EXR']

_ZERO = Integer(0)
_ONE = Integer(1)


def ex_normal(e: Expr) -> Expr:
    """SymPy's ``Expression.simplify``: ``e.cancel()`` expanded over the
    products (``expand(deep=False, mul=True, ...)``, the other hints off)."""
    from sympy_extras.polys.polytools import cancel
    return expand(cancel(e), deep=False, mul=True, multinomial=False, power_base=False, power_exp=False,
                  log=False)


class ExpressionDomain(Domain[Expr]):
    """The field ``EX`` of the expressions of the core (SymPy's
    ``ExpressionDomain``)."""

    name = 'EX'
    is_field = True
    is_ordered = False
    characteristic = 0

    def normal(self, e: Expr) -> Expr:
        """The normal form of the result of an operation (:func:`ex_normal`)."""
        return ex_normal(e)

    @property
    def zero(self) -> Expr:
        return _ZERO

    @property
    def one(self) -> Expr:
        return _ONE

    def is_zero(self, a: Expr) -> bool:
        if isinstance(a, Number):
            return a == _ZERO
        return a.is_zero is True

    def is_one(self, a: Expr) -> bool:
        return a == _ONE

    def is_negative(self, a: Expr) -> bool:
        """SymPy's ``EX.is_negative``: ``a.could_extract_minus_sign()``."""
        return a.could_extract_minus_sign()

    def is_unit(self, a: Expr) -> bool:
        return not self.is_zero(a)

    def add(self, a: Expr, b: Expr) -> Expr:
        if self.is_zero(b):
            return a
        if self.is_zero(a):
            return b
        return self.normal(add(a, b))

    def sub(self, a: Expr, b: Expr) -> Expr:
        if self.is_zero(b):
            return a
        if self.is_zero(a):
            return self.neg(b)
        return self.normal(add(a, mul(Integer(-1), b)))

    def neg(self, a: Expr) -> Expr:
        return mul(Integer(-1), a)

    def mul(self, a: Expr, b: Expr) -> Expr:
        if self.is_zero(a) or self.is_zero(b):
            return _ZERO
        if isinstance(a, Number) and isinstance(b, Number):
            return mul(a, b)
        return self.normal(mul(a, b))

    def pow(self, a: Expr, n: int) -> Expr:
        return self.normal(pow(a, Integer(n)))

    def exquo(self, a: Expr, b: Expr) -> Expr:
        if self.is_zero(b):
            raise ZeroDivisionError('division by zero in EX')
        return self.normal(mul(a, pow(b, Integer(-1))))

    def quo(self, a: Expr, b: Expr) -> Expr:
        return self.exquo(a, b)

    def rem(self, a: Expr, b: Expr) -> Expr:
        if self.is_zero(b):
            raise ZeroDivisionError('division by zero in EX')
        return _ZERO

    def invert(self, a: Expr) -> Expr:
        if self.is_zero(a):
            raise ZeroDivisionError('0 has no inverse')
        return self.normal(pow(a, Integer(-1)))

    def gcd(self, a: Expr, b: Expr) -> Expr:
        """``1``, as SymPy's ``EX.gcd``."""
        return _ONE

    def lcm(self, a: Expr, b: Expr) -> Expr:
        """SymPy's ``EX.lcm``: the ``lcm`` of the expressions."""
        from sympy_extras.polys.polytools import lcm
        return lcm(a, b)

    def gcdex(self, a: Expr, b: Expr) -> tuple[Expr, Expr, Expr]:
        if not self.is_zero(a):
            return self.invert(a), _ZERO, _ONE
        if not self.is_zero(b):
            return _ZERO, self.invert(b), _ONE
        return _ZERO, _ZERO, _ZERO

    def canonical_unit(self, a: Expr) -> Expr:
        return _ONE if self.is_zero(a) else self.invert(a)

    def numer(self, a: Expr) -> Expr:
        return a.as_numer_denom()[0]

    def denom(self, a: Expr) -> Expr:
        return a.as_numer_denom()[1]

    def from_int(self, n: int) -> Expr:
        return Integer(n)

    def convert(self, a: U, K: Domain[U]) -> Expr:
        return K.to_expr(a)

    def to_expr(self, a: Expr) -> Expr:
        return a

    def from_expr(self, e: Expr) -> Expr:
        return e


class RationalizedExpressionDomain(ExpressionDomain):
    """``EX`` with the denominators rationalized: the normal form of
    :func:`ex_normal` followed, for an expression with square roots of
    numbers, by ``radsimp`` and :func:`ex_normal` again, so that the
    arithmetic of ``QQ(sqrt(2), ...)`` keeps its elements small
    (``cancel`` takes ``sqrt(2)`` as a generator and does not reduce
    ``sqrt(2)**2``, so that the plain ``EX`` lets them grow through a
    gcd). Not SymPy's: its elements have other forms.

    >>> from sympy_extras.core import sqrt
    >>> from sympy_extras.polys.domains.expressiondomain import EX, EXR
    >>> EX.invert(3 - 2*sqrt(2)), EXR.invert(3 - 2*sqrt(2))
    (-1/(-3 + 2*sqrt(2)), 2*sqrt(2) + 3)
    """

    name = 'EX'

    def normal(self, e: Expr) -> Expr:
        from sympy_extras.core.basic import preorder_traversal
        from sympy_extras.core.numbers import Rational
        from sympy_extras.core.power import Pow
        from sympy_extras.simplify.radsimp import radsimp
        value = ex_normal(e)
        if any(isinstance(node, Pow) and isinstance(node.exp, Rational) and node.exp.q == 2
               and not node.base.free_symbols for node in preorder_traversal(value)):
            value = ex_normal(radsimp(value))
        return value


#: the domain of expressions
EX = ExpressionDomain()
#: the domain of expressions, the denominators rationalized
EXR = RationalizedExpressionDomain()
