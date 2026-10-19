"""Sums: the class ``Add`` and the evaluating function ``add``.

``Add(*terms)`` is a node with the given terms, in the given order, which
is not simplified: ``Add(x, x)`` has the arguments ``(x, x)``. The function
``add(*terms)`` returns the canonical sum of SymPy: nested sums are
flattened, the numbers are summed, the terms with the same non-numerical
part are collected (``add(x, x)`` is ``2*x``), the zero terms are dropped,
the arguments are sorted (``add(y, x) == add(x, y)``) with the number
first, and a sum of one term is that term. The operators of ``Expr`` call
``add``; ``Add(...).doit()`` is ``add(...)``.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Add*,
       https://docs.sympy.org/latest/modules/core.html#sympy.core.add.Add
"""
from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Optional

from .basic import Basic
from .expr import Expr, ExprLike, to_expr
from .numbers import (ComplexInfinity, Infinity, Integer, NaN, NegativeInfinity, Number, Rational,
    nan, number_from_fraction)
from .operations import AssocOp
from .parameters import global_parameters
from .sorting import ArgsKey, ClassKey, SortKey, default_sort_key

if TYPE_CHECKING:
    from .printing import StrPrinter

__all__ = ['Add', 'add', 'sub', 'to_expr']


class Add(AssocOp):
    """An unevaluated sum.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Add
    >>> x, y = symbols('x y')
    >>> s = Add(x, 1, x)
    >>> s
    x + 1 + x
    >>> s.args
    (x, 1, x)
    >>> s.doit()
    2*x + 1
    """

    __slots__ = ()

    is_Add = True
    identity = Integer(0)

    @classmethod
    def class_key(cls) -> ClassKey:
        return (3, 1, 'Add')

    def _eval_facts(self) -> dict[str, bool]:
        from .assumptions import add_facts
        return add_facts(self._args)

    def as_content_primitive(self, radical: bool = False, clear: bool = True) -> tuple[Number, Expr]:
        return self.primitive()

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return add(*[_as_expr(a) for a in args])

    def _sympystr(self, printer: StrPrinter) -> str:
        return printer.print_add(self._args)

    def _subs_partial(self, old: Basic, new: Basic) -> Optional[Basic]:
        if not (isinstance(old, Add) and isinstance(new, Expr)):
            return None
        remaining = list(self._args)
        for term in old.args:
            if term not in remaining:
                return None
            remaining.remove(term)
        return add(new, *[_as_expr(t._subs(old, new)) for t in remaining])

    def sort_key(self) -> SortKey:
        # the arguments in the printing order, as SymPy's key does
        from .printing import ordered_terms
        key = Expr.sort_key(self)
        args: ArgsKey = (len(self._args), tuple(a.sort_key() for a in ordered_terms(self._args)))
        return (key[0], args, key[2], key[3], key[4])

    def as_coeff_Add(self) -> tuple[Number, Expr]:
        number: Number = Integer(0)
        rest: list[Expr] = []
        for a in self._args:
            if isinstance(a, Number):
                number = number.number_add(a)
            else:
                rest.append(a)
        return (number, _raw(rest))

    def as_two_terms(self) -> tuple[Expr, Expr]:
        return (self._args[0], _raw(list(self._args[1:])))

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        from .mul import mul
        parts = [a.as_numer_denom() for a in self._args]
        denominator = mul(*[d for _, d in parts])
        numerators = [mul(n, *[d for j, (_, d) in enumerate(parts) if j != i])
                      for i, (n, _) in enumerate(parts)]
        return (add(*numerators), denominator)

    def primitive(self) -> tuple[Number, Expr]:
        from .mul import mul
        coefficients = [a.as_coeff_Mul()[0] for a in self._args]
        values = [c.exact_value() for c in coefficients]
        if any(v is None for v in values) or any(c.is_Float for c in coefficients):
            return (Integer(1), self)
        g = Fraction(0)
        for v in values:
            if v is not None:
                g = _gcd_fractions(g, abs(v))
        if g == 0 or g == 1:
            return (Integer(1), self)
        content = number_from_fraction(g)
        inverse = number_from_fraction(1 / g)
        return (content, add(*[mul(inverse, a) for a in self._args]))


def _gcd_fractions(a: Fraction, b: Fraction) -> Fraction:
    from math import gcd, lcm
    return Fraction(gcd(a.numerator, b.numerator), lcm(a.denominator, b.denominator))


def _raw(terms: list[Expr]) -> Expr:
    """``Add(*terms)`` without evaluation, one term as itself, none as 0."""
    if not terms:
        return Integer(0)
    if len(terms) == 1:
        return terms[0]
    return Add(*terms)


def _as_expr(b: Basic) -> Expr:
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def add(*args: ExprLike) -> Expr:
    """The evaluated sum of the arguments.

    Examples
    ========

    >>> from sympy_extras.core import symbols, add, Rational
    >>> x, y = symbols('x y')
    >>> add(x, x)
    2*x
    >>> add(x, 2*y, 1, x, -1)
    2*x + 2*y
    >>> add(Rational(1, 2), Rational(1, 2))
    1
    >>> add(x, -x)
    0
    >>> add()
    0
    """
    from .mul import mul
    stack = [to_expr(a) for a in args]
    if not global_parameters.evaluate:
        return _raw([t for a in stack for t in (a.args if isinstance(a, Add) else (a,))])
    if any(t.is_Order for a in stack for t in (a.args if isinstance(a, Add) else (a,))):
        return _with_orders(stack)
    number: Number = Integer(0)
    coefficients: dict[Expr, Number] = {}
    for a in stack:
        if isinstance(a, Add):
            stack.extend(a.args)
            continue
        if isinstance(a, Number):
            number = number.number_add(a)
            continue
        c, rest = a.as_coeff_Mul()
        if rest in coefficients:
            coefficients[rest] = coefficients[rest].number_add(c)
        else:
            coefficients[rest] = c
    if isinstance(number, NaN):
        return nan
    terms: list[Expr] = []
    needs_pass = False
    for rest, c in coefficients.items():
        if isinstance(c, NaN):
            return nan
        value = c.exact_value()
        if value is not None and value == 0:
            continue
        if _absorbed_in_sum(number, rest, c):
            continue
        if isinstance(c, Rational) and c.p == 1 and c.q == 1:
            term = rest
        else:
            term = mul(c, rest)
        if term.is_Add or isinstance(term, Number):
            needs_pass = True
        terms.append(term)
    if needs_pass:
        return add(number, *terms)
    terms.sort(key=default_sort_key)
    nonzero = not (isinstance(number, Rational) and number.p == 0)
    if nonzero:
        terms.insert(0, number)
    if not terms:
        return number
    if len(terms) == 1:
        return terms[0]
    return Add(*terms)


def _with_orders(terms: list[Expr]) -> Expr:
    """A sum with ``Order`` terms, evaluated by SymPy (the absorption of
    the terms into the order needs the series machinery)."""
    import sympy
    from .interop import expr_from_sympy, sympy_expr as _sympy_expr, to_sympy
    return expr_from_sympy(sympy.Add(*[_sympy_expr(to_sympy(t)) for t in terms]))


def _absorbed_in_sum(number: Number, rest: Expr, c: Number) -> bool:
    """Whether the term ``c*rest`` is absorbed by the infinite ``number``:
    ``oo`` absorbs the real terms (``oo + p`` is ``oo``), ``zoo`` the finite
    terms which are known real or known non-real (``zoo + 3*I`` is ``zoo``),
    as SymPy does."""
    if isinstance(number, (Infinity, NegativeInfinity)):
        if not c.is_finite_number:
            return False
        return rest.is_real is True
    if isinstance(number, ComplexInfinity):
        if not c.is_finite_number:
            return False
        return rest.is_finite is True and rest.is_extended_real is not None
    return False


def sub(a: ExprLike, b: ExprLike) -> Expr:
    """The evaluated difference ``a - b``.

    Examples
    ========

    >>> from sympy_extras.core import symbols, sub
    >>> x, y = symbols('x y')
    >>> sub(x, y), sub(x, x), sub(2*x, x)
    (x - y, 0, x)
    """
    from .mul import mul
    return add(a, mul(Integer(-1), b))
