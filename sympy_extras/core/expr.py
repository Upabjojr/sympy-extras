"""Algebraic expressions.

:class:`Expr` is the base class of everything which can be added and
multiplied. Its operators call the evaluating functions of the package
(``x + x`` is ``add(x, x)``, which is ``2*x``), so that expressions written
with operators behave as in SymPy, while the classes ``Add``, ``Mul`` and
``Pow`` build unevaluated nodes. The methods are SymPy's
(``as_coeff_Mul``, ``as_base_exp``, ``as_independent``, ``diff``,
``expand``, ``evalf``, the ``is_*`` assumptions, ...) with the same
meanings [1]_.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Expr*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.expr
"""
from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Mapping, Optional, Union

from .basic import Atom, Basic, SubsArg, SubsMapping
from .sorting import ArgsKey, ExpKey, SortKey

if TYPE_CHECKING:
    from .assumptions import Truth
    from .numbers import Number
    from .symbol import Symbol

__all__ = ['Expr', 'AtomicExpr', 'ExprLike', 'as_expr', 'terms_of', 'factors_of']

#: what the operators and the evaluating functions accept besides ``Expr``
ExprLike = Union['Expr', int, float, Fraction]


def as_expr(b: Basic) -> Expr:
    """``b``, which must be an :class:`Expr` (narrowing at the boundary)."""
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def terms_of(e: Expr) -> tuple[Expr, ...]:
    """The terms of a sum, ``(e,)`` for anything else."""
    from .add import Add
    return e.args if isinstance(e, Add) else (e,)


def factors_of(e: Expr) -> tuple[Expr, ...]:
    """The factors of a product, ``(e,)`` for anything else."""
    from .mul import Mul
    return e.args if isinstance(e, Mul) else (e,)


class Expr(Basic):
    """An algebraic expression.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, Add
    >>> x = Symbol('x')
    >>> x + x
    2*x
    >>> Add(x, x)
    x + x
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    # typed versions of the tree methods

    def doit(self) -> Expr:
        return as_expr(super().doit())

    def xreplace(self, rule: Mapping[Basic, Basic]) -> Expr:
        return as_expr(self._xreplace(rule))

    def subs(self, *args: SubsArg) -> Expr:
        return as_expr(super().subs(*args))

    # ------------------------------------------------------------------
    # arithmetic: the operators evaluate

    def __add__(self, other: ExprLike) -> Expr:
        from .add import add
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return add(self, other)

    def __radd__(self, other: ExprLike) -> Expr:
        from .add import add
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return add(other, self)

    def __sub__(self, other: ExprLike) -> Expr:
        from .add import sub
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return sub(self, other)

    def __rsub__(self, other: ExprLike) -> Expr:
        from .add import sub
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return sub(other, self)

    def __mul__(self, other: ExprLike) -> Expr:
        from .mul import mul
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return mul(self, other)

    def __rmul__(self, other: ExprLike) -> Expr:
        from .mul import mul
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return mul(other, self)

    def __truediv__(self, other: ExprLike) -> Expr:
        from .mul import div
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return div(self, other)

    def __rtruediv__(self, other: ExprLike) -> Expr:
        from .mul import div
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return div(other, self)

    def __pow__(self, other: ExprLike) -> Expr:
        from .power import pow
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return pow(self, other)

    def __rpow__(self, other: ExprLike) -> Expr:
        from .power import pow
        if not isinstance(other, (Expr, int, float, Fraction)):
            return NotImplemented
        return pow(other, self)

    def __neg__(self) -> Expr:
        from .mul import neg
        return neg(self)

    def __pos__(self) -> Expr:
        return self

    def __lt__(self, other: ExprLike) -> Basic:
        from .relational import lt
        return lt(self, other)

    def __le__(self, other: ExprLike) -> Basic:
        from .relational import le
        return le(self, other)

    def __gt__(self, other: ExprLike) -> Basic:
        from .relational import gt
        return gt(self, other)

    def __ge__(self, other: ExprLike) -> Basic:
        from .relational import ge
        return ge(self, other)

    def __float__(self) -> float:
        from .evalf import to_float
        return to_float(self)

    def __int__(self) -> int:
        return int(float(self))

    def __complex__(self) -> complex:
        from .evalf import to_complex
        return to_complex(self)

    # ------------------------------------------------------------------
    # ordering

    def sort_key(self) -> SortKey:
        coeff, rest = self.as_coeff_Mul()
        base, exp = rest.as_base_exp() if rest.is_Pow else (rest, None)
        if isinstance(base, Atom):
            args: ArgsKey = (1, ())
            name = str(base)
        else:
            args = (len(base.args), tuple(a.sort_key() for a in base.args))
            name = ''
        from .numbers import Integer
        exp_key: ExpKey = Integer(1).sort_key() if exp is None else exp.sort_key()
        return (base.class_key(), args, name, exp_key, coeff.sort_value())

    # ------------------------------------------------------------------
    # decomposition

    def as_coeff_Mul(self, rational: bool = False) -> tuple[Number, Expr]:
        """The numerical coefficient and the rest of a product.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (3*x*y).as_coeff_Mul()
        (3, x*y)
        >>> x.as_coeff_Mul()
        (1, x)
        """
        from .numbers import Integer
        return (Integer(1), self)

    def as_coeff_Add(self) -> tuple[Number, Expr]:
        """The numerical term and the rest of a sum.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (x + y + 1).as_coeff_Add()
        (1, x + y)
        """
        from .numbers import Integer
        return (Integer(0), self)

    def as_coeff_mul(self) -> tuple[Number, tuple[Expr, ...]]:
        """The numerical coefficient and the tuple of the other factors."""
        c, rest = self.as_coeff_Mul()
        if rest.is_Number and rest == 1:
            return (c, ())
        return (c, factors_of(rest))

    def as_coeff_add(self) -> tuple[Number, tuple[Expr, ...]]:
        """The numerical term and the tuple of the other terms."""
        c, rest = self.as_coeff_Add()
        if rest.is_Number and rest == 0:
            return (c, ())
        return (c, terms_of(rest))

    def as_base_exp(self) -> tuple[Expr, Expr]:
        """The base and the exponent, ``(self, 1)`` for a non-power.

        Examples
        ========

        >>> from sympy_extras.core import Symbol, Rational
        >>> x = Symbol('x')
        >>> (x**2).as_base_exp()
        (x, 2)
        >>> Rational(1, 2).as_base_exp()
        (2, -1)
        """
        from .numbers import Integer
        return (self, Integer(1))

    def as_two_terms(self) -> tuple[Expr, Expr]:
        """The first argument and the rest, for a sum or a product."""
        return (self, self)

    def as_numer_denom(self) -> tuple[Expr, Expr]:
        """A numerator and a denominator whose quotient is ``self``.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (x/y).as_numer_denom()
        (x, y)
        >>> (x + 1/y).as_numer_denom()
        (x*y + 1, y)
        """
        from .numbers import Integer
        return (self, Integer(1))

    def as_independent(self, *symbols: Basic, as_Add: Optional[bool] = None) -> tuple[Expr, Expr]:
        """The part free of ``symbols`` and the part depending on them, of a
        sum (``as_Add=True``) or of a product.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (2*x*y).as_independent(x)
        (2*y, x)
        >>> (x + y + 1).as_independent(x)
        (y + 1, x)
        >>> (2*x*y + 3).as_independent(y, as_Add=False)
        (1, 2*x*y + 3)
        """
        from .add import add
        from .mul import mul
        from .numbers import Integer
        if as_Add is None:
            as_Add = self.is_Add
        if as_Add:
            if not self.is_Add:
                return (self, Integer(0)) if not self.has(*symbols) else (Integer(0), self)
            free = [a for a in terms_of(self) if not a.has(*symbols)]
            dependent = [a for a in terms_of(self) if a.has(*symbols)]
            return (add(*free), add(*dependent))
        if not self.is_Mul:
            return (self, Integer(1)) if not self.has(*symbols) else (Integer(1), self)
        free = [a for a in factors_of(self) if not a.has(*symbols)]
        dependent = [a for a in factors_of(self) if a.has(*symbols)]
        return (mul(*free), mul(*dependent))

    def as_coefficients_dict(self) -> dict[Expr, Number]:
        """The terms of a sum mapped to their numerical coefficients.

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> x = Symbol('x')
        >>> (x**2 + 2*x).as_coefficients_dict() == {x**2: 1, x: 2}
        True
        """
        from .numbers import Integer
        result: dict[Expr, Number] = {}
        for term in terms_of(self):
            c, rest = term.as_coeff_Mul()
            if rest in result:
                result[rest] = result[rest].number_add(c)
            else:
                result[rest] = c
        if not result:
            result[Integer(1)] = Integer(0)
        return result

    def as_powers_dict(self) -> dict[Expr, Expr]:
        """The bases of a product mapped to their exponents."""
        result: dict[Expr, Expr] = {}
        from .add import add
        for f in factors_of(self):
            b, e = f.as_base_exp()
            result[b] = add(result[b], e) if b in result else e
        return result

    def as_ordered_terms(self) -> list[Expr]:
        """The terms of a sum in the printing order."""
        from .printing import ordered_terms
        return ordered_terms(terms_of(self))

    def as_ordered_factors(self) -> list[Expr]:
        """The factors of a product in canonical order."""
        from .sorting import default_sort_key
        return sorted(factors_of(self), key=default_sort_key)

    def coeff(self, x: Expr, n: int = 1) -> Expr:
        """The coefficient of ``x**n`` among the terms of a sum.

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> x = Symbol('x')
        >>> (2*x**2 + 3*x + 1).coeff(x, 2)
        2
        """
        from .add import add
        from .mul import mul
        from .numbers import Integer
        from .power import pow
        target = pow(x, Integer(n)) if n != 1 else x
        found: list[Expr] = []
        for term in terms_of(self):
            factors = list(factors_of(term))
            if n == 0:
                if not term.has(x):
                    found.append(term)
                continue
            if target in factors:
                factors.remove(target)
                found.append(mul(*factors))
        return add(*found)

    def could_extract_minus_sign(self) -> bool:
        """Whether the expression is printed with a leading minus."""
        c, _ = self.as_coeff_Mul()
        return c.is_negative is True

    def primitive(self) -> tuple[Number, Expr]:
        """The positive rational content of the terms and the rest."""
        from .numbers import Integer
        return (Integer(1), self)

    # ------------------------------------------------------------------
    # calculus and numerics

    def diff(self, *symbols: Union[Symbol, int, tuple[Symbol, int]]) -> Expr:
        """The derivative (:func:`~sympy_extras.core.function.diff`)."""
        from .function import diff
        return diff(self, *symbols)

    def expand(self) -> Expr:
        """The expanded form (:func:`~sympy_extras.core.function.expand`)."""
        from .function import expand
        return expand(self)

    def evalf(self, n: int = 15, subs: Optional[SubsMapping] = None) -> Expr:
        """The numerical value to ``n`` digits (:mod:`~sympy_extras.core.evalf`)."""
        from .evalf import evalf
        return evalf(self, n, subs)

    n = evalf

    # ------------------------------------------------------------------
    # assumptions

    def _ask(self, name: str) -> Truth:
        from .assumptions import ask_structural
        return ask_structural(self, name)

    @property
    def is_number(self) -> bool:
        """Whether the expression is a constant made of numbers only."""
        from .assumptions import is_number
        return is_number(self)

    @property
    def is_commutative(self) -> Truth:
        return self._ask('commutative')

    @property
    def is_real(self) -> Truth:
        return self._ask('real')

    @property
    def is_extended_real(self) -> Truth:
        return self._ask('extended_real')

    @property
    def is_complex(self) -> Truth:
        return self._ask('complex')

    @property
    def is_imaginary(self) -> Truth:
        return self._ask('imaginary')

    @property
    def is_integer(self) -> Truth:
        return self._ask('integer')

    @property
    def is_rational(self) -> Truth:
        return self._ask('rational')

    @property
    def is_irrational(self) -> Truth:
        return self._ask('irrational')

    @property
    def is_positive(self) -> Truth:
        return self._ask('positive')

    @property
    def is_negative(self) -> Truth:
        return self._ask('negative')

    @property
    def is_nonnegative(self) -> Truth:
        return self._ask('nonnegative')

    @property
    def is_nonpositive(self) -> Truth:
        return self._ask('nonpositive')

    @property
    def is_zero(self) -> Truth:
        return self._ask('zero')

    @property
    def is_nonzero(self) -> Truth:
        return self._ask('nonzero')

    @property
    def is_even(self) -> Truth:
        return self._ask('even')

    @property
    def is_odd(self) -> Truth:
        return self._ask('odd')

    @property
    def is_prime(self) -> Truth:
        return self._ask('prime')

    @property
    def is_finite(self) -> Truth:
        return self._ask('finite')

    @property
    def is_infinite(self) -> Truth:
        return self._ask('infinite')


class AtomicExpr(Atom, Expr):
    """An expression which is a leaf: a number, a symbol, a constant."""

    __slots__ = ()

    def sort_key(self) -> SortKey:
        from .numbers import Integer
        return (self.class_key(), (1, ()), str(self), Integer(1).sort_key(), Fraction(1))
