"""Algebraic expressions.

:class:`Expr` is the base class of everything which can be added and
multiplied. Its operators call the evaluating functions of the package
(``x + x`` is ``add(x, x)``, which is ``2*x``), so that expressions written
with operators behave as in SymPy, while the classes ``Add``, ``Mul`` and
``Pow`` build unevaluated nodes. The methods are SymPy's
(``as_coeff_Mul``, ``as_base_exp``, ``as_independent``, ``diff``,
``expand``, ``evalf``, the ``is_*`` assumptions, ...) with the same
meanings [1]_. The structural ones are written here; the ones which are
algorithms of other packages of SymPy (``series``, ``limit``,
``simplify``, ``factor``, ``integrate``, ``as_poly``, ...) convert the
expression to SymPy, call SymPy, and convert the result back.

References
==========

.. [1] SymPy Development Team, *SymPy documentation: Expr*,
       https://docs.sympy.org/latest/modules/core.html#module-sympy.core.expr
"""
from __future__ import annotations

from fractions import Fraction
from typing import Final, Iterator, Literal, Optional, Sequence, TYPE_CHECKING, Union

import mpmath

import sympy

from .basic import Atom, Basic, RefineArg, SubsArg, SubsMapping
from .cache import cacheit
from .kind import Kind, NumberKind, UndefinedKind
from .sorting import ArgsKey, ClassKey, SortKey

if TYPE_CHECKING:
    from .assumptions import Truth
    from .logic import Boolean
    from .basic import ReplacementRule
    from .function import DiffSpec, FunctionClass
    from .numbers import Number
    from .printing import StrPrinter
    from .symbol import Symbol

__all__ = ['Expr', 'AtomicExpr', 'ExprLike', 'UnevaluatedExpr', 'as_expr', 'to_expr', 'terms_of',
    'factors_of', 'IntegrationSpec', 'TermInfo']

#: what the operators and the evaluating functions accept besides ``Expr``
ExprLike = Union['Expr', int, float, Fraction]
#: an argument of :meth:`Expr.integrate`: a symbol or ``(symbol, a, b)``
IntegrationSpec = Union['Symbol', tuple['Symbol', ExprLike, ExprLike]]
#: an entry of :meth:`Expr.as_terms`: the complex coefficient, the
#: exponents of the generators, the noncommutative factors
TermInfo = tuple[tuple[float, float], tuple[int, ...], tuple['Expr', ...]]


def as_expr(b: Basic) -> Expr:
    """``b``, which must be an :class:`Expr` (narrowing at the boundary)."""
    if not isinstance(b, Expr):
        raise TypeError("an expression is expected, got %s" % (b,))
    return b


def to_expr(a: ExprLike) -> Expr:
    """``a`` as an expression: Python numbers are converted."""
    if isinstance(a, Expr):
        return a
    from .sympify import sympify
    return as_expr(sympify(a))


def terms_of(e: Expr) -> tuple[Expr, ...]:
    """The terms of a sum, ``(e,)`` for anything else."""
    from .add import Add
    return e.args if isinstance(e, Add) else (e,)


def factors_of(e: Expr) -> tuple[Expr, ...]:
    """The factors of a product, ``(e,)`` for anything else."""
    from .mul import Mul
    return e.args if isinstance(e, Mul) else (e,)


def _is_number_like(other: object) -> bool:
    return isinstance(other, (Expr, int, float, Fraction)) and not isinstance(other, bool)


_ONE: Optional['Number'] = None


def _one() -> 'Number':
    """The number ``1`` (made once; ``numbers`` imports this module)."""
    global _ONE
    if _ONE is None:
        from .numbers import Integer
        _ONE = Integer(1)
    return _ONE


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

    __slots__ = ('_facts_cache',)

    is_scalar = True
    #: the operators of an expression defer to an operand of higher
    #: priority (the mechanism of ``sympy.core.decorators``)
    _op_priority: float = 10.0

    def _after_init(self) -> None:
        self._facts_cache: Optional[dict[str, bool]] = None

    @property
    def kind(self) -> Kind:
        if not self.args:
            return NumberKind if self.is_commutative is True else UndefinedKind
        if all(a.kind == NumberKind for a in self.args):
            return NumberKind
        return UndefinedKind

    # ------------------------------------------------------------------
    # typed versions of the tree methods

    def doit(self) -> Expr:
        return as_expr(super().doit())

    def xreplace(self, rule: Union[ReplacementRule, SubsMapping]) -> Expr:
        from .basic import _as_rule
        return as_expr(self._xreplace(_as_rule(rule)))

    def subs(self, *args: SubsArg) -> Expr:
        return as_expr(super().subs(*args))

    def copy(self) -> Expr:
        return as_expr(super().copy())

    def as_dummy(self) -> Expr:
        return as_expr(super().as_dummy())

    # ------------------------------------------------------------------
    # arithmetic: the operators evaluate

    def _priority_operand(self, other: ExprLike) -> Optional[Expr]:
        """``other`` when its operators take precedence over those of ``self``."""
        if isinstance(other, Expr) and other._op_priority > self._op_priority:
            return other
        return None

    def __add__(self, other: ExprLike) -> Expr:
        from .add import add
        if not _is_number_like(other):
            return NotImplemented
        deferred = self._priority_operand(other)
        if deferred is not None:
            return deferred.__radd__(self)
        return add(self, other)

    def __radd__(self, other: ExprLike) -> Expr:
        from .add import add
        if not _is_number_like(other):
            return NotImplemented
        return add(other, self)

    def __sub__(self, other: ExprLike) -> Expr:
        from .add import sub
        if not _is_number_like(other):
            return NotImplemented
        deferred = self._priority_operand(other)
        if deferred is not None:
            return deferred.__rsub__(self)
        return sub(self, other)

    def __rsub__(self, other: ExprLike) -> Expr:
        from .add import sub
        if not _is_number_like(other):
            return NotImplemented
        return sub(other, self)

    def __mul__(self, other: ExprLike) -> Expr:
        from .mul import mul
        if not _is_number_like(other):
            return NotImplemented
        deferred = self._priority_operand(other)
        if deferred is not None:
            return deferred.__rmul__(self)
        return mul(self, other)

    def __rmul__(self, other: ExprLike) -> Expr:
        from .mul import mul
        if not _is_number_like(other):
            return NotImplemented
        return mul(other, self)

    def __truediv__(self, other: ExprLike) -> Expr:
        from .mul import div
        if not _is_number_like(other):
            return NotImplemented
        deferred = self._priority_operand(other)
        if deferred is not None:
            return deferred.__rtruediv__(self)
        return div(self, other)

    def __rtruediv__(self, other: ExprLike) -> Expr:
        from .mul import div
        if not _is_number_like(other):
            return NotImplemented
        return div(other, self)

    def __pow__(self, other: ExprLike) -> Expr:
        from .power import pow
        if not _is_number_like(other):
            return NotImplemented
        deferred = self._priority_operand(other)
        if deferred is not None:
            return deferred.__rpow__(self)
        return pow(self, other)

    def __rpow__(self, other: ExprLike) -> Expr:
        from .power import pow
        if not _is_number_like(other):
            return NotImplemented
        return pow(other, self)

    def __mod__(self, other: ExprLike) -> Expr:
        from .mod import mod
        if not _is_number_like(other):
            return NotImplemented
        return mod(self, other)

    def __rmod__(self, other: ExprLike) -> Expr:
        from .mod import mod
        if not _is_number_like(other):
            return NotImplemented
        return mod(other, self)

    def __floordiv__(self, other: ExprLike) -> Expr:
        from sympy_extras.functions.elementary.integers import floor
        from .mul import div
        if not _is_number_like(other):
            return NotImplemented
        return floor(div(self, other))

    def __rfloordiv__(self, other: ExprLike) -> Expr:
        from sympy_extras.functions.elementary.integers import floor
        from .mul import div
        if not _is_number_like(other):
            return NotImplemented
        return floor(div(other, self))

    def __neg__(self) -> Expr:
        from .mul import neg
        return neg(self)

    def __pos__(self) -> Expr:
        return self

    def __abs__(self) -> Expr:
        from sympy_extras.functions.elementary.complexes import abs_
        return abs_(self)

    def __lt__(self, other: ExprLike) -> Boolean:
        from .relational import lt
        return lt(self, other)

    def __le__(self, other: ExprLike) -> Boolean:
        from .relational import le
        return le(self, other)

    def __gt__(self, other: ExprLike) -> Boolean:
        from .relational import gt
        return gt(self, other)

    def __ge__(self, other: ExprLike) -> Boolean:
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

    def __round__(self, n: Optional[int] = None) -> Expr:
        return self.round(n)

    # ------------------------------------------------------------------
    # ordering

    def sort_key(self) -> SortKey:
        coeff, rest = self.as_coeff_Mul()
        base, exp = rest.as_base_exp() if rest.is_Pow else (rest, None)
        args: ArgsKey
        if isinstance(base, Atom):
            args = (1, (base._sort_name(),))
        else:
            ordered_args = base.as_ordered_terms() if base.is_Add else \
                base.as_ordered_factors() if base.is_Mul else list(base.args)
            args = (len(ordered_args), tuple(a.sort_key() for a in ordered_args))
        exp_key: SortKey = _one().sort_key() if exp is None else exp.sort_key()
        return (base._sort_class_key(), args, exp_key, coeff.sort_value())

    def _sort_name(self) -> str:
        """The name of an atom in its sort key."""
        return str(self)

    def _sort_class_key(self) -> ClassKey:
        """The class part of the sort key (the rank of a function of SymPy)."""
        return self.class_key()

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
        (1/2, 1)
        """
        return (self, _one())

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

    def normal(self) -> Expr:
        """The expression as one fraction (``as_numer_denom`` recombined).

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (x/y + 1).normal()
        (x + y)/y
        """
        # PORTED FROM SYMPY: sympy/core/expr.py:Expr.normal
        from .mul import div
        from .numbers import Integer, Number
        from .power import pow
        n, d = self.as_numer_denom()
        if isinstance(d, Number):
            if d == 1:
                return n
            # a numerical denominator stays outside: (2*x + 1)/2
            from sympy_extras.simplify.radsimp import unevaluated_mul
            return unevaluated_mul(n, pow(d, Integer(-1)))
        return div(n, d)

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
        if self.is_Number and self == 0:
            return (self, self)  # SymPy: (0, 0)
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

    # PORTED FROM SYMPY: sympy/core/expr.py:Expr.as_coeff_exponent
    def as_coeff_exponent(self, x: Expr) -> tuple[Expr, Expr]:
        """``(c, e)`` with ``self == c*x**e`` and ``c`` free of ``x`` (after
        collecting in ``x``), or ``(self, 0)``.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y, z = symbols('x y z')
        >>> (3*x**2*y).as_coeff_exponent(x), (3/x).as_coeff_exponent(x), (x**2 + x).as_coeff_exponent(x)
        ((3*y, 2), (3, -1), (x**2 + x, 0))
        >>> (x**2*y + x**2*z).as_coeff_exponent(x)
        (y + z, 2)
        """
        from .mul import Mul
        from .numbers import Integer
        from sympy_extras.simplify.radsimp import collect
        s = collect(self, x)
        c: Expr
        p: list[Expr]
        if isinstance(s, Mul):
            # SymPy's Mul.as_coeff_mul(x): the raw product of the free factors
            free = [f for f in s.args if isinstance(f, Expr) and not f.has(x)]
            p = [f for f in s.args if isinstance(f, Expr) and f.has(x)]
            c = Integer(1) if not free else free[0] if len(free) == 1 else Mul(*free)
        elif s.has(x):
            c, p = Integer(1), [s]
        else:
            c, p = s, []
        if len(p) == 1:
            b, e = p[0].as_base_exp()
            if b == x:
                return c, e
        return s, Integer(0)

    def as_coefficient(self, expr: ExprLike) -> Optional[Expr]:
        """The coefficient of ``expr`` when ``self`` is ``expr`` times
        something free of ``expr``, else ``None``.

        Examples
        ========

        >>> from sympy_extras.core import symbols, E
        >>> x, y = symbols('x y')
        >>> (3*x*y).as_coefficient(x), (2*E*x).as_coefficient(E), (x + y).as_coefficient(x)
        (3*y, 2*x, None)
        """
        e = to_expr(expr)
        r = self.extract_multiplicatively(e)
        if r is not None and not r.has(e):
            return r
        return None

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
        """The factors of a product in canonical order, a negative
        coefficient split into ``-1`` and its absolute value, as SymPy.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (-3*x*y/4).as_ordered_factors(), (2*x).as_ordered_factors(), x.as_ordered_factors()
        ([-1, 3/4, x, y], [2, x], [x])
        """
        from .numbers import Integer, Number
        from .sorting import default_sort_key
        factors = list(factors_of(self))
        if len(factors) > 1 and isinstance(factors[0], Number) and factors[0].is_negative is True and factors[0] != -1:
            factors[0:1] = [Integer(-1), factors[0].number_mul(Integer(-1))]
        return sorted(factors, key=default_sort_key)

    def as_terms(self) -> tuple[list[tuple[Expr, TermInfo]], list[Expr]]:
        """The terms of a sum with their complex coefficient, the exponents
        of the generators and their noncommutative factors, and the
        generators (SymPy's ``as_terms``).

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (2*x*y + 3).as_terms()
        ([(3, ((3.0, 0.0), (0, 0), ())), (2*x*y, ((2.0, 0.0), (1, 1), ()))], [x, y])
        """
        from .numbers import Integer, Number
        from .sorting import default_sort_key
        from .evalf import to_complex
        gens: set[Expr] = set()
        rows: list[tuple[Expr, complex, dict[Expr, int], tuple[Expr, ...]]] = []
        for term in terms_of(self):
            coeff: Expr = Integer(1)
            powers: dict[Expr, int] = {}
            noncommutative: list[Expr] = []
            for f in factors_of(term):
                if isinstance(f, Number) or f.is_number:
                    from .mul import mul
                    coeff = mul(coeff, f)
                elif f.is_commutative is False:
                    noncommutative.append(f)
                else:
                    from .printing import _decompose_power
                    b, n = _decompose_power(f)
                    powers[b] = powers.get(b, 0) + n
            gens.update(powers)
            rows.append((term, to_complex(coeff), powers, tuple(noncommutative)))
        ordered = sorted(gens, key=default_sort_key)
        result: list[tuple[Expr, TermInfo]] = []
        for term, c, powers, nc in rows:
            monom = tuple(powers.get(g, 0) for g in ordered)
            result.append((term, ((c.real, c.imag), monom, nc)))
        return (result, ordered)

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
        # SymPy's _coeff_isneg: an extended negative number (-oo*I counts)
        return c.is_extended_negative is True

    def primitive(self) -> tuple[Number, Expr]:
        """The positive rational content of the terms and the rest."""
        from .numbers import Integer
        return (Integer(1), self)

    def as_content_primitive(self, radical: bool = False, clear: bool = True) -> tuple[Number, Expr]:
        """A positive rational content and the rest.

        Examples
        ========

        >>> from sympy_extras.core import symbols
        >>> x, y = symbols('x y')
        >>> (2*x + 4*y).as_content_primitive(), (x/2 + y/3).as_content_primitive()
        ((2, x + 2*y), (1/6, 3*x + 2*y))
        >>> (-2*x).as_content_primitive(), (-x - y).as_content_primitive()
        ((2, -x), (1, -x - y))
        """
        from .numbers import Integer
        return (Integer(1), self)

    def extract_additively(self, c: ExprLike) -> Optional[Expr]:
        """``self - c`` when ``c`` is a part of ``self`` term by term (same
        signs, no larger coefficients), else ``None``.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Integer
        >>> x, y = symbols('x y')
        >>> (2*x + 3).extract_additively(x + 1), (2*x + 3).extract_additively(5), Integer(5).extract_additively(2)
        (x + 2, None, 3)
        """
        from .add import add
        from .mul import mul
        from .numbers import Integer, Number
        other = to_expr(c)
        if other.is_Number and other == 0:
            return self
        if self == other:
            return Integer(0)
        terms = self.as_coefficients_dict()
        for term in terms_of(other):
            ct, rest = term.as_coeff_Mul()
            cs = terms.get(rest)
            if cs is None:
                return None
            vs, vt = cs.exact_value(), ct.exact_value()
            if vs is None or vt is None or vt == 0:
                return None
            if (vs > 0) != (vt > 0) or abs(vs) < abs(vt):
                return None
            difference: Number = cs.number_add(ct.number_mul(Integer(-1)))
            terms[rest] = difference
        return add(*[mul(coefficient, rest) for rest, coefficient in terms.items()])

    def extract_multiplicatively(self, c: ExprLike) -> Optional[Expr]:
        """``self / c`` when ``c`` can be taken out without a change of
        form, else ``None``, with SymPy's rules: an integer coefficient is
        divided by an integer which divides it, a fraction by any rational
        (``x/3`` gives ``x/6`` for ``2``, ``x`` gives ``None``), the sign
        only out of a negative, the terms of a sum one by one after its
        content.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Integer, Rational
        >>> x, y = symbols('x y')
        >>> (6*x*y).extract_multiplicatively(2*x), (6*x*y).extract_multiplicatively(4), (x**3).extract_multiplicatively(x**2)
        (3*y, None, x)
        >>> (2*x + 4*y).extract_multiplicatively(2), (-x).extract_multiplicatively(-1), x.extract_multiplicatively(-1)
        (x + 2*y, x, None)
        >>> (x/3).extract_multiplicatively(2), x.extract_multiplicatively(Rational(1, 3)), (x/2 + y).extract_multiplicatively(2)
        (x/6, 3*x, x/4 + y/2)
        """
        from .add import Add, add
        from .mul import Mul, mul
        from .numbers import Infinity, Integer, NegativeInfinity, Number, Rational, number_from_fraction
        from .power import Pow, pow
        other = to_expr(c)
        if other.is_Number and other == 1:
            return self
        if self == other:
            return Integer(1)
        if self.is_Number and self == 0:
            return self
        if isinstance(other, Mul):
            result: Optional[Expr] = self
            for f in other.args:
                if result is None:
                    return None
                result = result.extract_multiplicatively(f)
            return result
        if isinstance(self, Add):
            content, primitive = self.primitive()
            if not (content.is_Number and content == 1):
                return Mul(content, primitive).extract_multiplicatively(other)
            if isinstance(other, Rational) and other.q != 1:
                return None  # SymPy: a fraction is not taken out of a primitive sum
            parts = [t.extract_multiplicatively(other) for t in self.args]
            if any(p is None for p in parts):
                return None
            return add(*[p for p in parts if p is not None])
        coefficient, rest = self.as_coeff_Mul()
        if isinstance(other, Number):
            if isinstance(coefficient, (Infinity, NegativeInfinity)):
                return self if other.is_positive is True else (None if other.is_negative is not True else mul(Integer(-1), self))
            vc, vo = coefficient.exact_value(), other.exact_value()
            if vc is None or vo is None or vo == 0:
                return None
            if (vo < 0 and vc > 0) or (vc.denominator == 1 and vo.denominator == 1 and vc % vo != 0):
                # the sign comes out of a negative only, an integer is divided by a divisor;
                # otherwise a factor of the product may take the number (SymPy tries each)
                if isinstance(self, Mul):
                    for i, f in enumerate(self.args):
                        if isinstance(f, Add):
                            reduced = f.extract_multiplicatively(other)
                            if reduced is not None:
                                return mul(*self.args[:i], reduced, *self.args[i + 1:])
                return None
            quotient = number_from_fraction(vc / vo)
            return quotient if rest.is_Number and rest == 1 else mul(quotient, rest)
        if isinstance(self, Pow) and isinstance(other, Pow) and self.base == other.base:
            n, d = self.exp.as_coeff_Mul(), other.exp.as_coeff_Mul()
            if n[1] == d[1] and isinstance(n[0], Rational) and isinstance(d[0], Rational):
                left = n[0].as_fraction() - d[0].as_fraction()
                if left != 0 and (left > 0) == (d[0].p > 0):  # x**3 gives x for x**2, x**-2 gives 1/x for 1/x
                    return pow(self.base, mul(number_from_fraction(left), n[1]))
            return None
        if isinstance(self, Pow) and self.base == other:
            e = self.exp.as_coeff_Mul()
            if isinstance(e[0], Rational) and e[0].as_fraction() > 1 and e[1].is_Number and e[1] == 1:
                return pow(self.base, e[0].number_add(Integer(-1)))
            return None
        if isinstance(self, Mul):
            factors = list(self.args)
            for i, f in enumerate(factors):
                if f == other:
                    return mul(*factors[:i], *factors[i + 1:])
                if isinstance(f, (Pow, Add)) or isinstance(other, Pow):
                    reduced = f.extract_multiplicatively(other)
                    if reduced is not None:
                        return mul(*factors[:i], reduced, *factors[i + 1:])
            return None
        return None

    def equals(self, other: ExprLike, failing_expression: bool = False) -> Optional[bool]:
        """Whether ``self - other`` simplifies to zero (SymPy decides what
        the structural rules of this core cannot).

        Examples
        ========

        >>> from sympy_extras.core import Symbol, sqrt
        >>> x = Symbol('x')
        >>> ((x + 1)**2).equals(x**2 + 2*x + 1), (sqrt(2)*sqrt(3)).equals(sqrt(6)), x.equals(x + 1)
        (True, True, False)
        """
        from .add import sub
        from .interop import to_sympy
        from .numbers import Number
        difference = sub(self, other)
        if isinstance(difference, Number):
            value = difference.exact_value()
            return value is not None and value == 0
        zero = difference.is_zero
        if zero is not None:
            return zero
        s = to_sympy(difference)
        if not isinstance(s, sympy.Expr):
            return None
        answer = s.equals(0)
        return answer if isinstance(answer, bool) else None

    # ------------------------------------------------------------------
    # complex structure

    def as_real_imag(self, deep: bool = True) -> tuple[Expr, Expr]:
        """The real and the imaginary parts.

        Examples
        ========

        >>> from sympy_extras.core import Symbol, I
        >>> x = Symbol('x')
        >>> (2 + 3*I).as_real_imag(), (x + 2*I).as_real_imag(), Symbol('r', real=True).as_real_imag()
        ((2, 3), (re(x), im(x) + 2), (r, 0))
        """
        from sympy_extras.functions.elementary.complexes import as_real_imag, im, re
        parts = as_real_imag(self, deep)
        return parts if parts is not None else (re(self), im(self))

    def conjugate(self) -> Expr:
        """The complex conjugate.

        Examples
        ========

        >>> from sympy_extras.core import Symbol, I
        >>> x = Symbol('x')
        >>> (x + 2*I).conjugate(), Symbol('r', real=True).conjugate(), I.conjugate()
        (conjugate(x) - 2*I, r, -I)
        """
        from sympy_extras.functions.elementary.complexes import conjugate
        return conjugate(self)

    def adjoint(self) -> Expr:
        """The adjoint: the conjugate of a commutative expression."""
        from .interop import expr_through_sympy
        if self.is_commutative is True:
            return self.conjugate()
        return expr_through_sympy(self, sympy.adjoint)

    def transpose(self) -> Expr:
        """The transpose: a commutative expression itself."""
        from .interop import expr_through_sympy
        if self.is_commutative is True:
            return self
        return expr_through_sympy(self, sympy.transpose)

    # ------------------------------------------------------------------
    # structural predicates

    def is_polynomial(self, *syms: Symbol) -> Truth:
        """Whether the expression is a polynomial in ``syms`` (all its
        symbols when none is given): sums, products and nonnegative
        integer powers. As in SymPy, a function of the symbols gives
        ``None`` by itself and ``False`` inside a sum or a product.

        Examples
        ========

        >>> from sympy_extras.core import symbols, sqrt, sympify
        >>> x, y = symbols('x y')
        >>> (x**2 + 1).is_polynomial(), (x + 1/y).is_polynomial(x), (x + 1/y).is_polynomial(), sqrt(x).is_polynomial(x)
        (True, True, False, False)
        >>> sympify('sin(x)').is_polynomial(x), sympify('sin(x) + 1').is_polynomial(x)
        (None, False)
        """
        return _is_polynomial_like(self, set(syms) or self.free_symbols, 'polynomial')

    def is_rational_function(self, *syms: Symbol) -> Truth:
        """Whether the expression is a rational function in ``syms``
        (integer powers only)."""
        return _is_polynomial_like(self, set(syms) or self.free_symbols, 'rational')

    def is_algebraic_expr(self, *syms: Symbol) -> Truth:
        """Whether the expression is algebraic in ``syms`` (rational
        powers only)."""
        return _is_polynomial_like(self, set(syms) or self.free_symbols, 'algebraic')

    def is_constant(self, *wrt: Symbol) -> Optional[bool]:
        """Whether the expression does not depend on ``wrt`` (on any symbol
        when none is given); SymPy decides the non-structural cases.

        Examples
        ========

        >>> from sympy_extras.core import symbols, Integer
        >>> x, y = symbols('x y')
        >>> Integer(2).is_constant(), (x + y).is_constant(), (x + y).is_constant(y), (x - x).is_constant()
        (True, False, False, True)
        """
        free = self.free_symbols
        if wrt:
            free = free & set(wrt)
        if not free:
            return True
        from .interop import to_sympy
        s = to_sympy(self)
        if not isinstance(s, sympy.Expr):
            return None
        answer = s.is_constant(*[to_sympy(w) for w in wrt])
        return answer if isinstance(answer, bool) else None

    def is_hypergeometric(self, k: Symbol) -> Optional[bool]:
        """Whether the expression is a hypergeometric term in ``k`` (SymPy)."""
        from .interop import to_sympy
        s = to_sympy(self)
        if not isinstance(s, sympy.Expr):
            return None
        answer = s.is_hypergeometric(to_sympy(k))
        return answer if isinstance(answer, bool) else None

    # PORTED FROM SYMPY: sympy/core/expr.py:Expr.is_meromorphic
    def is_meromorphic(self, x: Symbol, a: ExprLike) -> Optional[bool]:
        """Whether the expression is meromorphic in ``x`` at ``a`` (a Laurent
        series there): ``True``, ``False`` (a branch point or an essential
        singularity) or ``None``.

        Examples
        ========

        >>> from sympy_extras.core import Symbol, oo
        >>> from sympy_extras.functions import exp, log, sin, sqrt
        >>> x = Symbol('x')
        >>> (1/x).is_meromorphic(x, 0), sqrt(x).is_meromorphic(x, 0), exp(1/x).is_meromorphic(x, 0)
        (True, False, False)
        >>> log(x).is_meromorphic(x, 0), log(x).is_meromorphic(x, 1), sin(x).is_meromorphic(x, oo)
        (False, True, False)
        """
        from .symbol import Symbol as _Symbol
        if not isinstance(x, _Symbol):
            raise TypeError("{} should be of symbol type".format(x))
        return self._eval_is_meromorphic(x, to_expr(a))

    # PORTED FROM SYMPY: sympy/core/expr.py:Expr._eval_is_meromorphic
    def _eval_is_meromorphic(self, x: Symbol, a: Expr) -> Optional[bool]:
        if self == x:
            return True
        if x not in self.free_symbols:
            return True
        # the classes whose rule is not ported: SymPy's
        from .interop import meromorphic_node_through_sympy
        return meromorphic_node_through_sympy(self, x, a, True)[1]

    @property
    def is_comparable(self) -> bool:
        import mpmath
        from .evalf import numeric_value
        from .numbers import Infinity, NegativeInfinity
        if isinstance(self, (Infinity, NegativeInfinity)):
            return True
        if not self.is_number:
            return False
        value = numeric_value(self, 64)
        return value is not None and mpmath.im(value) == 0

    # ------------------------------------------------------------------
    # calculus and numerics

    def diff(self, *symbols: DiffSpec) -> Expr:
        """The derivative (:func:`~sympy_extras.core.function.diff`)."""
        from .function import diff
        return diff(self, *symbols)

    def expand(self, deep: bool = True, mul: bool = True, multinomial: bool = True, power_base: bool = True,
               power_exp: bool = True, log: bool = False, func: bool = False, trig: bool = False,
               complex: bool = False) -> Expr:
        """The expanded form (:func:`~sympy_extras.core.function.expand`)."""
        from .function import expand
        return expand(self, deep=deep, mul=mul, multinomial=multinomial, power_base=power_base,
                      power_exp=power_exp, log=log, func=func, trig=trig, complex=complex)

    def _numeric_node(self, prec: int) -> Optional[mpmath.mpc]:
        """The value of a numerical node of a class the core's evaluation
        does not know (``hyper``, ``meijerg``) at ``prec`` bits; ``None``
        to fall back."""
        return None

    def _eval_power_node(self, exponent: Expr) -> Optional[Expr]:
        """SymPy's ``base._eval_power(exponent)``: the power of this node
        when its class has its own rule (``exp_polar(a)**b`` is
        ``exp_polar(a*b)``); ``None`` for the general rules."""
        return None

    def _eval_derivative_node(self, s: Symbol) -> Optional[Expr]:
        """The derivative of a node of a class ``diff`` does not know
        (``hyper``, ``meijerg``); ``None`` for ``Derivative(self, s)``."""
        return None

    def evalf(self, n: int = 15, subs: Optional[SubsMapping] = None) -> Expr:
        """The numerical value to ``n`` digits (:mod:`~sympy_extras.core.evalf`)."""
        from .evalf import evalf
        return evalf(self, n, subs)

    n = evalf

    def round(self, n: Optional[int] = None) -> Expr:
        """The value rounded to ``n`` decimal places (an ``Integer`` when
        ``n`` is ``None``), the parts of a complex number separately.

        Examples
        ========

        >>> from sympy_extras.core import Float, Rational, pi, I
        >>> Float(2.5).round(), Rational(7, 2).round(), pi.round(3), (Float(2.5) + 3*I).round()
        (2, 4, 3.142, 2 + 3*I)
        """
        import mpmath
        from .add import add
        from .evalf import numeric_value
        from .mul import mul
        from .numbers import Float, I, Integer, number_from_fraction
        value = numeric_value(self, 64 + 4 * (abs(n) if n else 0))
        if value is None:
            raise TypeError("cannot round the symbolic expression %s" % (self,))

        def rounded(part: Fraction) -> Expr:
            if n is None or n <= 0:
                scale = Fraction(10) ** (-(n or 0))
                return number_from_fraction(round(part / scale) * scale)
            digits = len(str(abs(int(part)))) + n
            from .numbers import dps_to_prec
            return Float(number_from_fraction(round(part, n)), precision=dps_to_prec(digits))
        from .evalf import mpf_to_fraction
        real = rounded(mpf_to_fraction(mpmath.re(value)))
        imaginary = mpf_to_fraction(mpmath.im(value))
        if imaginary == 0:
            return real
        return add(real, mul(rounded(imaginary), I)) if real != Integer(0) else mul(rounded(imaginary), I)

    # ------------------------------------------------------------------
    # series (the algorithms of sympy.series, through the conversion)

    def series(self, x: Optional[Symbol] = None, x0: ExprLike = 0, n: int = 6, dir: str = '+',
               logx: Optional[Expr] = None, cdir: ExprLike = 0) -> Expr:
        """The series expansion at ``x0`` up to ``O((x - x0)**n, (x, x0))``
        (SymPy's ``series``; ``lseries`` gives the terms one by one).

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import cos, exp
        >>> x = Symbol('x')
        >>> cos(x).series(x, 0, 6), exp(x).series(x, 1, 3)
        (1 - x**2/2 + x**4/24 + O(x**6), E + E*(x - 1) + E*(x - 1)**2/2 + O((x - 1)**3, (x, 1)))
        >>> (1/(1 - x)).series(x, 0, 3).removeO(), (1/x).series(x, 0, 3)
        (x**2 + x + 1, 1/x)
        """
        from .series import series
        return series(self, x, to_expr(x0), n, dir, logx, to_expr(cdir))

    def nseries(self, x: Optional[Symbol] = None, x0: ExprLike = 0, n: int = 6, dir: str = '+',
                logx: Optional[Expr] = None, cdir: ExprLike = 0) -> Expr:
        """The series computed by the ``_eval_nseries`` of the classes (at
        zero from the right; ``series`` otherwise), which may have fewer
        terms than asked for, with a correct order term.

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import log, sin
        >>> x = Symbol('x')
        >>> sin(x).nseries(x, 0, 6), log(x + 1).nseries(x, 0, 5)
        (x - x**3/6 + x**5/120 + O(x**6), x - x**2/2 + x**3/3 - x**4/4 + O(x**5))
        """
        x0e = to_expr(x0)
        if x is not None and x not in self.free_symbols:
            return self
        if x is None or not (x0e.is_Number and x0e == 0) or dir != '+':
            return self.series(x, x0e, n, dir, cdir=cdir)
        return self._eval_nseries(x, n, logx, to_expr(cdir))

    def lseries(self, x: Optional[Symbol] = None, x0: ExprLike = 0, dir: str = '+',
                logx: Optional[Expr] = None, cdir: ExprLike = 0) -> Iterator[Expr]:
        """The terms of the series one by one (a generator, infinite for
        an infinite series).

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import cos
        >>> x = Symbol('x')
        >>> terms = cos(x).lseries(x)
        >>> next(terms), next(terms), next(terms)
        (1, -x**2/2, x**4/24)
        """
        from .series import lseries
        return lseries(self, x, to_expr(x0), dir, logx, to_expr(cdir))

    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        """The terms of the series up to ``O(x**n)`` at zero from the
        right (overridden by the classes; the classes whose expansion is
        not ported use SymPy's)."""
        from .interop import nseries_node_through_sympy
        return nseries_node_through_sympy(self, x, n, logx, cdir)

    # PORTED FROM SYMPY: sympy/core/expr.py:Expr._eval_lseries
    def _eval_lseries(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Iterator[Expr]:
        """The terms of the series one by one, by series of increasing
        orders."""
        from .numbers import Integer
        k = 0
        series = self._eval_nseries(x, k, logx, cdir)
        while series.is_Order:
            k += 1
            series = self._eval_nseries(x, k, logx, cdir)
        e = series.removeO()
        yield e
        if e.is_Number and e == 0:
            return
        while True:
            while True:
                k += 1
                series = self._eval_nseries(x, k, logx, cdir).removeO()
                if e != series:
                    break
                if (series - self).cancel() == Integer(0):
                    return
            yield series - e
            e = series

    def aseries(self, x: Optional[Symbol] = None, n: int = 6, bound: int = 0, hir: bool = False) -> Expr:
        """The asymptotic series at infinity (SymPy's ``aseries``)."""
        from .interop import expr_through_sympy, to_sympy
        sx = None if x is None else to_sympy(x)
        return expr_through_sympy(self, lambda s: s.aseries(sx, n, bound, hir))

    def fps(self, x: Optional[Symbol] = None, x0: ExprLike = 0, dir: int = 1) -> Expr:
        """The formal power series (SymPy's ``fps``, wrapped)."""
        from .interop import expr_through_sympy, to_sympy
        sx = None if x is None else to_sympy(x)
        return expr_through_sympy(self, lambda s: sympy.fps(s, sx, to_sympy(to_expr(x0)), dir))

    def fourier_series(self, limits: Optional[tuple[Symbol, ExprLike, ExprLike]] = None) -> Expr:
        """The Fourier series (SymPy's ``fourier_series``, wrapped)."""
        from .interop import expr_through_sympy, to_sympy
        bounds = None if limits is None else (to_sympy(limits[0]), to_sympy(to_expr(limits[1])), to_sympy(to_expr(limits[2])))
        return expr_through_sympy(self, lambda s: sympy.fourier_series(s, bounds))

    def limit(self, x: Symbol, xlim: ExprLike, dir: str = '+') -> Expr:
        """The limit as ``x`` goes to ``xlim`` (:func:`~sympy_extras.core.limits.limit`).

        Examples
        ========

        >>> from sympy_extras.core import Symbol, oo
        >>> from sympy_extras.functions import sin
        >>> x = Symbol('x')
        >>> (sin(x)/x).limit(x, 0), (1/x).limit(x, 0, '-'), (1/x).limit(x, oo)
        (1, -oo, 0)
        """
        from .limits import limit
        return limit(self, x, xlim, dir)

    def as_leading_term(self, *symbols: Symbol, logx: Optional[Expr] = None, cdir: ExprLike = 0) -> Expr:
        """The leading (nonzero) term of the series at zero in the given
        symbols (one after the other), from the ``_eval_as_leading_term``
        of the classes.

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> from sympy_extras.functions import sin
        >>> x, y = Symbol('x'), Symbol('y')
        >>> (x + x**2).as_leading_term(x), (x + x*y).as_leading_term(x), (x + y + 1).as_leading_term(x)
        (x, x*y + x, y + 1)
        >>> (1/(1 + x)).as_leading_term(x), ((1 + x)/x).as_leading_term(x), (sin(x) + x).as_leading_term(x)
        (1, 1/x, 2*x)
        """
        return _as_leading_term(self, symbols, logx, to_expr(cdir))

    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        """The leading term at zero (overridden by the classes; the
        classes whose leading term is not ported use SymPy's, the others
        are their own leading term)."""
        from .interop import leading_term_node_through_sympy
        found = leading_term_node_through_sympy(self, x, logx, cdir)
        return self if found is None else found

    # PORTED FROM SYMPY: sympy/core/expr.py:Expr.leadterm
    def leadterm(self, x: Symbol, logx: Optional[Expr] = None, cdir: ExprLike = 0) -> tuple[Expr, Expr]:
        """The leading term as ``(c, e)`` with the term ``c*x**e``.

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> x, y = Symbol('x'), Symbol('y')
        >>> (x**2 + x**3).leadterm(x), (x*y + x**2*y).leadterm(x), (x + x*y).leadterm(x), (1/x + 1).leadterm(x)
        ((1, 2), (y, 1), (y + 1, 1), (1, -1))
        """
        from .symbol import Dummy
        from sympy_extras.functions.elementary.exponential import log
        lt = self.as_leading_term(x, logx=logx, cdir=cdir)
        d = Dummy('logx')
        logarithm = log(x)
        if lt.has(logarithm):
            lt = lt.subs(logarithm, d)
        c, e = lt.as_coeff_exponent(x)
        if x in c.free_symbols:
            raise ValueError("cannot compute leadterm(%s, %s). The coefficient should have been free of %s "
                             "but got %s" % (self, x, x, c))
        return c.subs(d, logarithm), e

    # PORTED FROM SYMPY: sympy/core/expr.py:Expr.taylor_term
    def taylor_term(self, n: int, x: Expr, previous: Optional[Expr] = None) -> Expr:
        """The ``n``-th term of the Taylor series at zero (by ``n``
        derivatives; the classes have faster ones using the ``previous``
        term)."""
        from .symbol import Dummy, Symbol as _Symbol
        from .numbers import Integer
        from sympy_extras.functions.combinatorial.factorials import factorial
        _x = Dummy('x')
        e = self.subs(x, _x)
        for _ in range(n):
            e = e.diff(_x)
        return e.subs(_x, x).subs(x, Integer(0)) * x**n / factorial(Integer(n))

    # PORTED FROM SYMPY: sympy/core/expr.py:Expr.dir
    def dir(self, x: Symbol, cdir: Expr) -> Expr:
        """The direction from which ``self`` approaches its value at
        ``x = 0`` along ``cdir`` (the first nonzero derivative times a
        power of ``cdir``)."""
        from .numbers import ComplexInfinity, Integer, NaN
        from sympy_extras.functions.elementary.exponential import log
        if self.is_zero is True:
            return Integer(0)
        minexp: Expr = Integer(0)
        arg: Expr = self
        coeff: Expr = Integer(0)
        while not (arg.is_Number and arg == 0):
            minexp = minexp + Integer(1)
            arg = arg.diff(x)
            coeff = arg.subs(x, Integer(0))
            if isinstance(coeff, NaN):
                coeff = arg.limit(x, Integer(0))
            if isinstance(coeff, ComplexInfinity):
                try:
                    coeff, _ = arg.leadterm(x)
                    if coeff.has(log(x)):
                        raise ValueError()
                except ValueError:
                    coeff = arg.limit(x, Integer(0))
            if coeff != Integer(0):
                break
        return coeff * cdir**minexp

    def removeO(self) -> Expr:
        """The expression without its ``Order`` terms.

        Examples
        ========

        >>> from sympy_extras.core import Symbol, Order
        >>> x = Symbol('x')
        >>> (x + Order(x**2)).removeO(), (x + Order(x**2)).getO(), (x + Order(x**2)).getn()
        (x, O(x**2), 2)
        """
        from .add import add
        return add(*[t for t in terms_of(self) if not t.is_Order])

    def getO(self) -> Optional[Expr]:
        """The ``Order`` term of a sum, ``None`` when there is none."""
        from .add import add
        orders = [t for t in terms_of(self) if t.is_Order]
        if not orders:
            return None
        return add(*orders)

    # PORTED FROM SYMPY: sympy/core/expr.py:Expr.getn
    def getn(self) -> Optional[Expr]:
        """The order of the ``Order`` term (``2`` for ``O(x**2)``), ``None``
        when there is none.

        Examples
        ========

        >>> from sympy_extras.core import Symbol, order
        >>> x = Symbol('x')
        >>> (1 + x + order(x**2)).getn(), (1 + x).getn(), order(1, x).getn()
        (2, None, 0)
        """
        from .mul import Mul
        from .numbers import Integer, Rational
        from .power import Pow
        from .symbol import Dummy, Symbol as _Symbol
        from .order import Order
        o = self.getO()
        if o is None:
            return None
        if isinstance(o, Order):
            oe = o.expr
            if oe.is_Number and oe == 1:
                return Integer(0)
            if isinstance(oe, _Symbol):
                return Integer(1)
            if isinstance(oe, Pow):
                return oe.exp
            if isinstance(oe, Mul):  # x**n*log(x)**n or x**n/log(x)**n
                for oi in oe.args:
                    if isinstance(oi, _Symbol):
                        return Integer(1)
                    if isinstance(oi, Pow):
                        syms = oi.atoms(_Symbol)
                        if len(syms) == 1:
                            v = next(iter(syms))
                            positive = oi.subs(v, Dummy('x', positive=True))
                            if isinstance(positive, Pow) and isinstance(positive.base, _Symbol) \
                                    and isinstance(positive.exp, Rational):
                                return abs(positive.exp)
        raise NotImplementedError('not sure of order of %s' % (o,))

    # ------------------------------------------------------------------
    # simplification and algebra (SymPy's algorithms through the conversion)

    def simplify(self, ratio: float = 1.7, rational: Optional[bool] = False, inverse: bool = False,
                 doit: bool = True) -> Expr:
        """The expression simplified: :func:`sympy_extras.simplify.simplify.simplify`.

        Examples
        ========

        >>> import sympy
        >>> from sympy_extras.core import Symbol, SymPyFunction
        >>> x = Symbol('x')
        >>> sin, cos = SymPyFunction(sympy.sin), SymPyFunction(sympy.cos)
        >>> (sin(x)**2 + cos(x)**2).simplify()
        1
        """
        from sympy_extras.simplify.simplify import simplify
        return simplify(self, ratio=ratio, rational=rational, inverse=inverse, doit=doit)

    def factor(self, *gens: Symbol, deep: bool = False) -> Expr:
        """SymPy's ``factor``.

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> x = Symbol('x')
        >>> (x**2 + 2*x + 1).factor()
        (x + 1)**2
        """
        from sympy_extras.polys.polytools import factor
        return factor(self, *gens, deep=deep)

    def cancel(self, *gens: Symbol) -> Expr:
        """The common factors of numerator and denominator removed
        (:func:`sympy_extras.polys.polytools.cancel`; SymPy's with ``gens``)."""
        if not gens:
            from sympy_extras.polys.polytools import cancel
            return cancel(self)
        from .interop import expr_through_sympy, to_sympy
        return expr_through_sympy(self, lambda s: sympy.cancel(s, *[to_sympy(g) for g in gens]))

    def apart(self, x: Optional[Symbol] = None) -> Expr:
        """The partial fraction decomposition
        (:func:`sympy_extras.polys.partfrac.apart`)."""
        from sympy_extras.polys.partfrac import apart
        return apart(self, x)

    def collect(self, syms: Union[Symbol, Sequence[Symbol]], evaluate: bool = True) -> Expr:
        """SymPy's ``collect``: the terms grouped by powers of ``syms``."""
        if evaluate:
            from sympy_extras.simplify.radsimp import collect
            return collect(self, syms)
        from .interop import expr_through_sympy, to_sympy
        from .symbol import Symbol
        targets = [to_sympy(syms)] if isinstance(syms, Symbol) else [to_sympy(s) for s in syms]
        return expr_through_sympy(self, lambda s: sympy.collect(s, targets, evaluate=evaluate))

    def together(self) -> Expr:
        """The fractions combined (:func:`sympy_extras.polys.rationaltools.together`)."""
        from sympy_extras.polys.rationaltools import together
        return together(self)

    def powsimp(self, deep: bool = False, combine: Literal['all', 'exp', 'base'] = 'all', force: bool = False) -> Expr:
        """Powers combined (:func:`sympy_extras.simplify.powsimp.powsimp`)."""
        from sympy_extras.simplify.powsimp import powsimp
        return powsimp(self, deep=deep, combine=combine, force=force)

    def radsimp(self) -> Expr:
        """The denominator rationalised (:func:`sympy_extras.simplify.radsimp.radsimp`)."""
        from sympy_extras.simplify.radsimp import radsimp
        return radsimp(self)

    def ratsimp(self) -> Expr:
        """A polynomial plus a proper fraction (:func:`sympy_extras.simplify.ratsimp.ratsimp`)."""
        from sympy_extras.simplify.ratsimp import ratsimp
        return ratsimp(self)

    def trigsimp(self) -> Expr:
        """Trigonometric simplification (:func:`sympy_extras.simplify.trigsimp.trigsimp`)."""
        from sympy_extras.simplify.trigsimp import trigsimp
        return trigsimp(self)

    def combsimp(self) -> Expr:
        """Combinatorial simplification (:func:`sympy_extras.simplify.gammasimp.combsimp`)."""
        from sympy_extras.simplify.gammasimp import combsimp
        return combsimp(self)

    def gammasimp(self) -> Expr:
        """Gamma function simplification (:func:`sympy_extras.simplify.gammasimp.gammasimp`)."""
        from sympy_extras.simplify.gammasimp import gammasimp
        return gammasimp(self)

    def nsimplify(self, constants: Sequence[Expr] = (), tolerance: Optional[float] = None, full: bool = False,
                  rational: Optional[bool] = None) -> Expr:
        """SymPy's ``nsimplify``: a float replaced by a simple exact number.

        Examples
        ========

        >>> from sympy_extras.core import Float
        >>> Float(0.5).nsimplify()
        1/2
        """
        from sympy_extras.simplify.simplify import nsimplify
        return nsimplify(self, tuple(constants), tolerance, full, rational)

    def integrate(self, *args: IntegrationSpec) -> Expr:
        """SymPy's ``integrate``.

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> x = Symbol('x')
        >>> (x**2).integrate(x), (x**2).integrate((x, 0, 1))
        (x**3/3, 1/3)
        """
        from .interop import expr_through_sympy, to_sympy
        specs: list[Union[sympy.Basic, tuple[sympy.Basic, sympy.Basic, sympy.Basic]]] = []
        for a in args:
            if isinstance(a, tuple):
                specs.append((to_sympy(a[0]), to_sympy(to_expr(a[1])), to_sympy(to_expr(a[2]))))
            else:
                specs.append(to_sympy(a))
        return expr_through_sympy(self, lambda s: sympy.integrate(s, *specs))

    def as_poly(self, *gens: Symbol) -> Optional[sympy.Poly]:
        """SymPy's ``Poly`` of the expression, ``None`` when it is not a
        polynomial (the ``Poly`` class belongs to ``sympy.polys``).

        Examples
        ========

        >>> from sympy_extras.core import Symbol
        >>> x = Symbol('x')
        >>> (x**2 + 1).as_poly()
        Poly(x**2 + 1, x, domain='ZZ')
        """
        from .interop import to_sympy
        s = to_sympy(self)
        if not isinstance(s, sympy.Expr):
            return None
        poly = s.as_poly(*[to_sympy(g) for g in gens])
        return poly if isinstance(poly, sympy.Poly) else None

    def invert(self, g: ExprLike, *gens: Symbol) -> Expr:
        """SymPy's ``invert``: the inverse of ``self`` modulo ``g``."""
        if not gens:
            from sympy_extras.polys.polytools import invert
            return invert(self, g)
        from .interop import expr_through_sympy, to_sympy
        return expr_through_sympy(self, lambda s: sympy.invert(s, to_sympy(to_expr(g)), *[to_sympy(v) for v in gens]))

    def extract_branch_factor(self, allow_half: bool = False) -> tuple[Expr, Expr]:
        """SymPy's ``extract_branch_factor``: ``(z, n)`` with
        ``self == z*exp_polar(2*pi*I*n)``."""
        from .interop import expr_from_sympy, to_sympy
        s = to_sympy(self)
        if not isinstance(s, sympy.Expr):
            raise TypeError("an expression is expected")
        z, k = s.extract_branch_factor(allow_half=allow_half)
        return (expr_from_sympy(z), expr_from_sympy(k))

    def rewrite(self, *args: Union[Basic, FunctionClass, type], piecewise: bool = True) -> Expr:
        return as_expr(super().rewrite(*args, piecewise=piecewise))

    def refine(self, assumption: RefineArg = True) -> Expr:
        return as_expr(super().refine(assumption))

    # ------------------------------------------------------------------
    # assumptions

    def _ask(self, name: str) -> Truth:
        cached = self._facts_cache
        if cached is None:
            from .assumptions import facts_of
            cached = facts_of(self)
        return cached.get(name)

    def _eval_facts(self) -> dict[str, bool]:
        """The facts of the node before their closure (the structural
        rules of :mod:`~sympy_extras.core.assumptions`); subclasses override it."""
        from .assumptions import constant_facts
        return constant_facts(self)

    @property
    def assumptions0(self) -> dict[str, bool]:
        """The known facts of the expression, with their consequences."""
        from .assumptions import facts_of
        return dict(facts_of(self))

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
    def is_noninteger(self) -> Truth:
        return self._ask('noninteger')

    @property
    def is_rational(self) -> Truth:
        return self._ask('rational')

    @property
    def is_irrational(self) -> Truth:
        return self._ask('irrational')

    @property
    def is_algebraic(self) -> Truth:
        return self._ask('algebraic')

    @property
    def is_transcendental(self) -> Truth:
        return self._ask('transcendental')

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
    def is_extended_positive(self) -> Truth:
        return self._ask('extended_positive')

    @property
    def is_extended_negative(self) -> Truth:
        return self._ask('extended_negative')

    @property
    def is_extended_nonnegative(self) -> Truth:
        return self._ask('extended_nonnegative')

    @property
    def is_extended_nonpositive(self) -> Truth:
        return self._ask('extended_nonpositive')

    @property
    def is_extended_nonzero(self) -> Truth:
        return self._ask('extended_nonzero')

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
    def is_composite(self) -> Truth:
        return self._ask('composite')

    @property
    def is_finite(self) -> Truth:
        return self._ask('finite')

    @property
    def is_infinite(self) -> Truth:
        return self._ask('infinite')

    @property
    def is_hermitian(self) -> Truth:
        return self._ask('hermitian')

    @property
    def is_antihermitian(self) -> Truth:
        return self._ask('antihermitian')

    @property
    def is_polar(self) -> Truth:
        return self._ask('polar')


class AtomicExpr(Atom, Expr):
    """An expression which is a leaf: a number, a symbol, a constant."""

    __slots__ = ()

    def sort_key(self) -> SortKey:
        return (self.class_key(), (1, (str(self),)), _one().sort_key(), Fraction(1))

    # PORTED FROM SYMPY: sympy/core/expr.py:AtomicExpr._eval_nseries
    def _eval_nseries(self, x: Symbol, n: int, logx: Optional[Expr], cdir: Expr) -> Expr:
        return self

    # PORTED FROM SYMPY: sympy/core/expr.py:AtomicExpr._eval_is_meromorphic
    def _eval_is_meromorphic(self, x: Symbol, a: Expr) -> Optional[bool]:
        from .numbers import ComplexInfinity
        if isinstance(self, ComplexInfinity):
            return True  # zoo is not a Number in SymPy
        if not self.is_Number:
            return True
        return self.is_finite

    def _eval_as_leading_term(self, x: Symbol, logx: Optional[Expr], cdir: Expr) -> Expr:
        return self


class UnevaluatedExpr(Expr):
    """A wrapper which keeps its content out of the evaluation: the
    evaluating functions treat it as an opaque factor, ``doit`` removes it.

    Examples
    ========

    >>> from sympy_extras.core import Symbol, UnevaluatedExpr, srepr
    >>> x = Symbol('x')
    >>> e = x*UnevaluatedExpr(x)
    >>> e, e.doit()
    (x*x, x**2)
    >>> srepr(e)
    Mul(Symbol('x'), UnevaluatedExpr(Symbol('x')))
    """

    __slots__ = ('_expr',)

    def __init__(self, expr: ExprLike) -> None:
        self._expr: Final[Expr] = to_expr(expr)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Expr, ...]:
        return (self._expr,)

    @property
    def expr(self) -> Expr:
        return self._expr

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        return UnevaluatedExpr(as_expr(args[0]))

    def doit(self) -> Expr:
        return self._expr.doit()

    @property
    def is_number(self) -> bool:
        return self._expr.is_number

    def _sympystr(self, printer: StrPrinter) -> str:
        return printer.doprint(self._expr)

    def _eval_facts(self) -> dict[str, bool]:
        return {}


# PORTED FROM SYMPY: sympy/core/expr.py:Expr.as_leading_term
@cacheit
def _as_leading_term(e: Expr, symbols: tuple[Symbol, ...], logx: Optional[Expr], cdir: Expr) -> Expr:
    """The leading term of ``e`` in the symbols, one after the other."""
    from .symbol import Symbol as _Symbol
    from sympy_extras.simplify.powsimp import powsimp
    if len(symbols) > 1:
        c = e
        for v in symbols:
            c = c.as_leading_term(v, logx=logx, cdir=cdir)
        return c
    if not symbols:
        return e
    x = symbols[0]
    if not isinstance(x, _Symbol):
        raise ValueError('expecting a Symbol but got %s' % (x,))
    if x not in e.free_symbols:
        return e
    obj = e._eval_as_leading_term(x, logx, cdir)
    return powsimp(obj, deep=True, combine='exp')


def _is_polynomial_like(e: Expr, syms: set[Symbol], kind: str) -> Truth:
    """SymPy's three-valued answer: ``None`` for a function of the symbols,
    ``all`` of the arguments of a sum or a product (``None`` counting as
    ``False``), and for a power the answer of the base combined with the
    exponent (``bool`` of it for a polynomial)."""
    from .add import Add
    from .mul import Mul
    from .numbers import Rational
    from .power import Pow
    from .symbol import Symbol
    if not (e.free_symbols & syms):
        return True
    if isinstance(e, Symbol):
        return True
    if isinstance(e, (Add, Mul)):
        return all(_is_polynomial_like(a, syms, kind) is True for a in e.args)
    if isinstance(e, Pow):
        exponent = e.exp
        if exponent.free_symbols & syms:
            return False
        base = _is_polynomial_like(e.base, syms, kind)
        if kind == 'polynomial':
            return bool(base is True and isinstance(exponent, Rational) and exponent.q == 1 and exponent.p >= 0)
        if base is not True:
            return base
        if kind == 'rational':
            return isinstance(exponent, Rational) and exponent.q == 1
        return isinstance(exponent, Rational)
    return None


