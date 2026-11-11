"""The SymPy face of the quantified formulas
(:mod:`sympy_extras.assumptions.quantifiers`).

:mod:`~sympy_extras.assumptions.quantifiers` is written on the core. Code
on SymPy objects uses this module instead: :class:`ForAll` and
:class:`Exists` are SymPy Booleans (which :mod:`sympy_extras.interop`
converts to and from the nodes of the core) and can be combined with
SymPy's Boolean operators; :func:`prenex` takes and returns SymPy objects,
converted at the boundary around the native function.

Examples
========

>>> from sympy.abc import x, y
>>> from sympy_extras.assumptions.quantifiers_sympy import ForAll, Exists, prenex
>>> prenex(~ForAll(x, Exists(y, x < y)))
([('exists', x), ('forall', y)], x >= y)
"""
from __future__ import annotations

from typing import Iterable, Union

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse
from sympy.printing.printer import Printer

from sympy_extras._typing_sympy import QuantifierPrefix, as_boolean, as_symbol
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import quantifiers as native

__all__ = ['Quantifier', 'ForAll', 'Exists', 'prenex', 'native_quantifier', 'sympy_quantifier']


class Quantifier(Boolean):
    """Base class of :class:`ForAll` and :class:`Exists`.

    ``Quantifier(variables, formula)`` binds one symbol or a sequence of
    symbols in ``formula``. The constructor does not evaluate: the trivial
    cases (a constant formula, variables which do not occur in the formula)
    are reduced by :meth:`~sympy.core.basic.Basic.simplify`, by
    :func:`prenex` and by the functions taking quantified formulas.
    """

    #: 'forall' or 'exists' (``kind`` is taken by SymPy's Basic)
    quantifier: str = ''

    def __new__(cls, variables: Union[Symbol, Iterable[Symbol]],
                formula: Union[Boolean, bool]) -> Quantifier:
        if isinstance(variables, (list, tuple, set, frozenset, Tuple)):
            raw = tuple(variables)
        else:
            raw = (variables,)
        variables = tuple(sympify(v) for v in raw)
        for v in variables:
            if not isinstance(v, Symbol):
                raise TypeError("quantified variables must be symbols, got %s" % (v,))
        if len(set(variables)) != len(variables):
            raise ValueError("repeated quantified variable in %s" % (variables,))
        if not variables:
            raise ValueError("a quantifier needs at least one variable")
        formula_ = as_boolean(formula)
        result = Basic.__new__(cls, Tuple(*variables), formula_)
        assert isinstance(result, Quantifier)
        return result

    @property
    def variables(self) -> tuple[Symbol, ...]:
        """The quantified variables."""
        args0 = self.args[0]
        assert isinstance(args0, Tuple)
        return tuple(args0)

    @property
    def formula(self) -> Boolean:
        """The quantified formula."""
        return as_boolean(self.args[1])

    @property
    def free_symbols(self) -> set[Basic]:
        return self.formula.free_symbols - set(self.variables)

    @property
    def bound_symbols(self) -> list[Symbol]:
        return list(self.variables)

    @property
    def binary_symbols(self) -> set[Basic]:
        return self.formula.binary_symbols - set(self.variables)

    def _sympystr(self, printer: Printer) -> str:
        variables = self.variables
        v = printer._print(variables[0]) if len(variables) == 1 else \
            "(%s)" % ", ".join(printer._print(x) for x in variables)
        return "%s(%s, %s)" % (type(self).__name__, v, printer._print(self.formula))

    _sympyrepr = _sympystr

    def _latex(self, printer: Printer) -> str:
        symbol = r"\forall" if self.quantifier == 'forall' else r"\exists"
        return r"%s %s \, %s" % (symbol,
            ", ".join(printer._print(x) for x in self.variables),
            printer._print(self.formula))

    def _eval_simplify(self, **kwargs: object) -> Boolean:
        """The formula itself when it is a constant or does not contain
        the quantified variables; the variables which do not occur in the
        formula are dropped."""
        formula = self.formula.simplify(**kwargs)
        if isinstance(formula, (BooleanTrue, BooleanFalse)):
            return formula
        occurring = tuple(v for v in self.variables if v in formula.free_symbols)
        if not occurring:
            return formula
        return type(self)(occurring, formula)

    def _eval_subs(self, old: Basic, new: Basic) -> Boolean:
        if old in self.variables:
            return self
        return type(self)(self.variables, self.formula._subs(old, new))


class ForAll(Quantifier):
    """Universal quantification, ``ForAll(x, formula)``.

    Examples
    ========

    >>> from sympy.abc import b, c, x
    >>> from sympy_extras.assumptions.quantifiers_sympy import ForAll
    >>> from sympy_extras.assumptions.resolve_sympy import resolve
    >>> f = ForAll(x, x**2 + b*x + c > 0)
    >>> f
    ForAll(x, b*x + c + x**2 > 0)
    >>> sorted(f.free_symbols, key=str)
    [b, c]
    >>> resolve(f)
    b**2 < 4*c
    """
    quantifier = 'forall'


class Exists(Quantifier):
    """Existential quantification, ``Exists(x, formula)``.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import a, b, x
    >>> from sympy_extras.assumptions.quantifiers_sympy import Exists
    >>> from sympy_extras.assumptions.resolve_sympy import resolve
    >>> resolve(Exists(x, Eq(x**2 + a*x + b, 0)))
    a**2 >= 4*b
    >>> resolve(Exists([a, b], Eq(x**2 + a*x + b, 0) & (a > 0)))
    True
    """
    quantifier = 'exists'


def prenex(formula: Union[Boolean, bool]) -> tuple[QuantifierPrefix, Boolean]:
    """Prenex normal form of a formula with quantifiers
    (:func:`sympy_extras.assumptions.quantifiers.prenex` on SymPy objects).

    Returns ``(prefix, matrix)`` where ``prefix`` is a list of pairs
    ``(kind, variable)`` with ``kind`` ``'forall'`` or ``'exists'``,
    outermost first, and ``matrix`` is quantifier-free. Quantifiers under
    ``Not`` are flipped and quantifiers under ``And``, ``Or``, ``Implies``,
    ``Equivalent`` and ``Xor`` are moved in front, renaming the bound
    variables when they clash.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions.quantifiers_sympy import ForAll, Exists, prenex
    >>> prenex(~ForAll(x, Exists(y, x < y)))
    ([('exists', x), ('forall', y)], x >= y)
    >>> prefix, matrix = prenex(ForAll(x, x > 0) & Exists(x, x < 0))
    >>> prefix
    [('exists', x), ('forall', _x)]
    >>> matrix
    (_x > 0) & (x < 0)
    """
    prefix, matrix = native.prenex(_core_boolean(as_boolean(formula)))
    return [(k, as_symbol(to_sympy(v))) for k, v in prefix], as_boolean(to_sympy(matrix))


def _core_boolean(b: Boolean) -> CoreBoolean:
    converted = from_sympy(b)
    if not isinstance(converted, CoreBoolean):
        raise TypeError("a Boolean is expected, got %s" % (b,))
    return converted


def native_quantifier(q: Quantifier) -> native.Quantifier:
    """The node of the core of a SymPy quantifier (used by
    :func:`sympy_extras.interop.from_sympy`)."""
    variables: list[CoreSymbol] = []
    for v in q.variables:
        converted = from_sympy(v)
        if not isinstance(converted, CoreSymbol):
            raise TypeError("%s is not a symbol" % (v,))
        variables.append(converted)
    formula = _core_boolean(q.formula)
    if isinstance(q, ForAll):
        return native.ForAll(variables, formula)
    return native.Exists(variables, formula)


def sympy_quantifier(q: native.Quantifier) -> Quantifier:
    """The SymPy quantifier of a node of the core (used by
    :func:`sympy_extras.interop.to_sympy`)."""
    variables = [as_symbol(to_sympy(v)) for v in q.variables]
    formula = as_boolean(to_sympy(q.formula))
    if isinstance(q, native.ForAll):
        return ForAll(variables, formula)
    return Exists(variables, formula)
