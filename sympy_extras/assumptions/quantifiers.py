"""Quantified formulas on the core.

:class:`ForAll` and :class:`Exists` are the counterparts of Mathematica's
``ForAll[x, expr]`` and ``Exists[x, expr]``. They are Boolean nodes of
:mod:`sympy_extras.core`, unevaluating like every node: ``ForAll(x, True)``
stays a ``ForAll``, and the functions :func:`forall` and :func:`exists`
evaluate (they give the formula when it is a truth value or contains none
of the variables, and drop the variables which do not occur in it).
:func:`prenex` puts a formula in prenex normal form, with all the
quantifiers in front, which is what the quantifier elimination of
:func:`~sympy_extras.assumptions.resolve.resolve` works on.

This module imports no SymPy; code on SymPy objects uses
:mod:`sympy_extras.assumptions.quantifiers_sympy`, whose classes
:mod:`sympy_extras.interop` converts to and from these.

Examples
========

>>> from sympy_extras.core import symbols
>>> from sympy_extras.assumptions.quantifiers import ForAll, Exists, forall, prenex
>>> x, y = symbols('x y')
>>> ForAll(x, Exists(y, x < y)), forall([x, y], y > 0)
(ForAll(x, Exists(y, x < y)), ForAll(y, y > 0))
>>> prenex(~ForAll(x, Exists(y, x < y)))
([('exists', x), ('forall', y)], x >= y)
"""
from __future__ import annotations

from typing import Final, Iterable, Union

from sympy_extras.core.basic import Basic
from sympy_extras.core.containers import Tuple
from sympy_extras.core.logic import (ITE, And, Boolean, BooleanAtom, BooleanLike, Equivalent, Implies, Not, Or,
    Xor, and_, implies, not_, or_, to_boolean, xor)
from sympy_extras.core.printing import StrPrinter
from sympy_extras.core.sorting import ClassKey
from sympy_extras.core.symbol import Dummy, Symbol

__all__ = ['Quantifier', 'ForAll', 'Exists', 'forall', 'exists', 'prenex', 'Prefix', 'QuantifierKind']

#: 'forall' or 'exists'
QuantifierKind = str

#: the quantifier prefix of a prenex formula: ``(kind, variable)`` pairs,
#: outermost first
Prefix = list[tuple[QuantifierKind, Symbol]]
#: the variables a quantifier accepts: one symbol or a sequence of them
Variables = Union[Symbol, Iterable[Symbol], Tuple]


def _variables(variables: Variables) -> tuple[Symbol, ...]:
    raw: tuple[Basic, ...]
    if isinstance(variables, Symbol):
        raw = (variables,)
    elif isinstance(variables, Tuple):
        raw = variables.args
    elif isinstance(variables, Iterable):
        raw = tuple(variables)
    else:
        raw = (variables,)
    result: list[Symbol] = []
    for v in raw:
        if not isinstance(v, Symbol):
            raise TypeError("quantified variables must be symbols, got %s" % (v,))
        result.append(v)
    if len(set(result)) != len(result):
        raise ValueError("repeated quantified variable in %s" % (tuple(result),))
    if not result:
        raise ValueError("a quantifier needs at least one variable")
    return tuple(result)


class Quantifier(Boolean):
    """Base class of :class:`ForAll` and :class:`Exists`.

    ``Quantifier(variables, formula)`` binds one symbol or a sequence of
    symbols in ``formula``. The constructor does not evaluate: the trivial
    cases (a constant formula, variables which do not occur in the formula)
    are reduced by :func:`forall` and :func:`exists`, :meth:`simplify`,
    :func:`prenex` and the functions taking quantified formulas.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.assumptions.quantifiers import ForAll
    >>> x, y = symbols('x y')
    >>> f = ForAll([x, y], x > y)
    >>> f.variables, f.formula, f.free_symbols, ForAll(x, x > y).free_symbols
    ((x, y), x > y, set(), {y})
    >>> ForAll(x, x > y).subs(y, 2), ForAll(x, x > y).subs(x, 2)
    (ForAll(x, x > 2), ForAll(x, x > y))
    """

    __slots__ = ('_variables', '_formula')

    #: 'forall' or 'exists'
    quantifier: QuantifierKind = 'forall'

    def __init__(self, variables: Variables, formula: BooleanLike) -> None:
        self._variables: Final[tuple[Symbol, ...]] = _variables(variables)
        self._formula: Final[Boolean] = to_boolean(formula)
        Basic.__init__(self)

    @property
    def args(self) -> tuple[Basic, ...]:
        return (Tuple(*self._variables), self._formula)

    @property
    def variables(self) -> tuple[Symbol, ...]:
        """The quantified variables."""
        return self._variables

    @property
    def formula(self) -> Boolean:
        """The quantified formula."""
        return self._formula

    @classmethod
    def class_key(cls) -> ClassKey:
        return (5, 0, cls.__name__)

    @property
    def free_symbols(self) -> set[Symbol]:
        return self._formula.free_symbols - set(self._variables)

    @property
    def bound_symbols(self) -> tuple[Symbol, ...]:
        return self._variables

    @property
    def binary_symbols(self) -> set[Basic]:
        return self._formula.binary_symbols - set(self._variables)

    def _sympystr(self, printer: StrPrinter) -> str:
        variables = self._variables
        v = printer.doprint(variables[0]) if len(variables) == 1 else \
            "(%s)" % ", ".join(printer.doprint(x) for x in variables)
        return "%s(%s, %s)" % (type(self).__name__, v, printer.doprint(self._formula))

    def _rebuild(self, args: tuple[Basic, ...]) -> Basic:
        variables, formula = args
        if not isinstance(variables, Tuple):
            raise TypeError("the variables of a quantifier are a Tuple")
        return type(self)(variables, to_boolean(formula))

    def _evaluated(self, args: tuple[Basic, ...]) -> Basic:
        variables, formula = args
        if not isinstance(variables, Tuple):
            raise TypeError("the variables of a quantifier are a Tuple")
        return _quantify(self.quantifier, _variables(variables), to_boolean(formula))

    def _subs(self, old: Basic, new: Basic) -> Basic:
        if self == old:
            return new
        if old in self._variables:
            return self
        return type(self)(self._variables, to_boolean(self._formula._subs(old, new)))

    def simplify(self) -> Basic:
        """The formula itself when it is a constant or does not contain
        the quantified variables; the variables which do not occur in the
        formula are dropped (the formula is simplified first).

        Examples
        ========

        >>> import sympy_extras.interop  # SymPy's simplify (core.bridge)
        >>> from sympy_extras.core import symbols, true
        >>> from sympy_extras.assumptions.quantifiers import ForAll
        >>> x, y, z = symbols('x y z')
        >>> ForAll(x, True).simplify(), ForAll([x, z], y > x).simplify(), ForAll(x, y > 0).simplify()
        (True, ForAll(x, x < y), y > 0)
        """
        return _quantify(self.quantifier, self._variables, to_boolean(self._formula.simplify()))


class ForAll(Quantifier):
    """Universal quantification, ``ForAll(x, formula)``; :func:`forall`
    evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.assumptions.quantifiers import ForAll
    >>> x, b, c = symbols('x b c')
    >>> f = ForAll(x, x**2 + b*x + c > 0)
    >>> f, sorted(f.free_symbols, key=str)
    (ForAll(x, b*x + c + x**2 > 0), [b, c])
    """

    __slots__ = ()

    quantifier: QuantifierKind = 'forall'


class Exists(Quantifier):
    """Existential quantification, ``Exists(x, formula)``; :func:`exists`
    evaluates.

    Examples
    ========

    >>> from sympy_extras.core import symbols, Eq
    >>> from sympy_extras.assumptions.quantifiers import Exists
    >>> x, a, b = symbols('x a b')
    >>> Exists(x, Eq(x**2 + a*x + b, 0))
    Exists(x, Eq(a*x + b + x**2, 0))
    """

    __slots__ = ()

    quantifier: QuantifierKind = 'exists'


def _quantify(kind: QuantifierKind, variables: tuple[Symbol, ...], formula: Boolean) -> Boolean:
    if isinstance(formula, BooleanAtom):
        return formula
    symbols = formula.free_symbols
    occurring = tuple(v for v in variables if v in symbols)
    if not occurring:
        return formula
    if kind == 'forall':
        return ForAll(occurring, formula)
    return Exists(occurring, formula)


def forall(variables: Variables, formula: BooleanLike) -> Boolean:
    """``ForAll(variables, formula)`` evaluated: the formula when it is a
    truth value or contains none of the variables, the variables which do
    not occur in it dropped.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.assumptions.quantifiers import forall
    >>> x, y, z = symbols('x y z')
    >>> forall(x, True), forall(x, y > 0), forall([x, z], x > y)
    (True, y > 0, ForAll(x, x > y))
    """
    return _quantify('forall', _variables(variables), to_boolean(formula))


def exists(variables: Variables, formula: BooleanLike) -> Boolean:
    """``Exists(variables, formula)`` evaluated, as :func:`forall`.

    Examples
    ========

    >>> from sympy_extras.core import symbols, false
    >>> from sympy_extras.assumptions.quantifiers import exists
    >>> x, y = symbols('x y')
    >>> exists(x, false), exists([x, y], x > 1)
    (False, Exists(x, x > 1))
    """
    return _quantify('exists', _variables(variables), to_boolean(formula))


_DUAL: Final[dict[QuantifierKind, QuantifierKind]] = {'forall': 'exists', 'exists': 'forall'}


def prenex(formula: BooleanLike) -> tuple[Prefix, Boolean]:
    """Prenex normal form of a formula with quantifiers.

    Returns ``(prefix, matrix)`` where ``prefix`` is a list of pairs
    ``(kind, variable)`` with ``kind`` ``'forall'`` or ``'exists'``,
    outermost first, and ``matrix`` is quantifier-free. Quantifiers under
    ``Not`` are flipped and quantifiers under ``And``, ``Or``, ``Implies``,
    ``Equivalent``, ``Xor`` and ``ITE`` are moved in front, renaming the
    bound variables (to dummies) when they clash.

    Examples
    ========

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.assumptions.quantifiers import ForAll, Exists, prenex
    >>> x, y = symbols('x y')
    >>> prenex(~ForAll(x, Exists(y, x < y)))
    ([('exists', x), ('forall', y)], x >= y)
    >>> prefix, matrix = prenex(ForAll(x, x > 0) & Exists(x, x < 0))
    >>> prefix
    [('exists', x), ('forall', _x)]
    >>> matrix
    (_x > 0) & (x < 0)
    """
    return _prenex(to_boolean(formula), set())


def _prenex(formula: Boolean, taken: set[Symbol]) -> tuple[Prefix, Boolean]:
    """``taken`` is the set of symbols which are already used elsewhere:
    bound variables clashing with it are renamed."""
    if isinstance(formula, Quantifier):
        inner_prefix, matrix = _prenex(formula.formula, taken | set(formula.variables))
        prefix: Prefix = []
        for v in formula.variables:
            if v not in matrix.free_symbols:
                continue
            if v in taken:
                new = Dummy(v.name, **v.assumptions0)
                matrix = to_boolean(matrix.subs(v, new))
                inner_prefix = [(k, new if u == v else u) for k, u in inner_prefix]
                v = new
            prefix.append((formula.quantifier, v))
        return prefix + inner_prefix, matrix
    if isinstance(formula, Not):
        prefix, matrix = _prenex(formula.arg, taken)
        return [(_DUAL[k], v) for k, v in prefix], not_(matrix)
    if isinstance(formula, (And, Or)):
        prefix = []
        matrices: list[Boolean] = []
        used: set[Symbol] = set(taken)
        for a in formula.args:
            used |= a.free_symbols
        for arg in formula.args:
            p, m = _prenex(arg, used)
            used |= {v for _, v in p}
            prefix.extend(p)
            matrices.append(m)
        return prefix, and_(*matrices) if isinstance(formula, And) else or_(*matrices)
    if isinstance(formula, Implies):
        a, b = formula.args
        return _prenex(or_(not_(a), b), taken)
    if isinstance(formula, Equivalent):
        args = list(formula.args)
        return _prenex(and_(*[implies(a, b) for a, b in zip(args, args[1:] + args[:1])]), taken)
    if isinstance(formula, Xor):
        first = formula.args[0]
        rest = list(formula.args[1:])
        second = xor(*rest) if len(rest) > 1 else rest[0]
        return _prenex(or_(and_(first, not_(second)), and_(not_(first), second)), taken)
    if isinstance(formula, ITE):
        c, a, b = formula.args
        return _prenex(or_(and_(c, a), and_(not_(c), b)), taken)
    return [], formula
