"""Quantified formulas.

:class:`ForAll` and :class:`Exists` are the counterparts of Mathematica's
``ForAll[x, expr]`` and ``Exists[x, expr]``. They are Boolean objects and
can be combined with the other Boolean operators; :func:`prenex` puts a
formula in prenex normal form, with all the quantifiers in front, which is
what the quantifier elimination of :func:`~sympy_extras.assumptions.resolve`
works on.
"""
from __future__ import annotations

from typing import Iterable, Union

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or,
    Not, Implies, Equivalent, Xor, ITE)
from sympy.printing.printer import Printer

from sympy_extras._typing import QuantifierPrefix, as_boolean

__all__ = ['Quantifier', 'ForAll', 'Exists', 'prenex']


class Quantifier(Boolean):
    """Base class of :class:`ForAll` and :class:`Exists`.

    ``Quantifier(variables, formula)`` binds one symbol or a sequence of
    symbols in ``formula``.
    """

    #: 'forall' or 'exists' (``kind`` is taken by SymPy's Basic)
    quantifier: str = ''

    # SymPy constructors evaluate: the trivial cases return the formula
    # itself rather than an instance, which mypy does not allow for __new__
    def __new__(cls, variables: Union[Symbol, Iterable[Symbol]],  # type: ignore[misc]
                formula: Union[Boolean, bool]) -> Boolean:
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
        formula_ = as_boolean(formula)
        if isinstance(formula_, (BooleanTrue, BooleanFalse)) or not variables:
            return formula_
        # drop the variables that do not occur in the formula
        occurring = tuple(v for v in variables if v in formula_.free_symbols)
        if not occurring:
            return formula_
        result: Boolean = Basic.__new__(cls, Tuple(*occurring), formula_)
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

    def _eval_subs(self, old: Basic, new: Basic) -> Boolean:
        if old in self.variables:
            return self
        return type(self)(self.variables, self.formula._subs(old, new))


class ForAll(Quantifier):
    """Universal quantification, ``ForAll(x, formula)``.

    Examples
    ========

    >>> from sympy.abc import b, c, x
    >>> from sympy_extras.assumptions import ForAll, resolve
    >>> f = ForAll(x, x**2 + b*x + c > 0)
    >>> f
    ForAll(x, b*x + c + x**2 > 0)
    >>> sorted(f.free_symbols, key=str)
    [b, c]
    >>> resolve(f)
    b**2 - 4*c < 0
    """
    quantifier = 'forall'


class Exists(Quantifier):
    """Existential quantification, ``Exists(x, formula)``.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import a, b, x
    >>> from sympy_extras.assumptions import Exists, resolve
    >>> resolve(Exists(x, Eq(x**2 + a*x + b, 0)))
    a**2 - 4*b >= 0
    >>> resolve(Exists([a, b], Eq(x**2 + a*x + b, 0) & (a > 0)))
    True
    """
    quantifier = 'exists'


_DUAL = {'forall': 'exists', 'exists': 'forall'}


def prenex(formula: Union[Boolean, bool]) -> tuple[QuantifierPrefix, Boolean]:
    """Prenex normal form of a formula with quantifiers.

    Returns ``(prefix, matrix)`` where ``prefix`` is a list of pairs
    ``(kind, variable)`` with ``kind`` ``'forall'`` or ``'exists'``,
    outermost first, and ``matrix`` is quantifier-free. Quantifiers under
    ``Not`` are flipped and quantifiers under ``And``, ``Or``, ``Implies``,
    ``Equivalent`` and ``Xor`` are moved in front, renaming the bound
    variables when they clash.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.assumptions import ForAll, Exists, prenex
    >>> prenex(~ForAll(x, Exists(y, x < y)))
    ([('exists', x), ('forall', y)], x >= y)
    >>> prefix, matrix = prenex(ForAll(x, x > 0) & Exists(x, x < 0))
    >>> prefix
    [('exists', x), ('forall', _x)]
    >>> matrix
    (_x > 0) & (x < 0)
    """
    prefix, matrix = _prenex(as_boolean(formula), set())
    return prefix, matrix


def _prenex(formula: Boolean, taken: set[Basic]) -> tuple[QuantifierPrefix, Boolean]:
    """``taken`` is the set of symbols which are already used elsewhere:
    bound variables clashing with it are renamed."""
    if isinstance(formula, Quantifier):
        inner_prefix, matrix = _prenex(formula.formula, taken | set(formula.variables))
        prefix: QuantifierPrefix = []
        for v in formula.variables:
            if v in taken:
                new = Dummy(v.name, **v.assumptions0)
                matrix = matrix.xreplace({v: new})
                inner_prefix = [(k, new if u == v else u) for k, u in inner_prefix]
                v = new
            prefix.append((formula.quantifier, v))
        return prefix + inner_prefix, matrix
    if isinstance(formula, Not):
        prefix, matrix = _prenex(as_boolean(formula.args[0]), taken)
        return [(_DUAL[k], v) for k, v in prefix], Not(matrix)
    if isinstance(formula, (And, Or)):
        prefix = []
        matrices: list[Boolean] = []
        used = set(taken) | set().union(*[a.free_symbols for a in formula.args])
        for arg in formula.args:
            p, m = _prenex(as_boolean(arg), used)
            used |= {v for _, v in p}
            prefix.extend(p)
            matrices.append(m)
        return prefix, formula.func(*matrices)
    if isinstance(formula, Implies):
        a, b = formula.args
        return _prenex(Or(Not(as_boolean(a)), as_boolean(b)), taken)
    if isinstance(formula, Equivalent):
        args = [as_boolean(a) for a in formula.args]
        return _prenex(And(*[Implies(a, b) for a, b in zip(args, args[1:] + args[:1])]), taken)
    if isinstance(formula, Xor):
        a = as_boolean(formula.args[0])
        rest = [as_boolean(r) for r in formula.args[1:]]
        b = Xor(*rest) if len(rest) > 1 else rest[0]
        return _prenex(Or(And(a, Not(b)), And(Not(a), b)), taken)
    if isinstance(formula, ITE):
        c, a, b = [as_boolean(arg) for arg in formula.args]
        return _prenex(Or(And(c, a), And(Not(c), b)), taken)
    return [], formula
