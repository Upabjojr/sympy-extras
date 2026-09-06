"""Quantified formulas.

:class:`ForAll` and :class:`Exists` are the counterparts of Mathematica's
``ForAll[x, expr]`` and ``Exists[x, expr]``. They are Boolean objects and
can be combined with the other Boolean operators; :func:`prenex` puts a
formula in prenex normal form, with all the quantifiers in front, which is
what the quantifier elimination of :func:`~sympy_extras.assumptions.resolve`
works on.
"""
from __future__ import annotations

from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.logic.boolalg import (Boolean, BooleanTrue, BooleanFalse, And, Or,
    Not, Implies, Equivalent, Xor, ITE, true, false)

__all__ = ['Quantifier', 'ForAll', 'Exists', 'prenex']


class Quantifier(Boolean):
    """Base class of :class:`ForAll` and :class:`Exists`.

    ``Quantifier(variables, formula)`` binds one symbol or a sequence of
    symbols in ``formula``.
    """

    kind = None  # 'forall' or 'exists'

    def __new__(cls, variables, formula):
        if isinstance(variables, (list, tuple, set, frozenset, Tuple)):
            variables = tuple(variables)
        else:
            variables = (variables,)
        variables = tuple(sympify(v) for v in variables)
        for v in variables:
            if not isinstance(v, Symbol):
                raise TypeError("quantified variables must be symbols, got %s" % (v,))
        if len(set(variables)) != len(variables):
            raise ValueError("repeated quantified variable in %s" % (variables,))
        formula = sympify(formula)
        if formula is True or formula is False:
            formula = true if formula else false
        if not isinstance(formula, Boolean):
            raise TypeError("the formula must be a Boolean, got %s" % (formula,))
        if isinstance(formula, (BooleanTrue, BooleanFalse)) or not variables:
            return formula
        # drop the variables that do not occur in the formula
        occurring = tuple(v for v in variables if v in formula.free_symbols)
        if not occurring:
            return formula
        return Basic.__new__(cls, Tuple(*occurring), formula)

    @property
    def variables(self):
        """The quantified variables."""
        return tuple(self.args[0])

    @property
    def formula(self):
        """The quantified formula."""
        return self.args[1]

    @property
    def free_symbols(self):
        return self.formula.free_symbols - set(self.variables)

    @property
    def bound_symbols(self):
        return list(self.variables)

    @property
    def binary_symbols(self):
        return self.formula.binary_symbols - set(self.variables)

    def _sympystr(self, printer):
        variables = self.variables
        v = printer._print(variables[0]) if len(variables) == 1 else \
            "(%s)" % ", ".join(printer._print(x) for x in variables)
        return "%s(%s, %s)" % (type(self).__name__, v, printer._print(self.formula))

    _sympyrepr = _sympystr

    def _latex(self, printer):
        symbol = r"\forall" if self.kind == 'forall' else r"\exists"
        return r"%s %s \, %s" % (symbol,
            ", ".join(printer._print(x) for x in self.variables),
            printer._print(self.formula))

    def _eval_subs(self, old, new):
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
    kind = 'forall'


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
    kind = 'exists'


_DUAL = {'forall': 'exists', 'exists': 'forall'}


def prenex(formula):
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
    formula = sympify(formula)
    if formula is True or formula is False:
        formula = true if formula else false
    prefix, matrix = _prenex(formula, set())
    return prefix, matrix


def _prenex(formula, taken):
    """``taken`` is the set of symbols which are already used elsewhere:
    bound variables clashing with it are renamed."""
    if isinstance(formula, Quantifier):
        inner_prefix, matrix = _prenex(formula.formula, taken | set(formula.variables))
        prefix = []
        for v in formula.variables:
            if v in taken:
                new = Dummy(v.name, **v.assumptions0)
                matrix = matrix.xreplace({v: new})
                inner_prefix = [(k, new if u == v else u) for k, u in inner_prefix]
                v = new
            prefix.append((formula.kind, v))
        return prefix + inner_prefix, matrix
    if isinstance(formula, Not):
        prefix, matrix = _prenex(formula.args[0], taken)
        return [(_DUAL[k], v) for k, v in prefix], Not(matrix)
    if isinstance(formula, (And, Or)):
        prefix, matrices = [], []
        used = set(taken) | set().union(*[a.free_symbols for a in formula.args])
        for arg in formula.args:
            p, m = _prenex(arg, used)
            used |= {v for _, v in p}
            prefix.extend(p)
            matrices.append(m)
        return prefix, formula.func(*matrices)
    if isinstance(formula, Implies):
        a, b = formula.args
        return _prenex(Or(Not(a), b), taken)
    if isinstance(formula, Equivalent):
        args = formula.args
        return _prenex(And(*[Implies(a, b) for a, b in zip(args, args[1:] + args[:1])]), taken)
    if isinstance(formula, Xor):
        a, rest = formula.args[0], formula.args[1:]
        b = Xor(*rest) if len(rest) > 1 else rest[0]
        return _prenex(Or(And(a, Not(b)), And(Not(a), b)), taken)
    if isinstance(formula, ITE):
        c, a, b = formula.args
        return _prenex(Or(And(c, a), And(Not(c), b)), taken)
    return [], formula
