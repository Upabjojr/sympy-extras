"""The SymPy face of :mod:`sympy_extras.polys.comprehensive`: comprehensive
Gröbner systems and quantifier elimination over the complex numbers.

The functions take SymPy expressions, symbols and formulas, convert them
to the core (:mod:`sympy_extras.interop`), run the native functions and
convert the answers back: SymPy Booleans, and branches whose polynomials
are SymPy expressions. Nothing of the computation stays on SymPy.

Examples
========

>>> from sympy import Eq
>>> from sympy.abc import a, b, x
>>> from sympy_extras.polys.comprehensive_sympy import complex_quantifier_elimination, exists_complex
>>> exists_complex([a*x - b], [], [x])
Ne(a, 0) | (Eq(a, 0) & Eq(b, 0))
>>> complex_quantifier_elimination(Eq(a*x, 1), [('exists', x)])
Ne(a, 0)
"""
from __future__ import annotations

from typing import Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import And, Boolean

from sympy_extras._typing_sympy import QuantifierPrefix, as_boolean, as_expr, as_symbol
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.logic import Boolean as CoreBoolean
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import comprehensive as native

__all__ = ['Branch', 'comprehensive_groebner_system', 'exists_complex',
           'complex_quantifier_elimination', 'reduce_complex']


class Branch:
    """A branch of a comprehensive Gröbner system
    (:class:`sympy_extras.polys.comprehensive.Branch` with SymPy
    expressions): on the parameters where the ``equations`` vanish and the
    ``nonzero`` polynomials do not, the ``basis`` specialises to a Gröbner
    basis of the system.

    ``basis == [1]`` means that the system has no solution on the branch,
    ``basis == []`` that every point is a solution.
    """

    def __init__(self, equations: list[Expr], nonzero: list[Expr], basis: list[Expr]) -> None:
        self.equations = equations
        self.nonzero = nonzero
        self.basis = basis

    def condition(self) -> Boolean:
        """The constructible set of the branch, as a formula."""
        return And(*[Eq(e, 0) for e in self.equations], *[Ne(n, 0) for n in self.nonzero])

    @property
    def solvable(self) -> bool:
        """Whether the system has solutions on the branch."""
        return not any(g.is_number and g != 0 for g in self.basis)

    def __repr__(self) -> str:
        return "Branch(%s, %s, %s)" % (self.equations, self.nonzero, self.basis)


def _core_expr(e: Basic) -> CoreExpr:
    converted = from_sympy(as_expr(sympify(e)))
    if not isinstance(converted, CoreExpr):
        raise TypeError("%s is not an expression" % (e,))
    return converted


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(as_symbol(s))
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _core_formula(formula: Union[Boolean, bool]) -> CoreBoolean:
    converted = from_sympy(as_boolean(sympify(formula)))
    if not isinstance(converted, CoreBoolean):
        raise ValueError("%s is not a formula" % (formula,))
    return converted


def _sympy_expr(e: CoreExpr) -> Expr:
    return as_expr(to_sympy(e))


def _sympy_formula(formula: CoreBoolean) -> Boolean:
    return as_boolean(to_sympy(formula))


def _face(branch: native.Branch) -> Branch:
    return Branch([_sympy_expr(e) for e in branch.equations], [_sympy_expr(e) for e in branch.nonzero],
                  [_sympy_expr(e) for e in branch.basis])


def comprehensive_groebner_system(polynomials: Sequence[Expr], variables: Sequence[Symbol],
                                  parameters: Sequence[Symbol]) -> list[Branch]:
    """:func:`~sympy_extras.polys.comprehensive.comprehensive_groebner_system`
    on SymPy expressions: branches covering the parameter space, on each
    of which the basis specialises to a Gröbner basis of the specialised
    system.

    Examples
    ========

    >>> from sympy.abc import a, b, x
    >>> from sympy_extras.polys.comprehensive_sympy import comprehensive_groebner_system
    >>> for branch in comprehensive_groebner_system([a*x - b], [x], [a, b]):
    ...     print(branch)
    Branch([], [a], [a*x - b])
    Branch([a], [b], [1])
    Branch([a, b], [], [])
    """
    branches = native.comprehensive_groebner_system([_core_expr(p) for p in polynomials],
                                                    [_core_symbol(v) for v in variables],
                                                    [_core_symbol(p) for p in parameters])
    return [_face(branch) for branch in branches]


def exists_complex(equations: Sequence[Expr], inequations: Sequence[Expr],
                   variables: Sequence[Symbol]) -> Boolean:
    """:func:`~sympy_extras.polys.comprehensive.exists_complex` on SymPy
    expressions: the condition on the parameters for the system of
    equations ``f = 0`` and inequations ``g != 0`` to have a solution in
    the variables over the complex numbers.

    Examples
    ========

    >>> from sympy.abc import a, b, c, x
    >>> from sympy_extras.polys.comprehensive_sympy import exists_complex
    >>> exists_complex([a*x**2 + b*x + c, 2*a*x + b], [a], [x])
    Ne(a, 0) & Eq(-4*a*c + b**2, 0)
    >>> exists_complex([x**2 - 2*x + 1], [x - a], [x])
    Ne(a - 1, 0)
    """
    return _sympy_formula(native.exists_complex([_core_expr(e) for e in equations],
                                                [_core_expr(g) for g in inequations],
                                                [_core_symbol(v) for v in variables]))


def complex_quantifier_elimination(formula: Union[Boolean, bool], prefix: QuantifierPrefix) -> Boolean:
    """:func:`~sympy_extras.polys.comprehensive.complex_quantifier_elimination`
    on SymPy formulas: the quantifiers (outermost first in ``prefix``) of
    a Boolean combination of polynomial equations and inequations
    eliminated over the complex numbers.

    Examples
    ========

    >>> from sympy import Eq
    >>> from sympy.abc import a, b, x, y
    >>> from sympy_extras.polys.comprehensive_sympy import complex_quantifier_elimination
    >>> complex_quantifier_elimination(Eq(x**2, a), [('exists', x)])
    True
    >>> complex_quantifier_elimination(Eq(x*y, 1), [('forall', x), ('exists', y)])
    False
    >>> complex_quantifier_elimination(Eq(x**2 + a*x + b, 0) & Eq(2*x + a, 0), [('exists', x)])
    Eq(a**2 - 4*b, 0)
    """
    found = native.complex_quantifier_elimination(_core_formula(formula),
                                                  [(kind, _core_symbol(v)) for kind, v in prefix])
    return _sympy_formula(found)


def reduce_complex(formula: Union[Boolean, bool]) -> Boolean:
    """:func:`~sympy_extras.polys.comprehensive.reduce_complex` on SymPy
    formulas: a quantifier-free Boolean combination of polynomial
    equations and inequations simplified over the complex numbers with
    Gröbner bases.

    Examples
    ========

    >>> from sympy import Eq, Ne
    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.comprehensive_sympy import reduce_complex
    >>> reduce_complex(Eq(x**2 - y, 0) & Eq(x*y - 1, 0) & Ne(x, 0))
    Eq(x*y - 1, 0) & Eq(-x + y**2, 0) & Eq(x**2 - y, 0)
    >>> reduce_complex(Eq(x**2, 0) & Ne(x, 0))
    False
    """
    return _sympy_formula(native.reduce_complex(_core_formula(formula)))
