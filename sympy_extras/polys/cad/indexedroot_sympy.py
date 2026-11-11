"""The SymPy face of the indexed real roots
(:mod:`sympy_extras.polys.cad.indexedroot`).

:class:`IndexedRoot` is a SymPy ``Function`` which evaluates, as SymPy's
functions do: the bound variable becomes a dummy kept for its name, and a
root whose coefficients are rational becomes a number (a ``CRootOf`` of
its minimal polynomial, or its radical form). :mod:`sympy_extras.interop`
converts the native node and this class both ways
(:func:`sympy_indexed_root`, :func:`native_indexed_root`); a native root
with rational coefficients becomes the ``CRootOf`` of its minimal
polynomial in the variable of its name.

Examples
========

>>> from sympy.abc import x, y
>>> from sympy_extras.polys.cad.indexedroot_sympy import IndexedRoot
>>> r = IndexedRoot(y**3 + x*y - 1, y, 0); r
IndexedRoot(x*y + y**3 - 1, y, 0)
>>> r.subs(x, 0), r.subs(x, 1).evalf(10), r.free_symbols       # a CRootOf at x = 1
(1, 0.6823278038, {x})
>>> IndexedRoot(y**3 - 3*y + x, y, 2).subs(x, 5)     # one real root only
IndexedRoot(y**3 - 3*y + 5, y, 2)
"""
from __future__ import annotations

from typing import Optional

from mpmath.libmp.libhyper import NoConvergence
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Dummy, Symbol
from sympy.polys.polyerrors import BasePolynomialError
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import ComplexRootOf
from sympy.printing.printer import Printer

from sympy_extras._typing_sympy import as_expr
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.polys.roots_sympy import radical_form

from . import indexedroot as native
from .samplepoints_sympy import rational_real_roots, sympy_real

__all__ = ['IndexedRoot', 'sympy_indexed_root', 'native_indexed_root']

#: the bound variables of the roots, by the name of the variable: one per
#: name, so that equal root functions are equal expressions
_BOUND: dict[str, Dummy] = {}


def _bound_variable(name: str) -> Dummy:
    if name not in _BOUND:
        _BOUND[name] = Dummy(name, real=True)
    return _BOUND[name]


class IndexedRoot(Function):
    """``IndexedRoot(f, t, k)``: the ``k``-th distinct real root, counted
    from 0 in increasing order, of the polynomial ``f`` in ``t`` whose
    coefficients depend on other symbols, the counterpart of Mathematica's
    parametric ``Root[f, k]``. It is a function of those symbols where the
    polynomial has more than ``k`` real roots, and becomes a number when
    they are given rational values.

    The variable ``t`` is bound: it is replaced by a dummy of the same
    name, which a substitution of ``t`` in the formula around does not
    reach.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy_extras.polys.cad.indexedroot_sympy import IndexedRoot
    >>> IndexedRoot(y**2 - x, y, 1).subs(x, 4)
    2
    """

    @classmethod
    def eval(cls, f: Expr, t: Expr, k: Expr) -> Optional[Expr]:
        if not (isinstance(t, Symbol) and isinstance(k, Integer) and k >= 0):
            raise ValueError("IndexedRoot(f, t, k) takes a symbol t and an index k, from 0")
        bound = _bound_variable(t.name)
        if t != bound:
            return cls(as_expr(f.xreplace({t: bound})), bound, k)
        if not f.has(bound):
            # a polynomial which the values of the parameters made constant
            # has no root: the object stays as it is, like one whose index
            # is beyond the number of the roots
            return None
        if f.free_symbols == {bound}:
            return _numerical_root(f, bound, int(k))
        return None

    def _eval_is_extended_real(self) -> bool:
        return True

    @property
    def expr(self) -> Expr:
        return as_expr(self.args[0])

    @property
    def variable(self) -> Dummy:
        variable = self.args[1]
        assert isinstance(variable, Dummy)
        return variable

    @property
    def index(self) -> int:
        return int(as_expr(self.args[2]))

    @property
    def free_symbols(self) -> set[Basic]:
        return set(self.expr.free_symbols) - {self.variable}

    @property
    def bound_symbols(self) -> list[Symbol]:
        return [self.variable]

    def _eval_evalf(self, prec: int) -> Optional[Expr]:
        # coefficients which are not rational numbers: the real ones among
        # the numerical roots. A multiple root (which is what the polynomial
        # has over a section of its discriminant) is found with half of the
        # working digits, slowly: twice the digits asked for, and as many
        # steps as it takes
        if self.free_symbols:
            return None
        digits = max(15, prec_to_dps(prec))
        working = 2 * digits + 10
        try:
            numerical = Poly(self.expr.evalf(working + 10), self.variable).nroots(n=working, maxsteps=5000)
        except (BasePolynomialError, NoConvergence, ValueError, TypeError, ArithmeticError):
            return None
        tolerance = Float(10)**(3 - digits)
        values: list[Expr] = []
        for root in numerical:
            real_part, imaginary_part = as_expr(root).as_real_imag()
            if abs(imaginary_part) < tolerance * (1 + abs(real_part)):
                values.append(as_expr(real_part))
        values.sort(key=lambda value: float(value))
        distinct: list[Expr] = []
        for value in values:
            if not distinct or abs(value - distinct[-1]) > tolerance * (1 + abs(value)):
                distinct.append(value)
        if self.index >= len(distinct):
            return None
        return as_expr(distinct[self.index].evalf(digits))

    def _sympystr(self, printer: Printer) -> str:
        shown = Symbol(self.variable.name)
        return "IndexedRoot(%s, %s, %s)" % (printer._print(self.expr.xreplace({self.variable: shown})),
                                            printer._print(shown), self.index)


def _numerical_root(expression: Expr, t: Symbol, index: int) -> Optional[Expr]:
    """The ``index``-th distinct real root of a polynomial with rational
    coefficients, when it has that many."""
    try:
        # in a plain symbol: the root object shows its generator
        plain = Symbol(t.name)
        poly = Poly(expression.xreplace({t: plain}), plain, domain='QQ')
    except BasePolynomialError:
        return None
    if poly.degree() < 1:
        return None
    roots = [as_expr(r) for r, _ in rational_real_roots(poly)]
    if index >= len(roots):
        return None
    root = roots[index]
    form = radical_form(root) if isinstance(root, ComplexRootOf) else None
    return root if form is None else form


def sympy_indexed_root(r: native.IndexedRoot) -> Expr:
    """The SymPy counterpart of a native root: the ``CRootOf`` of the
    minimal polynomial (in the plain symbol of the name of the variable)
    for a root with rational coefficients, else :class:`IndexedRoot`.

    >>> from sympy_extras.core import symbols
    >>> from sympy_extras.polys.cad import indexedroot as native
    >>> from sympy_extras.polys.cad.indexedroot_sympy import sympy_indexed_root
    >>> x, y = symbols('x y')
    >>> from sympy import CRootOf, Symbol
    >>> sympy_indexed_root(native.indexed_root(y**3 - y - 1, y, 0)) == CRootOf(Symbol('y')**3 - Symbol('y') - 1, 0)
    True
    >>> sympy_indexed_root(native.indexed_root(y**2 - x, y, 1))
    IndexedRoot(-x + y**2, y, 1)
    """
    from sympy_extras.interop import to_sympy
    a = native.algebraic_of(r)
    plain = Symbol(r.variable.name)
    if a is not None:
        return sympy_real(a, plain)
    f = as_expr(to_sympy(r.expr.xreplace({r.variable: CoreSymbol(r.variable.name)})))
    return as_expr(IndexedRoot(f, plain, Integer(r.index)))


def native_indexed_root(r: IndexedRoot) -> native.IndexedRoot:
    """The native counterpart of a SymPy root (no evaluation: the node of
    the bound variable of the native module)."""
    from sympy_extras.interop import from_sympy
    name = r.variable.name
    f = from_sympy(r.expr.xreplace({r.variable: Symbol(name)}))
    if not isinstance(f, CoreExpr):
        raise TypeError("the polynomial of %s is not an expression" % (r,))
    bound = native.bound_variable(name)
    return native.IndexedRoot(f.xreplace({CoreSymbol(name): bound}), bound, r.index)
