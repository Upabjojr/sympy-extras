r"""The SymPy face of the $\Pi\Sigma$-fields of
:mod:`sympy_extras.concrete.pisigma`.

:class:`SymPyPiSigmaField` is the field used by the summation algorithms
which are still written on SymPy (:mod:`~sympy_extras.concrete.creative`,
through the builder of :mod:`~sympy_extras.concrete.karr_sympy`; Karr's
algorithm itself, :mod:`~sympy_extras.concrete.karr`, uses the native
field): it takes and returns SymPy
expressions, its generators are SymPy ``Dummy`` symbols and the constants
of the solutions are SymPy expressions in the parameters, while the
elements, the shift and the solver are those of the native
:class:`~sympy_extras.concrete.pisigma.PiSigmaField` (the elements are
:class:`~sympy_extras.concrete.pisigma.PiSigmaElement`, with the
arithmetic operators). Unlike the native field it is mutable:
:meth:`~SymPyPiSigmaField.add_sigma` and :meth:`~SymPyPiSigmaField.add_pi`
replace the field by the extended one, so that the elements built before
belong to the old field.

The expressions are converted directly between SymPy's trees and the
polynomials of the field (rational operations on the index, the
parameters and the generator symbols), without the conversion of
:mod:`sympy_extras.interop`; only the sequences the generators stand for
are converted, once per extension.

Examples
========

>>> from sympy import harmonic, factorial
>>> from sympy.abc import k
>>> from sympy_extras.concrete.pisigma_sympy import SymPyPiSigmaField
>>> F = SymPyPiSigmaField(k)
>>> h = F.add_sigma(1/(k + 1), harmonic(k))
>>> g = F.telescope(F.from_expr(h))
>>> F.to_expr(g)
k*(harmonic(k) - 1)
>>> F.sigma(g) - g == F.from_expr(h)
True
>>> t = F.add_pi(k + 1, factorial(k))
>>> F.to_expr(F.telescope(F.from_expr(k*t)))
factorial(k)
>>> F.telescope(F.from_expr(t/k)) is None
True
"""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable, Optional, Sequence, Union

import sympy
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify

from sympy_extras._typing import as_expr
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import expr_from_sympy
from sympy_extras.polys.rings import PolyElement

from .pisigma import PiSigmaElement, PiSigmaTower, parametric_pisigma_field, rational_pisigma_field

__all__ = ['SymPyPiSigmaField', 'SymPyExtension', 'SymPySolutions']

#: a basis of solutions with the constants as SymPy expressions, or None
#: when the equation is out of scope
SymPySolutions = Optional[list[tuple[list[Expr], PiSigmaElement]]]


class SymPyExtension:
    """A generator of a :class:`SymPyPiSigmaField`.

    Attributes
    ==========

    kind : ``'sigma'`` or ``'pi'``
    symbol : Dummy
        The generator as a SymPy symbol.
    value : Expr
        The defining expression of $\\alpha$ ($\\Pi$) or $\\beta$ ($\\Sigma$)
        in terms of the summation index and the previous generators.
    expr : Expr
        The sequence the generator stands for (for instance
        ``harmonic(k)``).
    """

    __slots__ = ('kind', 'symbol', 'value', 'expr')

    def __init__(self, kind: str, symbol: Dummy, value: Expr, expr: Expr) -> None:
        self.kind = kind
        self.symbol = symbol
        self.value = value
        self.expr = expr

    def __repr__(self) -> str:
        return "Extension(%s, %s, %s)" % (self.kind, self.expr, self.value)


def _core_symbol(s: Symbol) -> CoreSymbol:
    converted = expr_from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _poly_to_sympy(p: PolyElement[Fraction], images: Sequence[Expr]) -> Expr:
    """The polynomial with the variables replaced by ``images``."""
    terms: list[Expr] = []
    for m, c in p.terms.items():
        factors: list[Expr] = [sympy.Rational(c.numerator, c.denominator)]
        for image, e in zip(images, m):
            if e:
                factors.append(as_expr(image**e))
        terms.append(as_expr(sympy.Mul(*factors)))
    return as_expr(sympy.Add(*terms))


class SymPyPiSigmaField:
    r"""A $\Pi\Sigma$-field $\mathbb{C}(k)(t_1, \ldots, t_m)$ on SymPy
    expressions.

    Parameters
    ==========

    k : Symbol
        The summation index, with $\sigma(k) = k + 1$.
    params : iterable of Symbol
        Parameters adjoined to the constants.
    """

    def __init__(self, k: Symbol, params: Iterable[Symbol] = ()) -> None:
        k_ = sympify(k)
        if not isinstance(k_, Symbol):
            raise TypeError("the summation index must be a symbol")
        self.k: Symbol = k_
        params_: list[Symbol] = []
        for p in params:
            p_ = sympify(p)
            if not isinstance(p_, Symbol):
                raise TypeError("the parameters must be symbols")
            params_.append(p_)
        self.params: tuple[Symbol, ...] = tuple(params_)
        native_k = _core_symbol(k_)
        native_params = [_core_symbol(p) for p in self.params]
        self.tower: PiSigmaTower = (parametric_pisigma_field(native_k, native_params) if native_params
                                    else rational_pisigma_field(native_k))
        self.extensions: list[SymPyExtension] = []
        #: the atoms represented in the field, filled by build_pisigma_field
        self.known_atoms: list[Expr] = []
        self.symbols: tuple[Symbol, ...]
        self.gens: tuple[PiSigmaElement, ...]
        self.zero: PiSigmaElement
        self.one: PiSigmaElement
        self._variables: dict[Symbol, PiSigmaElement]
        self._images: list[Expr]
        self._substituted: list[Expr]
        self._refresh()

    def _refresh(self) -> None:
        """The data derived from the native field, after a change of it."""
        T = self.tower
        symbols: list[Symbol] = [self.k] + [e.symbol for e in self.extensions]
        self.symbols = tuple(symbols)
        self.gens = T.gens
        self.zero = T.zero
        self.one = T.one
        variables: dict[Symbol, PiSigmaElement] = {}
        for p, q in zip(self.params, T.params):
            variables[p] = T.from_expr(q)
        for s, g in zip(symbols, T.gens):
            variables[s] = g
        self._variables = variables
        images: list[Expr] = [p for p in self.params]
        substituted: list[Expr] = [p for p in self.params]
        images.append(self.k)
        substituted.append(self.k)
        for e in self.extensions:
            images.append(e.symbol)
            substituted.append(e.expr)
        self._images = images
        self._substituted = substituted

    @property
    def level(self) -> int:
        """The number of extensions."""
        return self.tower.level

    # ------------------------------------------------------------------
    # conversions

    def from_expr(self, expr: Union[Expr, int]) -> PiSigmaElement:
        """Convert an expression in the index, the parameters and the
        generator symbols to an element of the field (``ValueError`` if it
        is not a rational function of them)."""
        return self._convert(as_expr(expr))

    def _convert(self, e: Expr) -> PiSigmaElement:
        if isinstance(e, Symbol):
            element = self._variables.get(e)
            if element is None:
                raise ValueError("%s is not a variable of the field" % (e,))
            return element
        if isinstance(e, sympy.Rational):
            return self.tower.from_fraction(Fraction(int(e.p), int(e.q)))
        if isinstance(e, sympy.Add):
            result = self.zero
            for a in e.args:
                result = result + self._convert(as_expr(a))
            return result
        if isinstance(e, sympy.Mul):
            result = self.one
            for a in e.args:
                result = result*self._convert(as_expr(a))
            return result
        if isinstance(e, sympy.Pow) and isinstance(e.exp, sympy.Integer):
            base = self._convert(as_expr(e.base))
            n = int(e.exp)
            if n < 0 and not base:
                raise ValueError("division by zero in %s" % (e,))
            return base**n
        raise ValueError("%s is not a rational function of %s" % (
            e, ', '.join(str(s) for s in self.params + self.symbols)))

    def to_expr(self, f: PiSigmaElement, substitute: bool = True) -> Expr:
        """Convert an element to an expression; with ``substitute`` the
        generators are replaced by the sequences they stand for."""
        images = self._substituted if substitute else self._images
        numer, denom = f.numer, f.denom
        if self._negative_denominator(denom):
            numer, denom = -numer, -denom
        expr = _poly_to_sympy(numer, images)
        if denom.terms != {denom.ring.zero_monom: Fraction(1)}:
            expr = as_expr(expr/_poly_to_sympy(denom, images))
        if substitute:
            expr = as_expr(factor_terms(expr))
        return expr

    def _negative_denominator(self, p: PolyElement[Fraction]) -> bool:
        """Whether the leading coefficient of ``p`` is negative in the
        lexicographic order of ``k``, the generators and then the
        parameters: the sign of the denominators of SymPy's fraction field
        over ``QQ(params)``, which the expressions keep (the native field
        orders the parameters first)."""
        P = len(self.params)
        if not P or not p.terms:
            return False
        leading = max(p.terms, key=lambda m: m[P:] + m[:P])
        return p.terms[leading] < 0

    def numer_expr(self, f: PiSigmaElement) -> Expr:
        """The numerator of ``f`` (a polynomial in the parameters, the
        index and the generator symbols with integer coefficients)."""
        return _poly_to_sympy(f.numer, self._images)

    def denom_expr(self, f: PiSigmaElement) -> Expr:
        """The denominator of ``f``, see :meth:`numer_expr`."""
        return _poly_to_sympy(f.denom, self._images)

    # ------------------------------------------------------------------
    # extensions

    def add_sigma(self, beta: Union[Expr, int], expr: Union[Expr, int], name: Optional[str] = None) -> Dummy:
        r"""Adjoin a $\Sigma$-extension $t$ with $\sigma(t) = t + \beta$.

        ``beta`` is an expression in the index and the existing generators
        and ``expr`` the sequence $t$ stands for. Returns the symbol of the
        new generator. No check is made that the extension is proper.
        """
        return self._add('sigma', beta, expr, name)

    def add_pi(self, alpha: Union[Expr, int], expr: Union[Expr, int], name: Optional[str] = None) -> Dummy:
        r"""Adjoin a $\Pi$-extension $t$ with $\sigma(t) = \alpha t$."""
        return self._add('pi', alpha, expr, name)

    def _add(self, kind: str, value: Union[Expr, int], expr: Union[Expr, int], name: Optional[str]) -> Dummy:
        value_ = as_expr(value)
        expr_ = as_expr(expr)
        element = self.from_expr(value_)
        if kind == 'pi' and not element:
            raise ValueError("alpha must be nonzero")
        symbol = Dummy(name or 't%d' % (len(self.extensions) + 1))
        self.tower = self.tower.extend(kind, element, expr_from_sympy(expr_), name)[0]
        self.extensions.append(SymPyExtension(kind, symbol, value_, expr_))
        self._refresh()
        return symbol

    # ------------------------------------------------------------------
    # the shift, levels and constants

    def sigma(self, f: PiSigmaElement, power: int = 1) -> PiSigmaElement:
        """The shift $\\sigma^{j}(f)$ for an integer ``j = power``."""
        return self.tower.sigma(f, power)

    def level_of(self, f: PiSigmaElement) -> int:
        """The index of the highest generator occurring in ``f`` (``-1``
        for a constant, ``0`` for an element of $\\mathbb{C}(k)$)."""
        return self.tower.level_of(f)

    def as_poly_in(self, f: PiSigmaElement, i: int, laurent: bool = False) -> Optional[dict[int, PiSigmaElement]]:
        """``f`` as a (Laurent) polynomial in the generator of index ``i``,
        see :meth:`~sympy_extras.concrete.pisigma.PiSigmaTower.as_poly_in`."""
        return self.tower.as_poly_in(f, i, laurent)

    def to_constant(self, f: PiSigmaElement) -> Expr:
        """The constant ``f`` as an expression in the parameters."""
        if not self.tower.is_constant(f):
            raise ValueError("%s is not a constant" % (self.to_expr(f, substitute=False),))
        return self.to_expr(f, substitute=False)

    def is_constant_expr(self, c: Expr) -> bool:
        """Whether ``c`` is an element of the constants (a rational
        function of the parameters)."""
        try:
            return self.tower.is_constant(self.from_expr(c))
        except (ValueError, TypeError, ZeroDivisionError):
            return False

    # ------------------------------------------------------------------
    # the solver

    def solve(self, a: PiSigmaElement, fs: Sequence[PiSigmaElement], level: Optional[int] = None) -> SymPySolutions:
        r"""Solve $\sigma(g) - a\,g = c_1 f_1 + \cdots + c_r f_r$ (see
        :meth:`~sympy_extras.concrete.pisigma.PiSigmaField.solve`), the
        constants $c_i$ as SymPy expressions in the parameters."""
        sols = self.tower.solve_embedded(a, fs, level)
        if sols is None:
            return None
        return [([self.to_expr(ci, substitute=False) for ci in c], g) for c, g in sols]

    def telescope(self, f: PiSigmaElement) -> Optional[PiSigmaElement]:
        r"""A solution $g$ of $\sigma(g) - g = f$, or ``None``."""
        return self.tower.telescope(f)
