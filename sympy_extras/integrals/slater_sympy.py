"""The SymPy face of Slater's theorem and of the Mellin–Barnes integrals of
gamma quotients (:mod:`sympy_extras.integrals.slater`), for the
integrators which are still written on SymPy.

:class:`MeijerG` is the container of the native module on SymPy objects,
with ``to_core`` and ``from_core`` to the native class; the functions
(:func:`mellin_barnes`, :func:`slater_expansion`, :func:`expand_meijerg`,
:func:`line_conditions`) convert their arguments by
:mod:`sympy_extras.interop` and run the native code. Nothing of the
module stays on SymPy (the native ``hyperexpand``, ``simplify`` and
limits lend the nodes they have not ported to SymPy through the bridge
of the core).

Examples
========

>>> from sympy import symbols, meijerg, S
>>> from sympy_extras.integrals.slater_sympy import slater_expansion
>>> z = symbols('z', positive=True)
>>> slater_expansion(meijerg([], [], [0], [], z))
exp(-z)
>>> slater_expansion(meijerg([1], [], [S.Half], [0], z))
sqrt(pi)*erf(sqrt(z))
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.functions.special.hyper import meijerg
from sympy.logic.boolalg import Boolean

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, sympy_boolean

from . import slater as native
from .conditions_sympy import ConditionalValue
from .mellin_sympy import GammaQuotient

__all__ = ['MeijerG', 'mellin_barnes', 'slater_expansion', 'expand_meijerg', 'line_conditions']


class MeijerG:
    """:class:`sympy_extras.integrals.slater.MeijerG` on SymPy objects: a
    G-function `G^{m,n}_{p,q}(z \\mid a_n, a_p; b_m, b_q)` with a
    multiplicative prefactor."""

    __slots__ = ('prefactor', 'an', 'ap', 'bm', 'bq', 'z')

    def __init__(self, prefactor: ExprLike, an: Sequence[ExprLike], ap: Sequence[ExprLike],
                 bm: Sequence[ExprLike], bq: Sequence[ExprLike], z: ExprLike) -> None:
        self.prefactor: Final[Expr] = as_expr(prefactor)
        self.an: Final[tuple[Expr, ...]] = tuple(as_expr(a) for a in an)
        self.ap: Final[tuple[Expr, ...]] = tuple(as_expr(a) for a in ap)
        self.bm: Final[tuple[Expr, ...]] = tuple(as_expr(b) for b in bm)
        self.bq: Final[tuple[Expr, ...]] = tuple(as_expr(b) for b in bq)
        self.z: Final[Expr] = as_expr(z)

    def to_core(self) -> native.MeijerG:
        """The same G-function on the objects of the core."""
        return native.MeijerG(core_expr(self.prefactor), [core_expr(a) for a in self.an],
                              [core_expr(a) for a in self.ap], [core_expr(b) for b in self.bm],
                              [core_expr(b) for b in self.bq], core_expr(self.z))

    @staticmethod
    def from_core(g: native.MeijerG) -> MeijerG:
        """A G-function of the core on SymPy objects."""
        return MeijerG(sympy_expr(g.prefactor), [sympy_expr(a) for a in g.an], [sympy_expr(a) for a in g.ap],
                       [sympy_expr(b) for b in g.bm], [sympy_expr(b) for b in g.bq], sympy_expr(g.z))

    def __repr__(self) -> str:
        return "MeijerG(%s, %s)" % (self.prefactor, self.as_sympy())

    @property
    def m(self) -> int:
        return len(self.bm)

    @property
    def n(self) -> int:
        return len(self.an)

    @property
    def p(self) -> int:
        return len(self.an) + len(self.ap)

    @property
    def q(self) -> int:
        return len(self.bm) + len(self.bq)

    @property
    def delta(self) -> Rational:
        """`m + n - (p + q)/2`."""
        return Rational(2 * (self.m + self.n) - self.p - self.q, 2)

    def as_sympy(self) -> Expr:
        """The :class:`~sympy.functions.special.hyper.meijerg` expression
        (without the prefactor)."""
        return as_expr(meijerg(list(self.an), list(self.ap), list(self.bm), list(self.bq), self.z))

    def reflected(self) -> MeijerG:
        """The same function written with argument ``1/z``."""
        return MeijerG.from_core(self.to_core().reflected())


def mellin_barnes(quotient: GammaQuotient) -> Optional[MeijerG]:
    """:func:`sympy_extras.integrals.slater.mellin_barnes` on SymPy
    objects: the G-function equal to ``(1/2 pi i) Integral(quotient(t), t)``
    along a vertical line inside the strip; ``None`` when the quotient has
    an ``extra`` factor.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy_extras.integrals.mellin_sympy import GammaQuotient
    >>> from sympy_extras.integrals.slater_sympy import mellin_barnes
    >>> q = GammaQuotient(1, [(3, -1)], [(0, S.One), (1, -S.One)], [], 0, 1)
    >>> mellin_barnes(q)
    MeijerG(1, meijerg(((1,), ()), ((1,), ()), 1/3))
    >>> q = GammaQuotient(1, [], [(S.Half, S.Half)], [(1, -S.Half)], -1, 1)
    >>> mellin_barnes(q)
    MeijerG(2, meijerg(((1/2,), (1,)), ((), ()), 1))
    """
    found = native.mellin_barnes(quotient.to_core())
    return None if found is None else MeijerG.from_core(found)


def slater_expansion(g: Expr, assumptions: Assumptions = None) -> Expr:
    """:func:`sympy_extras.integrals.slater.slater_expansion` on SymPy
    objects: a :class:`~sympy.functions.special.hyper.meijerg` expression
    as a sum of hypergeometric functions by Slater's theorem, each then
    expanded by ``hyperexpand``; unchanged when the theorem does not apply.

    Examples
    ========

    >>> from sympy import symbols, meijerg, S
    >>> from sympy_extras.integrals.slater_sympy import slater_expansion
    >>> z, a = symbols('z a', positive=True)
    >>> slater_expansion(meijerg([1 - a], [], [0], [], z))
    gamma(a)/(z + 1)**a
    >>> slater_expansion(meijerg([1], [], [S.Half], [0], z))
    sqrt(pi)*erf(sqrt(z))
    """
    if not isinstance(g, meijerg):
        return g
    value = native.slater_expansion(core_expr(g), core_assumptions(assumptions))
    return sympy_expr(value)


def expand_meijerg(g: MeijerG, assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.slater.expand_meijerg` on SymPy
    objects: ``g`` (with its prefactor) expanded by Slater's theorem;
    ``None`` when no expansion is found."""
    found = native.expand_meijerg(g.to_core(), core_assumptions(assumptions))
    return None if found is None else ConditionalValue.from_core(found)


def line_conditions(g: MeijerG, lower: Expr, upper: Expr) -> Optional[Boolean]:
    """:func:`sympy_extras.integrals.slater.line_conditions` on SymPy
    objects: the condition for the Mellin–Barnes integral along a vertical
    line in the strip ``lower < Re s < upper`` to converge absolutely;
    ``None`` when it fails."""
    found = native.line_conditions(g.to_core(), core_expr(lower), core_expr(upper))
    return None if found is None else sympy_boolean(found)
