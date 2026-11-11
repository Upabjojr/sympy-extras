"""The SymPy face of the Abel equations of the first kind
(:mod:`sympy_extras.solvers.abel`): their invariants, equivalence classes
and the integrable classes recognised through them.

:func:`abel_coefficients`, :func:`abel_invariants`,
:func:`constant_invariant`, :func:`abel_equivalence`,
:func:`particular_solution`, :func:`air_solution` and
:func:`abel_by_invariants` take and return SymPy objects, converted by
:mod:`sympy_extras.interop` around the native functions;
:class:`AbelInvariants` and :class:`AbelTransformation` carry SymPy
expressions and ``REPRESENTATIVES`` holds SymPy coefficients.

Examples
========

>>> from sympy import Symbol
>>> from sympy_extras.solvers.abel_sympy import abel_invariants
>>> x = Symbol('x')
>>> abel_invariants((-1, -2*x, 0, 0), x).I1
11664*x**6*(8*x**3 - 15)**3/(8*x**3 - 9)**5
"""
from __future__ import annotations

from typing import Final, Optional, Sequence, Union

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef
from sympy.core.symbol import Symbol

from sympy_extras._typing_sympy import as_expr
from sympy_extras.assumptions.ask_sympy import _core_symbol, _sympy_symbol
from sympy_extras.assumptions.intervals_sympy import core_expr, sympy_expr
from sympy_extras.core.basic import Basic as CoreBasic
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.function import AppliedUndef as CoreAppliedUndef
from sympy_extras.core.numbers import Integer as CoreInteger
from sympy_extras.interop import from_sympy, to_sympy

from . import abel as native

__all__ = ['AbelInvariants', 'Coefficients', 'CoefficientsLike', 'abel_coefficients', 'abel_invariants',
           'abel_equivalence', 'AbelTransformation', 'particular_solution', 'air_solution', 'abel_by_invariants',
           'constant_invariant', 'REPRESENTATIVES']

#: the coefficients ``(f3, f2, f1, f0)`` of ``y' = f3 y**3 + f2 y**2 + f1 y + f0``
Coefficients = tuple[Expr, Expr, Expr, Expr]
#: the same, as accepted from the user (integers allowed)
CoefficientsLike = tuple[Union[Expr, int], Union[Expr, int], Union[Expr, int], Union[Expr, int]]


# ---------------------------------------------------------------------------
# conversions

def _core_basic(e: Basic) -> CoreBasic:
    converted = from_sympy(e)
    if not isinstance(converted, CoreBasic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _core_function(f: AppliedUndef) -> CoreAppliedUndef:
    converted = from_sympy(f)
    if not isinstance(converted, CoreAppliedUndef):
        raise TypeError("%s is not an applied undefined function" % (f,))
    return converted


def _sympy_basic(e: CoreBasic) -> Basic:
    converted = to_sympy(e)
    if not isinstance(converted, Basic):
        raise TypeError("%s cannot be converted" % (e,))
    return converted


def _core_coefficient(c: Union[Expr, int]) -> CoreExpr:
    return CoreInteger(c) if isinstance(c, int) else core_expr(c)


def _core_coefficients(coefficients: CoefficientsLike) -> native.Coefficients:
    f3, f2, f1, f0 = coefficients
    return _core_coefficient(f3), _core_coefficient(f2), _core_coefficient(f1), _core_coefficient(f0)


def _sympy_coefficients(coefficients: native.Coefficients) -> Coefficients:
    f3, f2, f1, f0 = coefficients
    return sympy_expr(f3), sympy_expr(f2), sympy_expr(f1), sympy_expr(f0)


class AbelInvariants:
    """:class:`sympy_extras.solvers.abel.AbelInvariants` with SymPy
    expressions: the relative invariants ``s3``, ``s5`` and the absolute
    invariants ``I1``, ``I2`` of an Abel equation (expressions in ``x``)."""

    def __init__(self, s3: Expr, s5: Expr, I1: Expr, I2: Expr) -> None:
        self.s3: Final[Expr] = s3
        self.s5: Final[Expr] = s5
        self.I1: Final[Expr] = I1
        self.I2: Final[Expr] = I2

    def __repr__(self) -> str:
        return "AbelInvariants(s3=%s, s5=%s, I1=%s, I2=%s)" % (self.s3, self.s5, self.I1, self.I2)

    @staticmethod
    def of_core(found: native.AbelInvariants) -> AbelInvariants:
        """The invariants of the native module with SymPy expressions."""
        return AbelInvariants(sympy_expr(found.s3), sympy_expr(found.s5), sympy_expr(found.I1),
                              sympy_expr(found.I2))


class AbelTransformation:
    """:class:`sympy_extras.solvers.abel.AbelTransformation` with SymPy
    expressions: ``y = P(x) u(xi(x)) + Q(x)`` taking one Abel equation to
    another."""

    def __init__(self, xi: Expr, P: Expr, Q: Expr, x: Symbol, t: Symbol) -> None:
        self.xi: Final[Expr] = xi
        self.P: Final[Expr] = P
        self.Q: Final[Expr] = Q
        self.x: Final[Symbol] = x
        self.t: Final[Symbol] = t

    def __repr__(self) -> str:
        return "AbelTransformation(xi=%s, P=%s, Q=%s)" % (self.xi, self.P, self.Q)

    @staticmethod
    def of_core(found: native.AbelTransformation) -> AbelTransformation:
        """The transformation of the native module with SymPy expressions."""
        return AbelTransformation(sympy_expr(found.xi), sympy_expr(found.P), sympy_expr(found.Q),
                                  _sympy_symbol(found.x), _sympy_symbol(found.t))


#: representatives of integrable classes with a non-constant invariant
#: (those of :data:`sympy_extras.solvers.abel.REPRESENTATIVES`): the
#: coefficients in ``t`` and the pair ``(g1, g0)`` of their inverse
#: Riccati equation ``dx/dz = g1(x) z + g0(x)``
REPRESENTATIVES: Final[list[tuple[Coefficients, tuple[Expr, Expr]]]] = [
    (_sympy_coefficients(c), (sympy_expr(g1), sympy_expr(g0))) for c, (g1, g0) in native.REPRESENTATIVES]


def abel_coefficients(equation: Basic, f: AppliedUndef) -> tuple[Coefficients, Symbol]:
    """``((f3, f2, f1, f0), x)`` of an equation ``y' = f3 y**3 + ...`` of the
    first kind (``ValueError`` otherwise;
    :func:`sympy_extras.solvers.abel.abel_coefficients` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.abel_sympy import abel_coefficients
    >>> y = Function('y')(x)
    >>> abel_coefficients(y.diff(x) - x*y**3 - y**2 + 1, y)
    ((x, 1, 0, -1), x)
    """
    coefficients, x = native.abel_coefficients(_core_basic(equation), _core_function(f))
    return _sympy_coefficients(coefficients), _sympy_symbol(x)


def abel_invariants(coefficients: CoefficientsLike, x: Symbol) -> AbelInvariants:
    """The invariants of ``y' = f3 y**3 + f2 y**2 + f1 y + f0``
    (:func:`sympy_extras.solvers.abel.abel_invariants` on SymPy objects).

    Examples
    ========

    >>> from sympy import Symbol, Rational
    >>> from sympy_extras.solvers.abel_sympy import abel_invariants
    >>> x = Symbol('x')
    >>> abel_invariants((-1, -2*x, 0, 0), x).I1
    11664*x**6*(8*x**3 - 15)**3/(8*x**3 - 9)**5
    >>> abel_invariants((1, 0, 0, x**Rational(-3, 2)), x).I1
    -27/8
    """
    return AbelInvariants.of_core(native.abel_invariants(_core_coefficients(coefficients), _core_symbol(x)))


def constant_invariant(coefficients: CoefficientsLike, x: Symbol) -> Optional[bool]:
    """Whether the invariant ``I1`` is constant (the separable class), or
    ``None`` when this cannot be decided
    (:func:`sympy_extras.solvers.abel.constant_invariant` on SymPy objects).

    >>> from sympy import Symbol, Rational
    >>> from sympy_extras.solvers.abel_sympy import constant_invariant
    >>> x = Symbol('x')
    >>> constant_invariant((1, 0, 0, x**Rational(-3, 2)), x), constant_invariant((-1, -2*x, 0, 0), x)
    (True, False)
    """
    return native.constant_invariant(_core_coefficients(coefficients), _core_symbol(x))


def _transformed(coefficients: CoefficientsLike, x: Symbol, xi: Expr, P: Expr, Q: Expr, t: Symbol) -> Coefficients:
    """The coefficients of the equation for ``u(t)`` when ``y = P u + Q``,
    ``x = xi(t)`` (``P``, ``Q`` given in ``t``)."""
    return _sympy_coefficients(native._transformed(_core_coefficients(coefficients), _core_symbol(x),
                                                   core_expr(xi), core_expr(P), core_expr(Q), _core_symbol(t)))


def abel_equivalence(target: CoefficientsLike, x: Symbol, representative: CoefficientsLike, t: Symbol
                     ) -> Optional[AbelTransformation]:
    """A transformation ``y = P(x) u(xi(x)) + Q(x)`` taking the target
    equation (in ``x``) to the representative (in ``t``), or ``None``
    (:func:`sympy_extras.solvers.abel.abel_equivalence` on SymPy objects).

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy_extras.solvers.abel_sympy import abel_equivalence
    >>> x, t = symbols('x t')
    >>> abel_equivalence((-1, -2*(x - 1), 0, 0), x, (-1, -2*t, 0, 0), t)
    AbelTransformation(xi=x - 1, P=1, Q=0)
    """
    found = native.abel_equivalence(_core_coefficients(target), _core_symbol(x),
                                    _core_coefficients(representative), _core_symbol(t))
    return None if found is None else AbelTransformation.of_core(found)


def particular_solution(coefficients: CoefficientsLike, x: Symbol, degree: int = 2) -> Optional[Expr]:
    """A polynomial or rational particular solution ``Q(x)`` of ``y' = f3
    y**3 + f2 y**2 + f1 y + f0``, or ``None``
    (:func:`sympy_extras.solvers.abel.particular_solution` on SymPy objects).

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy_extras.solvers.abel_sympy import particular_solution
    >>> x = Symbol('x')
    >>> particular_solution((-1, -2*x, 0, 0), x)
    0
    >>> particular_solution((1, 0, -x**2, 1), x)         # y = x solves y' = y**3 - x**2 y + 1
    x
    """
    found = native.particular_solution(_core_coefficients(coefficients), _core_symbol(x), degree)
    return None if found is None else sympy_expr(found)


def air_solution(coefficients: CoefficientsLike, x: Symbol, f: AppliedUndef, Q: Optional[Expr] = None
                 ) -> Optional[Basic]:
    """An implicit solution of an equation of the AIR class through its
    inverse Riccati equation, or ``None``
    (:func:`sympy_extras.solvers.abel.air_solution` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.abel_sympy import air_solution
    >>> y = Function('y')(x)
    >>> air_solution((-1/x, 1/x**2, 0, 0), x, y)     # doctest: +ELLIPSIS
    Eq(..., C1)
    """
    found = native.air_solution(_core_coefficients(coefficients), _core_symbol(x), _core_function(f),
                                None if Q is None else core_expr(as_expr(Q)))
    return None if found is None else _sympy_basic(found)


def abel_by_invariants(equation: Basic, f: AppliedUndef,
                       representatives: Optional[Sequence[tuple[Coefficients, tuple[Expr, Expr]]]] = None
                       ) -> Optional[Basic]:
    """An implicit solution of an Abel equation of the first kind with a
    non-constant invariant, through the AIR class or the equivalence with
    a representative (``REPRESENTATIVES`` by default), or ``None``
    (:func:`sympy_extras.solvers.abel.abel_by_invariants` on SymPy objects).

    Examples
    ========

    >>> from sympy import Function
    >>> from sympy.abc import x
    >>> from sympy_extras.solvers.abel_sympy import abel_by_invariants
    >>> y = Function('y')(x)
    >>> abel_by_invariants(y.diff(x) + y**3 + 2*(x - 1)*y**2, y)     # doctest: +ELLIPSIS
    Eq(..., C1)
    """
    core_representatives: Optional[list[tuple[native.Coefficients, tuple[CoreExpr, CoreExpr]]]] = None
    if representatives is not None:
        core_representatives = [(_core_coefficients(c), (core_expr(g1), core_expr(g0)))
                                for c, (g1, g0) in representatives]
    found = native.abel_by_invariants(_core_basic(equation), _core_function(f), core_representatives)
    return None if found is None else _sympy_basic(found)
