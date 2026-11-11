"""The SymPy face of the Mellin transforms written as gamma quotients
(:mod:`sympy_extras.integrals.mellin`), for the integrators which are
still written on SymPy (:mod:`.slater`, :mod:`.marichev`, :mod:`.brackets`,
:mod:`.definite`, ...).

:class:`GammaQuotient`, :class:`Kernel`, :class:`Match`, :class:`Product`
and :class:`MellinTransform` are the containers of the native module on
SymPy objects, with ``to_core`` and ``from_core`` to the native classes;
their operations and the functions (:func:`monomial`,
:func:`mellin_kernel`, :func:`mellin_transform`,
:func:`decompose_integrand`, :func:`poles_separated`) convert their
arguments by :mod:`sympy_extras.interop` and run the native code. Nothing
of the module stays on SymPy.

Examples
========

>>> from sympy import exp, symbols, sin, besselj
>>> from sympy_extras.integrals.mellin_sympy import mellin_transform
>>> x, s = symbols('x s')
>>> mellin_transform(exp(-2*x), x, s)
MellinTransform(gamma(s)/2**s, (0, oo))
>>> mellin_transform(sin(x)/x, x, s)
MellinTransform(2**s*sqrt(pi)*gamma(s/2)/(4*gamma(3/2 - s/2)), (0, 2))
>>> mellin_transform(besselj(0, 3*x)/(x**2 + 4), x, s) is None
True
"""
from __future__ import annotations

from typing import Final, Optional, Sequence

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import Rational, oo
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.logic.boolalg import Boolean, true

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask_sympy import Assumptions, core_assumptions, core_boolean, sympy_boolean
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.numbers import Rational as CoreRational
from sympy_extras.core.symbol import Symbol as CoreSymbol
from sympy_extras.interop import from_sympy, to_sympy

from . import mellin as native

__all__ = ['GammaFactor', 'GammaQuotient', 'MellinTransform', 'Kernel', 'mellin_kernel',
           'mellin_transform', 'monomial', 'KERNELS', 'Match', 'Product', 'decompose_integrand', 'poles_separated',
           'BETA', 'BETA_LOWER']

#: `\\Gamma(\\text{shift} + \\text{scale} \\cdot s)`
GammaFactor = tuple[Expr, Rational]


def _sympy_dummy(d: CoreSymbol) -> Dummy:
    converted = to_sympy(d)
    if not isinstance(converted, Dummy):
        raise TypeError("%s is not a dummy" % (converted,))
    return converted


#: the variable of the ``extra`` factor of a quotient (the native one, converted)
_S: Final[Dummy] = _sympy_dummy(native._S)

#: the scale of the argument in the scale conditions of the kernels
BETA: Final[Dummy] = _sympy_dummy(native.BETA)

#: the name of the Beta kernel on (0, 1)
BETA_LOWER: Final[str] = native.BETA_LOWER


def _core_symbol(s: Basic) -> CoreSymbol:
    converted = from_sympy(s)
    if not isinstance(converted, CoreSymbol):
        raise TypeError("%s is not a symbol" % (s,))
    return converted


def _core(e: ExprLike) -> CoreExpr:
    return core_expr(sympify(e))


def _core_rational(e: Expr) -> CoreRational:
    converted = core_expr(e)
    if not isinstance(converted, CoreRational):
        raise TypeError("expected a rational number, got %s" % (e,))
    return converted


def _sympy_rational(e: CoreExpr) -> Rational:
    converted = sympy_expr(e)
    if not isinstance(converted, Rational):
        raise TypeError("expected a rational number, got %s" % (converted,))
    return converted


def _factor(shift: ExprLike, scale: ExprLike) -> GammaFactor:
    scale_ = as_expr(scale)
    if not isinstance(scale_, Rational):
        raise ValueError("the coefficient of s in a gamma factor must be rational, got %s" % (scale_,))
    return (as_expr(shift), scale_)


def _pair(e: Optional[tuple[CoreExpr, CoreExpr]]) -> Optional[tuple[Expr, Expr]]:
    return None if e is None else (sympy_expr(e[0]), sympy_expr(e[1]))


class GammaQuotient:
    """:class:`sympy_extras.integrals.mellin.GammaQuotient` on SymPy
    objects: a constant times powers ``rho**(c s)`` times an ``extra``
    factor times a quotient of products of ``gamma(a + b s)`` (``b``
    rational), in the strip ``lower < Re s < upper`` under ``condition``.

    Examples
    ========

    >>> from sympy import S, symbols
    >>> from sympy_extras.integrals.mellin_sympy import GammaQuotient
    >>> s = symbols('s')
    >>> q = GammaQuotient(1, [(2, -1)], [(0, S.One)], [], 0)
    >>> q.as_expr(s)
    gamma(s)/2**s
    >>> q.compose(1, -2).as_expr(s)
    2**(2*s)*gamma(1 - 2*s)/2
    >>> q.compose(1, -2).strip
    (-oo, 1/2)
    """

    __slots__ = ('constant', 'bases', 'numerator', 'denominator', 'lower', 'upper', 'condition', 'extra')

    def __init__(self, constant: ExprLike, bases: Sequence[tuple[ExprLike, ExprLike]],
                 numerator: Sequence[tuple[ExprLike, ExprLike]],
                 denominator: Sequence[tuple[ExprLike, ExprLike]],
                 lower: ExprLike = -oo, upper: ExprLike = oo, condition: object = True,
                 extra: ExprLike = 1) -> None:
        self.constant: Final[Expr] = as_expr(constant)
        self.bases: Final[tuple[tuple[Expr, Expr], ...]] = tuple(
            (as_expr(b), as_expr(c)) for b, c in bases if as_expr(c) != 0)
        self.numerator: Final[tuple[GammaFactor, ...]] = tuple(_factor(a, b) for a, b in numerator)
        self.denominator: Final[tuple[GammaFactor, ...]] = tuple(_factor(a, b) for a, b in denominator)
        self.lower: Final[Expr] = as_expr(lower)
        self.upper: Final[Expr] = as_expr(upper)
        self.condition: Final[Boolean] = as_boolean(condition)
        self.extra: Final[Expr] = as_expr(extra)

    def to_core(self) -> native.GammaQuotient:
        """The same quotient on the objects of the core."""
        return native.GammaQuotient(
            core_expr(self.constant), [(core_expr(b), core_expr(c)) for b, c in self.bases],
            [(core_expr(a), _core_rational(b)) for a, b in self.numerator],
            [(core_expr(a), _core_rational(b)) for a, b in self.denominator],
            core_expr(self.lower), core_expr(self.upper), core_boolean(self.condition), core_expr(self.extra))

    @staticmethod
    def from_core(q: native.GammaQuotient) -> GammaQuotient:
        """A quotient of the core on SymPy objects."""
        return GammaQuotient(
            sympy_expr(q.constant), [(sympy_expr(b), sympy_expr(c)) for b, c in q.bases],
            [(sympy_expr(a), _sympy_rational(b)) for a, b in q.numerator],
            [(sympy_expr(a), _sympy_rational(b)) for a, b in q.denominator],
            sympy_expr(q.lower), sympy_expr(q.upper), sympy_boolean(q.condition), sympy_expr(q.extra))

    @property
    def strip(self) -> tuple[Expr, Expr]:
        """The strip of convergence ``(lower, upper)``."""
        return (self.lower, self.upper)

    def __repr__(self) -> str:
        return "GammaQuotient(%s, %s)" % (self.as_expr(Symbol('s')), self.strip)

    def as_expr(self, s: ExprLike) -> Expr:
        """The value at ``s`` as a SymPy expression."""
        return sympy_expr(self.to_core().as_expr(_core(s)))

    def scaled(self, factor: ExprLike) -> GammaQuotient:
        """The quotient multiplied by a constant."""
        return GammaQuotient.from_core(self.to_core().scaled(_core(factor)))

    def times(self, other: GammaQuotient) -> GammaQuotient:
        """The product of two quotients on the intersection of their strips."""
        return GammaQuotient.from_core(self.to_core().times(other.to_core()))

    def compose(self, p: ExprLike, q: ExprLike) -> GammaQuotient:
        """The quotient at ``p + q*s`` (``q`` a nonzero rational number)."""
        return GammaQuotient.from_core(self.to_core().compose(_core(p), _core(q)))

    def cancelled(self) -> GammaQuotient:
        """The quotient with the gamma factors common to the numerator and
        the denominator removed."""
        return GammaQuotient.from_core(self.to_core().cancelled())

    def strip_condition(self, s: ExprLike) -> Boolean:
        """The condition for ``s`` (a real number) to lie in the strip."""
        return sympy_boolean(self.to_core().strip_condition(_core(s)))

    def is_gamma_quotient(self) -> bool:
        """Whether there is no ``extra`` factor."""
        return self.extra == 1


def poles_separated(quotient: GammaQuotient) -> Optional[bool]:
    """:func:`sympy_extras.integrals.mellin.poles_separated` on SymPy
    objects.

    >>> from sympy_extras.integrals.mellin_sympy import GammaQuotient, poles_separated
    >>> poles_separated(GammaQuotient(1, [], [(0, 1)], [(1, 1)], 0, 1))
    True
    >>> poles_separated(GammaQuotient(1, [], [(0, 1)], [(1, 1)], -1, 0))
    False
    """
    return native.poles_separated(quotient.to_core())


class MellinTransform:
    """A Mellin transform on SymPy objects: the transform as an expression
    in ``s``, its strip of convergence and the conditions on the
    parameters."""

    __slots__ = ('transform', 'strip', 'condition')

    def __init__(self, transform: Expr, strip: tuple[Expr, Expr], condition: Boolean = true) -> None:
        self.transform: Final[Expr] = transform
        self.strip: Final[tuple[Expr, Expr]] = strip
        self.condition: Final[Boolean] = condition

    @staticmethod
    def from_core(t: native.MellinTransform) -> MellinTransform:
        """A transform of the core on SymPy objects."""
        return MellinTransform(sympy_expr(t.transform), (sympy_expr(t.strip[0]), sympy_expr(t.strip[1])),
                               sympy_boolean(t.condition))

    def __repr__(self) -> str:
        if self.condition is true:
            return "MellinTransform(%s, (%s, %s))" % (self.transform, self.strip[0], self.strip[1])
        return "MellinTransform(%s, (%s, %s), %s)" % (self.transform, self.strip[0], self.strip[1],
                                                       self.condition)


class Kernel:
    """:class:`sympy_extras.integrals.mellin.Kernel` on SymPy objects: one
    entry of the table, a function with its Mellin transform (``quotient``),
    its ``parameters``, the ``scale_condition`` on :data:`BETA` and the
    ``parity``."""

    __slots__ = ('name', 'quotient', 'parameters', 'scale_condition', 'parity')

    def __init__(self, name: str, quotient: GammaQuotient, parameters: Sequence[Expr] = (),
                 scale_condition: object = True, parity: int = 0) -> None:
        self.name: Final[str] = name
        self.quotient: Final[GammaQuotient] = quotient
        self.parameters: Final[tuple[Expr, ...]] = tuple(parameters)
        self.scale_condition: Final[Boolean] = as_boolean(scale_condition)
        self.parity: Final[int] = parity

    def to_core(self) -> native.Kernel:
        """The same kernel on the objects of the core."""
        return native.Kernel(self.name, self.quotient.to_core(), [core_expr(p) for p in self.parameters],
                             core_boolean(self.scale_condition), self.parity)

    @staticmethod
    def from_core(k: native.Kernel) -> Kernel:
        """A kernel of the core on SymPy objects: an entry of the native
        :data:`KERNELS` is the same entry of this module's table."""
        known = _TABLE.get(k.name)
        if known is not None and known[0] is k:
            return known[1]
        return Kernel(k.name, GammaQuotient.from_core(k.quotient), [sympy_expr(p) for p in k.parameters],
                      sympy_boolean(k.scale_condition), k.parity)

    def __repr__(self) -> str:
        return "Kernel(%s)" % self.name


#: the native kernels without parameters and their counterparts here, by name
_TABLE: Final[dict[str, tuple[native.Kernel, Kernel]]] = {}
for _name, _kernel in native.KERNELS.items():
    _TABLE[_name] = (_kernel, Kernel.from_core(_kernel))

#: the kernels without parameters, by the head of the expression they match
KERNELS: Final[dict[str, Kernel]] = {name: pair[1] for name, pair in _TABLE.items()}


def _beta_kernel(b: Expr) -> Kernel:
    """The Beta kernel ``(1 - x)**(b - 1)`` on ``(0, 1)``."""
    return Kernel.from_core(native._beta_kernel(core_expr(b)))


class Match:
    """:class:`sympy_extras.integrals.mellin.Match` on SymPy objects: a
    factor of an integrand recognised as ``constant * f(beta * x**gamma)``
    for a kernel ``f``."""

    __slots__ = ('kernel', 'beta', 'gamma', 'constant', 'cutoff')

    def __init__(self, kernel: Kernel, beta: Expr, gamma_: Expr, constant: Expr = S.One,
                 cutoff: bool = False) -> None:
        self.kernel: Final[Kernel] = kernel
        self.beta: Final[Expr] = beta
        self.gamma: Final[Expr] = gamma_
        self.constant: Final[Expr] = constant
        #: whether the kernel includes the cutoff at 1 of a finite range
        self.cutoff: Final[bool] = cutoff

    def to_core(self) -> native.Match:
        """The same match on the objects of the core."""
        return native.Match(self.kernel.to_core(), core_expr(self.beta), core_expr(self.gamma),
                            core_expr(self.constant), self.cutoff)

    @staticmethod
    def from_core(m: native.Match) -> Match:
        """A match of the core on SymPy objects."""
        return Match(Kernel.from_core(m.kernel), sympy_expr(m.beta), sympy_expr(m.gamma), sympy_expr(m.constant),
                     m.cutoff)

    def __repr__(self) -> str:
        return "Match(%s, beta=%s, gamma=%s, constant=%s)" % (self.kernel.name, self.beta, self.gamma, self.constant)

    def quotient(self, assumptions: Assumptions = None) -> GammaQuotient:
        """The Mellin transform of the factor
        (:meth:`sympy_extras.integrals.mellin.Match.quotient`)."""
        return GammaQuotient.from_core(self.to_core().quotient(core_assumptions(assumptions)))


class Product:
    """:class:`sympy_extras.integrals.mellin.Product` on SymPy objects: an
    integrand ``constant * x**alpha * log(x)**n * k_1(...) * k_2(...)``, on
    ``(0, 1)`` with a factor ``log(1 - x)**m`` as well
    (``one_minus_power``)."""

    __slots__ = ('constant', 'alpha', 'log_power', 'matches', 'one_minus_power')

    def __init__(self, constant: Expr, alpha: Expr, log_power: int, matches: Sequence[Match],
                 one_minus_power: int = 0) -> None:
        self.constant: Final[Expr] = constant
        self.alpha: Final[Expr] = alpha
        self.log_power: Final[int] = log_power
        self.matches: Final[tuple[Match, ...]] = tuple(matches)
        self.one_minus_power: Final[int] = one_minus_power

    def to_core(self) -> native.Product:
        """The same product on the objects of the core."""
        return native.Product(core_expr(self.constant), core_expr(self.alpha), self.log_power,
                              [m.to_core() for m in self.matches], self.one_minus_power)

    @staticmethod
    def from_core(p: native.Product) -> Product:
        """A product of the core on SymPy objects."""
        return Product(sympy_expr(p.constant), sympy_expr(p.alpha), p.log_power,
                       [Match.from_core(m) for m in p.matches], p.one_minus_power)

    def __repr__(self) -> str:
        return "Product(%s, x**%s, log**%s, %s)" % (self.constant, self.alpha, self.log_power,
                                                  list(self.matches))

    def quotient(self, assumptions: Assumptions = None) -> Optional[GammaQuotient]:
        """The Mellin transform of the integrand without the logarithm
        (:meth:`sympy_extras.integrals.mellin.Product.quotient`)."""
        q = self.to_core().quotient(core_assumptions(assumptions))
        return None if q is None else GammaQuotient.from_core(q)


def monomial(e: Expr, x: Symbol) -> Optional[tuple[Expr, Expr]]:
    """:func:`sympy_extras.integrals.mellin.monomial` on SymPy objects:
    ``(beta, gamma)`` when ``e == beta * x**gamma`` with ``beta`` free of
    ``x`` and ``gamma`` a nonzero rational number, ``None`` otherwise.

    >>> from sympy import sqrt, symbols
    >>> from sympy_extras.integrals.mellin_sympy import monomial
    >>> x, a = symbols('x a')
    >>> monomial(3*a*sqrt(x), x)
    (3*a, 1/2)
    >>> monomial(x + 1, x) is None
    True
    """
    return _pair(native.monomial(core_expr(e), _core_symbol(x)))


def mellin_kernel(f: Expr, x: Symbol, cutoff: Optional[str] = None) -> Optional[Match]:
    """:func:`sympy_extras.integrals.mellin.mellin_kernel` on SymPy
    objects: ``f`` recognised as ``constant * k(beta * x**gamma)`` for a
    kernel ``k`` of the table, or ``None``.

    Examples
    ========

    >>> from sympy import exp, symbols, besselj, sqrt
    >>> from sympy_extras.integrals.mellin_sympy import mellin_kernel
    >>> x, a = symbols('x a')
    >>> mellin_kernel(exp(-3*x**2), x)
    Match(exp, beta=3, gamma=2, constant=1)
    >>> mellin_kernel(1/(x + 2)**a, x)
    Match((1 + x)**(-a), beta=1/2, gamma=1, constant=2**(-a))
    >>> mellin_kernel(besselj(0, sqrt(x)), x)
    Match(besselj, beta=1, gamma=1/2, constant=1)
    >>> mellin_kernel(exp(x), x) is None
    True
    """
    found = native.mellin_kernel(core_expr(f), _core_symbol(x), cutoff)
    return None if found is None else Match.from_core(found)


def mellin_transform(f: ExprLike, x: Symbol, s: Symbol) -> Optional[MellinTransform]:
    """:func:`sympy_extras.integrals.mellin.mellin_transform` on SymPy
    objects: the Mellin transform of ``f`` looked up in the table, with its
    strip of convergence, ``None`` when ``f`` is not a constant times a
    power of ``x`` times at most two kernels.

    Examples
    ========

    >>> from sympy import exp, symbols, sqrt, log
    >>> from sympy_extras.integrals.mellin_sympy import mellin_transform
    >>> x, s, a = symbols('x s a')
    >>> mellin_transform(exp(-x**2)/sqrt(x), x, s)
    MellinTransform(gamma(s/2 - 1/4)/2, (1/2, oo))
    >>> mellin_transform(1/(1 + x)**a, x, s)
    MellinTransform(gamma(s)*gamma(a - s)/gamma(a), (0, a))
    >>> mellin_transform(exp(-x)*log(x), x, s)
    MellinTransform(gamma(s)*polygamma(0, s), (0, oo))
    """
    found = native.mellin_transform(_core(f), _core_symbol(x), _core_symbol(s))
    return None if found is None else MellinTransform.from_core(found)


def decompose_integrand(f: Expr, x: Symbol, cutoff: Optional[str] = None) -> Optional[Product]:
    """:func:`sympy_extras.integrals.mellin.decompose_integrand` on SymPy
    objects: ``f`` split into a constant, a power of ``x``, a power of
    ``log(x)`` and kernel factors; ``None`` when a factor is not recognised
    or there are more than two kernels.

    >>> from sympy import exp, log, symbols
    >>> from sympy_extras.integrals.mellin_sympy import decompose_integrand
    >>> x = symbols('x')
    >>> decompose_integrand(3*x**2*log(x)*exp(-x), x)
    Product(3, x**2, log**1, [Match(exp, beta=1, gamma=1, constant=1)])
    """
    found = native.decompose_integrand(core_expr(f), _core_symbol(x), cutoff)
    return None if found is None else Product.from_core(found)
