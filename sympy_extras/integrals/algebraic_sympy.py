"""The SymPy face of the rationalising substitutions for algebraic
integrands of genus zero (:mod:`sympy_extras.integrals.algebraic`), for
the integrators which are still written on SymPy (:mod:`.definite`,
:mod:`.indefinite`).

The functions convert their arguments by :mod:`sympy_extras.interop`, run
the native code and convert the results back (the values of
:mod:`.conditions_sympy`, :class:`Rationalisation` and
:class:`BinomialDifferential` with SymPy attributes). The steps of the
native module which stay on SymPy are here too, on core expressions: the
rational integral by
:func:`~sympy_extras.integrals.definite.conditional_integral`
(:func:`conditional_integral_of_core`, of :mod:`.laplace_sympy`) and its
numerical check by
:func:`~sympy_extras.integrals.definite.verify_numerically`
(:func:`verify_numerically_of_core`), both not ported yet.

Examples
========

>>> from sympy import symbols, sqrt, oo
>>> from sympy_extras.integrals.algebraic_sympy import algebraic_integral
>>> x = symbols('x')
>>> algebraic_integral(1/(x*sqrt(x**2 - 1)), x, 1, 2)
ConditionalValue(pi/3)
>>> algebraic_integral(sqrt(x)/(1 + x)**2, x, 0, oo)
ConditionalValue(pi/2)
>>> algebraic_integral(sqrt((1 - x)/(1 + x)), x, 0, 1)
ConditionalValue(-1 + pi/2)
"""
from __future__ import annotations

from typing import Final, Optional

from sympy.core.expr import Expr
from sympy.core.numbers import Rational
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify

from sympy_extras._numeric_sympy import core_expr, sympy_expr
from sympy_extras._typing_sympy import ExprLike, as_expr
from sympy_extras.assumptions import ask as native_ask
from sympy_extras.assumptions.ask_sympy import Assumptions, _core_symbol, _sympy_symbol, core_assumptions
from sympy_extras.core.expr import Expr as CoreExpr
from sympy_extras.core.numbers import Rational as CoreRational
from sympy_extras.core.symbol import Symbol as CoreSymbol

from . import algebraic as native
from . import conditions as native_conditions
from .conditions_sympy import ConditionalValue
from .laplace_sympy import _sympy_assumptions, conditional_integral_of_core

__all__ = ['algebraic_integral', 'euler_substitution', 'euler_substitutions', 'binomial_differential', 'moebius_root',
           'rationalised_integral', 'Rationalisation', 'BinomialDifferential', 'binomial_exponents',
           'chebyshev_elementary', 'binomial_hypergeometric_antiderivative', 'conditional_integral_of_core',
           'verify_numerically_of_core']


def _conditional(found: Optional[native_conditions.ConditionalValue]) -> Optional[ConditionalValue]:
    return None if found is None else ConditionalValue.from_core(found)


def _rational(r: CoreRational) -> Rational:
    return Rational(r.p, r.q)


def _core_rational(r: Rational) -> CoreRational:
    converted = core_expr(r)
    if not isinstance(converted, CoreRational):
        raise TypeError("expected a rational number, got %s" % (r,))
    return converted


class Rationalisation:
    """:class:`sympy_extras.integrals.algebraic.Rationalisation` on SymPy
    objects: a rationalising substitution ``x = X(t)``, the new variable,
    the map, the inverse ``t(x)`` and the replacement of the radical atoms
    of the integrand by rational functions of ``t``."""

    def __init__(self, t: Symbol, x_of_t: Expr, t_of_x: Expr, radicals: dict[Expr, Expr]) -> None:
        self.t: Final[Symbol] = t
        self.x_of_t: Final[Expr] = x_of_t
        self.t_of_x: Final[Expr] = t_of_x
        self.radicals: Final[dict[Expr, Expr]] = dict(radicals)

    def __repr__(self) -> str:
        return "Rationalisation(x = %s, t = %s)" % (self.x_of_t, self.t_of_x)

    @staticmethod
    def from_core(r: native.Rationalisation) -> Rationalisation:
        return Rationalisation(_sympy_symbol(r.t), sympy_expr(r.x_of_t), sympy_expr(r.t_of_x),
                               {sympy_expr(k): sympy_expr(v) for k, v in r.radicals.items()})

    def to_core(self) -> native.Rationalisation:
        return native.Rationalisation(_core_symbol(self.t), core_expr(self.x_of_t), core_expr(self.t_of_x),
                                      {core_expr(k): core_expr(v) for k, v in self.radicals.items()})


class BinomialDifferential:
    """:class:`sympy_extras.integrals.algebraic.BinomialDifferential` on
    SymPy objects: the exponents and coefficients of
    ``x**m*(a + b*x**n)**p``, ``n > 0``."""

    def __init__(self, m: Rational, a: Expr, b: Expr, n: Rational, p: Rational) -> None:
        self.m: Final[Rational] = m
        self.a: Final[Expr] = a
        self.b: Final[Expr] = b
        self.n: Final[Rational] = n
        self.p: Final[Rational] = p

    def __repr__(self) -> str:
        return "BinomialDifferential(x**(%s)*(%s + %s*x**(%s))**(%s))" % (self.m, self.a, self.b, self.n, self.p)

    @staticmethod
    def from_core(b: native.BinomialDifferential) -> BinomialDifferential:
        return BinomialDifferential(_rational(b.m), sympy_expr(b.a), sympy_expr(b.b), _rational(b.n), _rational(b.p))

    def to_core(self) -> native.BinomialDifferential:
        return native.BinomialDifferential(_core_rational(self.m), core_expr(self.a), core_expr(self.b),
                                           _core_rational(self.n), _core_rational(self.p))


def euler_substitutions(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                        assumptions: Assumptions = None) -> list[Rationalisation]:
    """:func:`sympy_extras.integrals.algebraic.euler_substitutions` on SymPy
    objects: Euler's substitutions applicable to an integrand whose
    radicals are half-integer powers of one polynomial ``Q`` of degree one
    or two, in the order ``a > 0``, ``c > 0``, real roots.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic_sympy import euler_substitutions
    >>> x = symbols('x')
    >>> [s.t_of_x for s in euler_substitutions(1/sqrt(x**2 + 1), x, 0, 1)]
    [-x + sqrt(x**2 + 1), (sqrt(x**2 + 1) - 1)/x]
    """
    found = native.euler_substitutions(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                       core_expr(sympify(b)), core_assumptions(assumptions))
    return [Rationalisation.from_core(s) for s in found]


def euler_substitution(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                       assumptions: Assumptions = None) -> Optional[Rationalisation]:
    """The first of :func:`euler_substitutions`, or ``None``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic_sympy import euler_substitution
    >>> x = symbols('x')
    >>> euler_substitution(1/sqrt(x**2 + 1), x, 0, 1).t_of_x
    -x + sqrt(x**2 + 1)
    """
    found = euler_substitutions(f, x, a, b, assumptions)
    return found[0] if found else None


def moebius_root(f: ExprLike, x: Symbol) -> Optional[Rationalisation]:
    """:func:`sympy_extras.integrals.algebraic.moebius_root` on SymPy
    objects: the substitution ``t**n = M(x)`` for an integrand whose
    radicals are rational powers of one Möbius (or linear) function ``M``.

    >>> from sympy import symbols, cbrt
    >>> from sympy_extras.integrals.algebraic_sympy import moebius_root
    >>> x, t = symbols('x t')
    >>> s = moebius_root(cbrt(x + 1)/x, x)
    >>> s.x_of_t.subs(s.t, t), s.t_of_x
    (t**3 - 1, (x + 1)**(1/3))
    """
    found = native.moebius_root(core_expr(sympify(f)), _core_symbol(x))
    return None if found is None else Rationalisation.from_core(found)


def binomial_differential(f: ExprLike, x: Symbol) -> Optional[tuple[Expr, Expr, Expr, Expr, Expr, Expr]]:
    """:func:`sympy_extras.integrals.algebraic.binomial_differential` on
    SymPy objects: ``(C, m, a, b, n, p)`` with ``f == C*x**m*(a +
    b*x**n)**p``.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic_sympy import binomial_differential
    >>> x = symbols('x')
    >>> binomial_differential(3*x**3*sqrt(x**2 + 1), x)
    (3, 3, 1, 1, 2, 1/2)
    """
    found = native.binomial_differential(core_expr(sympify(f)), _core_symbol(x))
    if found is None:
        return None
    C, m, a, b, n, p = found
    return (sympy_expr(C), sympy_expr(m), sympy_expr(a), sympy_expr(b), sympy_expr(n), sympy_expr(p))


def binomial_exponents(f: ExprLike, x: Symbol) -> Optional[BinomialDifferential]:
    """:func:`sympy_extras.integrals.algebraic.binomial_exponents` on SymPy
    objects: the binomial differential ``x**m*(a + b*x**n)**p`` of which
    ``f`` is a locally constant multiple, ``None`` when ``f`` is none.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic_sympy import binomial_exponents
    >>> x = symbols('x')
    >>> binomial_exponents(sqrt(x**7/(1 - 5*x**2)), x)
    BinomialDifferential(x**(7/2)*(1 + -5*x**(2))**(-1/2))
    """
    found = native.binomial_exponents(core_expr(sympify(f)), _core_symbol(x))
    return None if found is None else BinomialDifferential.from_core(found)


def chebyshev_elementary(binomial: BinomialDifferential) -> bool:
    """:func:`sympy_extras.integrals.algebraic.chebyshev_elementary` on
    SymPy objects: whether the integral of the binomial differential is
    elementary.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic_sympy import binomial_exponents, chebyshev_elementary
    >>> x = symbols('x')
    >>> chebyshev_elementary(binomial_exponents(sqrt(x**3 + 1), x))
    False
    """
    return native.chebyshev_elementary(binomial.to_core())


def binomial_hypergeometric_antiderivative(f: ExprLike, x: Symbol) -> Optional[Expr]:
    """:func:`sympy_extras.integrals.algebraic.binomial_hypergeometric_antiderivative`
    on SymPy objects: an antiderivative of a binomial differential whose
    integral is not elementary, by the Gauss hypergeometric function.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic_sympy import binomial_hypergeometric_antiderivative
    >>> x = symbols('x')
    >>> binomial_hypergeometric_antiderivative(sqrt(x**3 + 1), x)
    x*hyper((-1/2, 1/3), (4/3,), -x**3)
    >>> binomial_hypergeometric_antiderivative(sqrt(x**7/(1 - 5*x**2)), x)
    Piecewise((2*sqrt(5)*x*sqrt(x**7/(1 - 5*x**2))*sqrt(5 - 1/x**2)*hyper((-7/4, 1/2), (-3/4,), 1/(5*x**2))/35, x < -sqrt(5)/5), (2*x*sqrt(x**7/(1 - 5*x**2))*sqrt(1 - 5*x**2)*hyper((1/2, 9/4), (13/4,), 5*x**2)/9, True))
    """
    found = native.binomial_hypergeometric_antiderivative(core_expr(sympify(f)), _core_symbol(x))
    return None if found is None else sympy_expr(found)


def rationalised_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike, substitution: Rationalisation,
                          assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.algebraic.rationalised_integral` on
    SymPy objects: ``Integral(f, (x, a, b))`` through a rationalising
    substitution.

    >>> from sympy import symbols, sqrt
    >>> from sympy_extras.integrals.algebraic_sympy import moebius_root, rationalised_integral
    >>> x = symbols('x')
    >>> f = sqrt(x)/(1 + x)
    >>> rationalised_integral(f, x, 0, 1, moebius_root(f, x))
    ConditionalValue(2 - pi/2)
    """
    return _conditional(native.rationalised_integral(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                                     core_expr(sympify(b)), substitution.to_core(),
                                                     core_assumptions(assumptions)))


def algebraic_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                       assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """:func:`sympy_extras.integrals.algebraic.algebraic_integral` on SymPy
    objects: ``Integral(f, (x, a, b))`` for an algebraic integrand of
    genus zero (the roots of a Möbius function, Euler's substitutions,
    Chebyshev's binomial differentials); ``None`` when none applies.

    Examples
    ========

    >>> from sympy import symbols, sqrt, cbrt
    >>> from sympy_extras.integrals.algebraic_sympy import algebraic_integral
    >>> x = symbols('x')
    >>> algebraic_integral(1/sqrt(x**2 + 1), x, 0, 1)
    ConditionalValue(-log(-1 + sqrt(2)))
    >>> algebraic_integral(x**3*sqrt(x**2 + 1), x, 0, 1)
    ConditionalValue(2*(1 + sqrt(2))/15)
    >>> algebraic_integral(cbrt(x)/(1 + x), x, 0, 1)
    ConditionalValue(-sqrt(3)*pi/3 - log(2) + 3)
    """
    return _conditional(native.algebraic_integral(core_expr(sympify(f)), _core_symbol(x), core_expr(sympify(a)),
                                                  core_expr(sympify(b)), core_assumptions(assumptions)))


# ---------------------------------------------------------------------------
# the steps of the native module which stay on SymPy

def verify_numerically_of_core(value: CoreExpr, f: CoreExpr, x: CoreSymbol, a: CoreExpr, b: CoreExpr,
                               assumptions: native_ask.Assumptions) -> Optional[bool]:
    """:func:`~sympy_extras.integrals.definite.verify_numerically` of
    ``value`` against ``Integral(f, (x, a, b))``, on core expressions (not
    ported yet): ``False`` when a quadrature disagrees, ``True`` when it
    agrees, ``None`` when nothing was checked or the expressions cannot be
    converted."""
    from .definite import verify_numerically
    try:
        converted = (sympy_expr(value), sympy_expr(f), _sympy_symbol(x), sympy_expr(a), sympy_expr(b))
    except (TypeError, ValueError, NotImplementedError):
        return None
    v, integrand, variable, lower, upper = converted
    return verify_numerically(as_expr(v), integrand, variable, lower, upper, _sympy_assumptions(assumptions))
