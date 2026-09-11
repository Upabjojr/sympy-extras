"""Definite integrals by table lookup: a table of closed forms from
Gradshteyn and Ryzhik with the conditions on their parameters.

Mathematica's implementation notes say that "many definite integrals
are done by table lookup with the conditions on the parameters", and
every computer algebra system keeps such a table for the integrals which
no algorithm reaches: the log-sine integrals, the Dirichlet and Fejér
kernels, products of Bessel functions, Fourier transforms of hyperbolic
functions. The entries here are taken from [GR]_ (the values are facts
and are cited by section; nothing of the book's presentation is copied)
and each one is checked numerically in the tests at sample values of
its parameters, so that a transcription error cannot survive.

An entry is a pattern in the integration variable and ``Wild``
parameters, a range, the value in the same parameters, a condition on
them and the reference. :func:`lookup` matches the integrand (as given,
and in a few rewritten forms) against the entries of the range, and
:func:`table_integral` returns the value with the conditions the
assumptions do not settle, as a
:class:`~sympy_extras.integrals.conditions.ConditionalValue`.

Examples
========

>>> from sympy import symbols, sin, log, pi, oo, besselj
>>> from sympy_extras.integrals.tables import table_integral
>>> x = symbols('x')
>>> a = symbols('a', positive=True)
>>> table_integral(sin(a*x)**3/x**3, x, 0, oo)
ConditionalValue(3*pi*a**2/8)
>>> table_integral(log(sin(x))**2, x, 0, pi/2)
ConditionalValue(pi*(log(2)**2 + pi**2/12)/2)
>>> table_integral(besselj(2, x)**2/x, x, 0, oo)
ConditionalValue(1/4)

References
==========

.. [GR] I. S. Gradshteyn, I. M. Ryzhik, *Table of Integrals, Series, and
   Products*, 7th edition, Academic Press, 2007.
"""
from __future__ import annotations

from typing import Optional

from sympy.core.expr import Expr
from sympy.core.numbers import Integer, Rational, oo, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, tanh
from sympy.functions.elementary.miscellaneous import Min, sqrt
from sympy.functions.elementary.trigonometric import atan, cos, cot, sin, tan
from sympy.functions.special.bessel import besselj
from sympy.functions.special.error_functions import erf, erfc
from sympy.functions.special.gamma_functions import gamma, loggamma
from sympy.functions.special.zeta_functions import zeta
from sympy.core.relational import Ne
from sympy.logic.boolalg import And, Boolean, true
from sympy.simplify.powsimp import powsimp
from sympy.simplify.trigsimp import trigsimp

from sympy_extras._typing import ExprLike, as_boolean, as_expr
from sympy_extras.assumptions.ask import Assumptions
from sympy_extras.assumptions.facts import element
from .conditions import ConditionalValue, decide

__all__ = ['TableEntry', 'TABLE', 'lookup', 'table_integral', 'X', 'A', 'B', 'N', 'M']

#: the integration variable of the patterns
X = Dummy('x')
#: the parameters of the patterns (they never match the variable)
A = Wild('a', exclude=[X])
B = Wild('b', exclude=[X])
N = Wild('n', exclude=[X])
M = Wild('m', exclude=[X])

#: a sample of parameter values (for the numerical checks of the tests)
Sample = dict[Wild, Expr]


class TableEntry:
    """One entry of the table.

    Attributes
    ==========

    pattern : Expr
        The integrand in :data:`X` and the Wild parameters.
    lower, upper : Expr
        The range.
    value : Expr
        The value in the parameters.
    condition : Boolean
        The condition on the parameters under which the value holds.
    reference : str
        The section of Gradshteyn and Ryzhik.
    samples : tuple of dict
        Parameter values satisfying the condition, for the numerical
        checks of the tests.
    oscillatory : bool
        Whether the integrand oscillates on an infinite range (the checks
        then use ``mpmath.quadosc``).
    """

    def __init__(self, pattern: ExprLike, lower: ExprLike, upper: ExprLike, value: ExprLike,
                 condition: object, reference: str, samples: tuple[Sample, ...] = ({},),
                 oscillatory: bool = False) -> None:
        self.pattern = as_expr(pattern)
        self.lower = as_expr(lower)
        self.upper = as_expr(upper)
        self.value = as_expr(value)
        self.condition = as_boolean(condition)
        self.reference = reference
        self.samples = samples
        self.oscillatory = oscillatory

    def __repr__(self) -> str:
        return "TableEntry(%s, (%s, %s), %s)" % (self.pattern, self.lower, self.upper, self.reference)

    def integrand(self, x: Symbol, sample: Sample) -> Expr:
        """The integrand in ``x`` at the sample values."""
        return as_expr(self.pattern.xreplace(dict(sample)).xreplace({X: x}))


def _positive(*symbols: Wild) -> Boolean:
    return as_boolean(And(*[s > 0 for s in symbols]))


def _natural(*symbols: Wild) -> Boolean:
    return as_boolean(And(*[element(s, S.Naturals0) for s in symbols]))


_HALF = S.Half

#: the table
TABLE: list[TableEntry] = [
    # ---------------------------------------------------- trigonometric, GR 3.6
    TableEntry(cos(N * X) / (1 - 2 * A * cos(X) + A**2), 0, pi, pi * A**N / (1 - A**2),
               _natural(N) & (A**2 < 1), 'GR 3.613.1', ({A: Rational(1, 3), N: Integer(2)}, {A: -Rational(1, 2), N: Integer(3)})),
    TableEntry(cos(X)**N * cos(N * X), 0, pi / 2, pi / 2**(N + 1), _natural(N), 'GR 3.631.9',
               ({N: Integer(3)}, {N: Integer(5)})),
    TableEntry(cos(X)**N * cos(N * X), 0, pi, pi / 2**N, _natural(N), 'GR 3.631.9',
               ({N: Integer(2)}, {N: Integer(4)})),
    TableEntry(sin((N + _HALF) * X) / sin(X / 2), 0, pi, pi, _natural(N), 'GR 3.612.7 (the Dirichlet kernel)',
               ({N: Integer(2)}, {N: Integer(5)})),
    TableEntry(sin(N * X)**2 / sin(X)**2, 0, pi, N * pi, _natural(N), 'GR 3.624.6 (the Fejer kernel)',
               ({N: Integer(3)}, {N: Integer(4)})),
    TableEntry(1 / (A + B * cos(X)), 0, pi, pi / sqrt(A**2 - B**2), (A > 0) & (A**2 > B**2), 'GR 3.613.1',
               ({A: Integer(3), B: Integer(2)}, {A: Integer(5), B: -Integer(3)})),
    TableEntry(1 / (A**2 * cos(X)**2 + B**2 * sin(X)**2), 0, pi / 2, pi / (2 * A * B), _positive(A, B), 'GR 3.642.1',
               ({A: Integer(2), B: Integer(3)}, {A: Rational(1, 2), B: Integer(1)})),
    TableEntry(1 / (A**2 * cos(X)**2 + B**2 * sin(X)**2)**2, 0, pi / 2, pi * (A**2 + B**2) / (4 * A**3 * B**3),
               _positive(A, B), 'GR 3.642.3', ({A: Integer(2), B: Integer(3)}, {A: Rational(1, 2), B: Integer(1)})),
    TableEntry(X * cot(X), 0, pi / 2, pi * log(2) / 2, true, 'GR 3.747.7'),
    TableEntry(X / sin(X), 0, pi / 2, 2 * S.Catalan, true, 'GR 3.747.1'),
    TableEntry(X**2 / sin(X)**2, 0, pi / 2, pi * log(2), true, 'GR 3.747.9'),
    # -------------------------------------------- sine integrals over the half line, GR 3.8
    TableEntry(sin(A * X)**2 / X**2, 0, oo, pi * A / 2, A > 0, 'GR 3.821.9', ({A: Integer(2)}, {A: Rational(1, 3)}),
               oscillatory=True),
    TableEntry(sin(A * X)**3 / X**3, 0, oo, 3 * pi * A**2 / 8, A > 0, 'GR 3.821.12',
               ({A: Integer(1)}, {A: Rational(3, 2)}), oscillatory=True),
    TableEntry(sin(A * X)**4 / X**4, 0, oo, pi * A**3 / 3, A > 0, 'GR 3.821.13',
               ({A: Integer(1)}, {A: Integer(2)}), oscillatory=True),
    TableEntry(sin(A * X) * sin(B * X) / X**2, 0, oo, pi * Min(A, B) / 2, _positive(A, B), 'GR 3.741.3',
               ({A: Integer(1), B: Integer(3)}, {A: Integer(2), B: Rational(1, 2)}), oscillatory=True),
    TableEntry(sin(A * X) * cos(B * X) / X, 0, oo, pi / 2, (A > B) & (B > 0), 'GR 3.741.1',
               ({A: Integer(3), B: Integer(1)},), oscillatory=True),
    TableEntry(sin(A * X) * cos(B * X) / X, 0, oo, S.Zero, (B > A) & (A > 0), 'GR 3.741.1',
               ({A: Integer(1), B: Integer(3)},), oscillatory=True),
    TableEntry(sin(A * X) / (exp(X) - 1), 0, oo, pi * coth(pi * A) / 2 - 1 / (2 * A), A > 0, 'GR 3.911.1',
               ({A: Integer(1)}, {A: Rational(1, 2)})),
    TableEntry(sin(A * X) / (exp(X) + 1), 0, oo, 1 / (2 * A) - pi / (2 * sinh(pi * A)), A > 0, 'GR 3.911.3',
               ({A: Integer(1)}, {A: Integer(2)})),
    TableEntry(cos(A * X) / cosh(B * X), 0, oo, pi / (2 * B * cosh(pi * A / (2 * B))), _positive(A, B), 'GR 3.981.3',
               ({A: Integer(1), B: Integer(2)}, {A: Rational(1, 2), B: Integer(1)})),
    TableEntry(sin(A * X) / sinh(B * X), 0, oo, pi * tanh(pi * A / (2 * B)) / (2 * B), _positive(A, B), 'GR 3.981.1',
               ({A: Integer(1), B: Integer(2)}, {A: Integer(2), B: Integer(1)})),
    TableEntry(1 / (cosh(X) + cos(A)), 0, oo, A / sin(A), (A > 0) & (A < pi), 'GR 3.513.2',
               ({A: Integer(1)}, {A: Integer(2)})),
    TableEntry(1 / (cosh(X) + cos(A)), -oo, oo, 2 * A / sin(A), (A > 0) & (A < pi), 'GR 3.513.2',
               ({A: Integer(1)}, {A: Rational(5, 2)})),
    # ------------------------------------------------- logarithms, GR 4.2-4.3
    TableEntry(log(sin(X))**2, 0, pi / 2, pi * (log(2)**2 + pi**2 / 12) / 2, true, 'GR 4.224.7'),
    TableEntry(log(cos(X))**2, 0, pi / 2, pi * (log(2)**2 + pi**2 / 12) / 2, true, 'GR 4.224.7'),
    TableEntry(X * log(sin(X)), 0, pi, -pi**2 * log(2) / 2, true, 'GR 4.224.9'),
    TableEntry(log(1 + tan(X)), 0, pi / 4, pi * log(2) / 8, true, 'GR 4.225.?'),
    TableEntry(log(A**2 * cos(X)**2 + B**2 * sin(X)**2), 0, pi / 2, pi * log((A + B) / 2), _positive(A, B),
               'GR 4.225.5', ({A: Integer(2), B: Integer(3)}, {A: Integer(1), B: Rational(1, 2)})),
    TableEntry(log(1 - 2 * A * cos(X) + A**2), 0, pi, S.Zero, A**2 < 1, 'GR 4.224.14',
               ({A: Rational(1, 2)}, {A: -Rational(1, 3)})),
    TableEntry(log(1 - 2 * A * cos(X) + A**2), 0, pi, 2 * pi * log(A), A > 1, 'GR 4.224.14',
               ({A: Integer(2)}, {A: Integer(3)})),
    TableEntry(log(A + B * cos(X)), 0, pi, pi * log((A + sqrt(A**2 - B**2)) / 2), (A > 0) & (A**2 >= B**2),
               'GR 4.224.9', ({A: Integer(3), B: Integer(2)}, {A: Integer(2), B: -Integer(1)})),
    TableEntry(log(1 + X) / (1 + X**2), 0, 1, pi * log(2) / 8, true, 'GR 4.291.8'),
    TableEntry(log(X) / (1 + X**2), 0, 1, -S.Catalan, true, 'GR 4.231.12'),
    TableEntry(log(X) / (1 - X**2), 0, 1, -pi**2 / 8, true, 'GR 4.231.2'),
    TableEntry(log(X) / (X**2 - 1), 0, oo, pi**2 / 4, true, 'GR 4.231.1'),
    TableEntry(log(X)**2 / (1 + X), 0, 1, 3 * zeta(3) / 2, true, 'GR 4.261.4'),
    TableEntry(log(X) * log(1 - X), 0, 1, 2 - pi**2 / 6, true, 'GR 4.221.1'),
    TableEntry(log(X) * log(1 + X), 0, 1, 2 - pi**2 / 12 - 2 * log(2), true, 'GR 4.221.2'),
    TableEntry(loggamma(X), 0, 1, log(2 * pi) / 2, true, 'GR 6.441.1'),
    TableEntry(log(gamma(X)), 0, 1, log(2 * pi) / 2, true, 'GR 6.441.1'),
    # ------------------------------------------------- inverse trigonometric, GR 4.5
    TableEntry(atan(X) / X, 0, 1, S.Catalan, true, 'GR 4.531.1'),
    TableEntry(atan(X)**2 / X**2, 0, oo, pi * log(2), true, 'GR 4.535.1'),
    TableEntry(atan(A * X) * atan(B * X) / X**2, 0, oo,
               pi * ((A + B) * log(A + B) - A * log(A) - B * log(B)) / 2, _positive(A, B), 'GR 4.536.1',
               ({A: Integer(1), B: Integer(2)}, {A: Rational(1, 2), B: Integer(3)})),
    TableEntry(atan(A * X) / (X * (1 + X**2)), 0, oo, pi * log(1 + A) / 2, A > 0, 'GR 4.535.2',
               ({A: Integer(2)}, {A: Rational(1, 3)})),
    # ------------------------------------------------- Bessel functions, GR 6.5-6.6
    TableEntry(besselj(A, X), 0, oo, S.One, A > -1, 'GR 6.511.1', ({A: Integer(0)}, {A: Rational(3, 2)}),
               oscillatory=True),
    TableEntry(besselj(A, X)**2 / X, 0, oo, 1 / (2 * A), A > 0, 'GR 6.574.2 (mu = nu)',
               ({A: Integer(2)}, {A: Rational(1, 2)}), oscillatory=True),
    TableEntry(besselj(A, X) * besselj(B, X) / X, 0, oo,
               2 * sin(pi * (A - B) / 2) / (pi * (A**2 - B**2)), (A + B > 0) & Ne(A, B), 'GR 6.574.2',
               ({A: Integer(2), B: Integer(1)}, {A: Rational(5, 2), B: Rational(1, 2)}), oscillatory=True),
    TableEntry(besselj(0, A * X) * besselj(1, B * X), 0, oo, 1 / B, (B > A) & (A > 0), 'GR 6.512.3',
               ({A: Integer(1), B: Integer(2)},), oscillatory=True),
    TableEntry(besselj(0, A * X) * besselj(1, B * X), 0, oo, S.Zero, (A > B) & (B > 0), 'GR 6.512.3',
               ({A: Integer(3), B: Integer(2)},), oscillatory=True),
    TableEntry(besselj(0, A * X) * sin(B * X), 0, oo, 1 / sqrt(B**2 - A**2), (B > A) & (A > 0), 'GR 6.671.7',
               ({A: Integer(1), B: Integer(2)},), oscillatory=True),
    TableEntry(besselj(0, A * X) * sin(B * X), 0, oo, S.Zero, (A > B) & (B > 0), 'GR 6.671.7',
               ({A: Integer(3), B: Integer(1)},), oscillatory=True),
    TableEntry(besselj(0, A * X) * cos(B * X), 0, oo, 1 / sqrt(A**2 - B**2), (A > B) & (B > 0), 'GR 6.671.8',
               ({A: Integer(3), B: Integer(1)},), oscillatory=True),
    TableEntry(besselj(0, A * X) * cos(B * X), 0, oo, S.Zero, (B > A) & (A > 0), 'GR 6.671.8',
               ({A: Integer(1), B: Integer(2)},), oscillatory=True),
    # ------------------------------------------------- error functions, GR 6.28
    TableEntry(erfc(X)**2, 0, oo, (2 - sqrt(2)) / sqrt(pi), true, 'GR 6.283.?'),
    TableEntry(exp(-X**2) * erf(X), 0, oo, sqrt(pi) / 4, true, 'GR 6.285.?'),
    TableEntry(erf(A * X) * exp(-B**2 * X**2), 0, oo, atan(A / B) / (B * sqrt(pi)), _positive(A, B), 'GR 6.285.1',
               ({A: Integer(1), B: Integer(2)}, {A: Integer(3), B: Integer(1)})),
    # ------------------------------------------------- even powers of the sine, GR 3.621
    TableEntry(sin(X)**(2 * N), 0, pi / 2, pi * factorial(2 * N) / (2**(2 * N + 1) * factorial(N)**2),
               _natural(N), 'GR 3.621.3', ({N: Integer(2)}, {N: Integer(3)})),
    TableEntry(sin(X)**(2 * N), 0, pi, pi * factorial(2 * N) / (2**(2 * N) * factorial(N)**2),
               _natural(N), 'GR 3.621.3', ({N: Integer(2)}, {N: Integer(3)})),
    TableEntry(cos(X)**(2 * N), 0, pi, pi * factorial(2 * N) / (2**(2 * N) * factorial(N)**2),
               _natural(N), 'GR 3.621.3', ({N: Integer(2)}, {N: Integer(3)})),
]


def _forms(g: Expr) -> list[Expr]:
    """The integrand as given and in the rewritten forms the patterns
    may need."""
    forms = [g]
    for candidate in (as_expr(powsimp(g)), as_expr(g.expand()), as_expr(trigsimp(g))):
        if candidate not in forms:
            forms.append(candidate)
    return forms


def lookup(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
           assumptions: Assumptions = None) -> Optional[tuple[TableEntry, dict[Wild, Expr], Boolean]]:
    """The first entry of the table matching ``Integral(f, (x, a, b))``
    whose condition is not refuted by the assumptions, with the values
    of its parameters and the part of the condition left undecided;
    ``None`` when no entry matches.

    Examples
    ========

    >>> from sympy import symbols, sin, oo
    >>> from sympy_extras.integrals.tables import lookup
    >>> x, a = symbols('x a')
    >>> entry, match, condition = lookup(sin(a*x)**2/x**2, x, 0, oo)
    >>> entry.reference, condition
    ('GR 3.821.9', a > 0)
    >>> lookup(sin(x)**2/x**3, x, 0, oo) is None
    True
    """
    g = as_expr(f).xreplace({x: X})
    lower, upper = as_expr(a), as_expr(b)
    forms = _forms(as_expr(g))
    for entry in TABLE:
        if entry.lower != lower or entry.upper != upper:
            continue
        for candidate in forms:
            match = candidate.match(entry.pattern)
            if match is None:
                continue
            values: dict[Wild, Expr] = {}
            for key, value in match.items():
                if not isinstance(key, Wild):
                    continue
                value_ = as_expr(value)
                if value_.has(X):
                    break
                values[key] = value_
            else:
                condition = decide(as_boolean(entry.condition.xreplace(dict(values))), assumptions)
                if condition is None:
                    # refuted: another entry may hold (the other case of a
                    # case distinction)
                    continue
                return entry, values, condition
    return None


def table_integral(f: ExprLike, x: Symbol, a: ExprLike, b: ExprLike,
                   assumptions: Assumptions = None) -> Optional[ConditionalValue]:
    """``Integral(f, (x, a, b))`` from the table, with the condition on
    the parameters the assumptions do not settle; ``None`` when no entry
    matches.

    Examples
    ========

    >>> from sympy import symbols, cos, pi, log
    >>> from sympy_extras.integrals.tables import table_integral
    >>> x, a = symbols('x a')
    >>> table_integral(log(1 - 2*a*cos(x) + a**2), x, 0, pi, a > 1)
    ConditionalValue(2*pi*log(a))
    >>> table_integral(log(1 - 2*a*cos(x) + a**2), x, 0, pi, a < 1)
    ConditionalValue(0, a**2 < 1)
    """
    found = lookup(f, x, a, b, assumptions)
    if found is None:
        return None
    entry, values, condition = found
    return ConditionalValue(as_expr(entry.value.xreplace(dict(values))), condition)
