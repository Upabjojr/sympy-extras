"""Random numbers for tests: random complex numbers and the numerical
verification of identities and derivatives (``sympy.core.random``)."""
from __future__ import annotations

import random as _random
from typing import Optional, Sequence

from .expr import Expr, ExprLike, to_expr
from .numbers import Float, number_from_fraction, I
from .symbol import Symbol

__all__ = ['rng', 'seed', 'random', 'randint', 'uniform', 'choice', 'shuffle', 'sample',
    'random_complex_number', 'verify_numerically', 'test_derivative_numerically']

#: the generator used by this module
rng = _random.Random()
seed = rng.seed
random = rng.random
randint = rng.randint
uniform = rng.uniform
choice = rng.choice
shuffle = rng.shuffle
sample = rng.sample


def random_complex_number(a: float = 2, b: float = -1, c: float = 3, d: float = 1, rational: bool = False,
                          tolerance: Optional[float] = None) -> Expr:
    """A random complex number ``x + I*y`` with ``a <= x <= c`` and
    ``b <= y <= d`` (floats, or rationals when ``rational`` is set).

    Examples
    ========

    >>> from sympy_extras.core.random import random_complex_number, seed
    >>> seed(0)
    >>> z = random_complex_number()
    >>> z.is_number, z.has(Float)
    (True, True)
    """
    from fractions import Fraction
    from .add import add
    from .mul import mul
    x, y = uniform(a, c), uniform(b, d)
    if rational:
        denominator = 10 ** (tolerance_digits(tolerance))
        re = number_from_fraction(Fraction(round(x * denominator), denominator))
        im = number_from_fraction(Fraction(round(y * denominator), denominator))
        return add(re, mul(im, I))
    return add(Float(x), mul(Float(y), I))


def tolerance_digits(tolerance: Optional[float]) -> int:
    """The number of decimal digits of a tolerance (``6`` by default)."""
    if tolerance is None:
        return 6
    digits = 0
    while 10 ** (-digits) > tolerance and digits < 50:
        digits += 1
    return digits


def verify_numerically(f: ExprLike, g: ExprLike, z: Optional[Symbol] = None, tol: float = 1e-6,
                       a: float = 2, b: float = -1, c: float = 3, d: float = 1) -> bool:
    """Whether ``f`` and ``g`` agree at a random complex point for their
    symbols (``z``, or all the free symbols).

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.core.random import verify_numerically
    >>> x = Symbol('x')
    >>> verify_numerically((x + 1)**2, x**2 + 2*x + 1), verify_numerically((x + 1)**2, x**2 + 1)
    (True, False)
    """
    from .add import sub
    from .evalf import to_complex
    fe, ge = to_expr(f), to_expr(g)
    symbols: Sequence[Symbol] = [z] if z is not None else sorted(fe.free_symbols | ge.free_symbols, key=str)
    point = {s: random_complex_number(a, b, c, d) for s in symbols}
    difference = to_complex(sub(fe, ge).subs(point).evalf())
    return abs(difference) < tol


def test_derivative_numerically(f: ExprLike, z: Symbol, tol: float = 1e-6, a: float = 2, b: float = -1,
                                c: float = 3, d: float = 1) -> bool:
    """Whether the derivative of ``f`` in ``z`` agrees with a numerical
    derivative at a random complex point.

    Examples
    ========

    >>> from sympy_extras.core import Symbol
    >>> from sympy_extras.core.random import test_derivative_numerically
    >>> x = Symbol('x')
    >>> test_derivative_numerically(x**3, x)
    True
    """
    from .evalf import to_complex
    from .function import diff
    fe = to_expr(f)
    point = random_complex_number(a, b, c, d)
    symbolic = to_complex(diff(fe, z).subs(z, point).evalf())
    h = 1e-5
    centre = to_complex(point)
    forward = to_complex(fe.subs(z, _complex_expr(centre + h)).evalf(30))
    backward = to_complex(fe.subs(z, _complex_expr(centre - h)).evalf(30))
    numeric = (forward - backward) / (2 * h)
    return abs(symbolic - numeric) < tol * max(1.0, abs(symbolic))


def _complex_expr(t: complex) -> Expr:
    from .add import add
    from .mul import mul
    value = complex(t)
    return add(Float(value.real), mul(Float(value.imag), I))
